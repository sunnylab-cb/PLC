"""Shared local spatial operators followed by ligand-atom readout.

No ligand pose, target mask, protein pooling or global protein attention is
used. ESM and cached descriptors retain their original extraction context.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .gvp_layers import GVP, Dropout, rbf_encode
from .ligand_encoder_3d import Ligand3DEncoder


def vector_norm(v):
    return (v.float().square().sum(-1) + 1e-8).sqrt().to(v.dtype)


def normalize_vectors(v):
    scale = (v.float().square().sum(-1).mean(-1, keepdim=True) + 1e-8).sqrt()
    return v / scale.clamp_min(1.0).to(v.dtype).unsqueeze(-1)


@torch.no_grad()
def local_neighbors(coords, batch_ids, chains, positions, radius=12.0, k=32, chunk=128):
    """Center + k-1 neighbors; exact distance ties use ascending input index.

    Non-finite coordinates cannot send/receive geometry messages. They remain
    scored by the pointwise stem/readout. Offsets are row-order offsets within
    a chain (not PDB residue numbers); never offsets across different chains.
    """
    total = len(coords)
    indices = torch.arange(total, device=coords.device)[:, None].expand(-1, k).clone()
    valid = torch.zeros(total, k, dtype=torch.bool, device=coords.device)
    capped = torch.zeros(total, dtype=torch.bool, device=coords.device)
    finite = torch.isfinite(coords).all(-1)
    xyz = torch.where(finite[:, None], coords.float(), 0.0)
    for batch_id in batch_ids.unique():
        rows = torch.where(batch_ids == batch_id)[0]
        for centers in rows.split(chunk):
            dist = torch.cdist(xyz[centers], xyz[rows], compute_mode="donot_use_mm_for_euclid_dist")
            eligible = finite[centers, None] & finite[rows][None] & (centers[:, None] != rows[None])
            dist = dist.masked_fill(~eligible, float("inf"))
            capped[centers] = (dist <= radius).sum(-1) > k - 1
            count = min(k - 1, max(len(rows) - 1, 0))
            valid[centers, 0] = finite[centers]
            if count:
                nearest = dist.argsort(dim=-1, stable=True)[:, :count]
                indices[centers, 1:count + 1] = rows[nearest]
                valid[centers, 1:count + 1] = dist.gather(1, nearest) <= radius
    delta = xyz[indices] - xyz[:, None]
    dist = delta.norm(dim=-1)
    same = chains[:, None] == chains[indices]
    offset = (positions[:, None] - positions[indices]).abs()
    offsets = torch.stack((offset == 1, (offset >= 2) & (offset <= 5),
                           (offset >= 6) & (offset <= 12), offset > 12), -1)
    edge_s = torch.cat((rbf_encode(dist, D_max=20.0, num_rbf=16),
                        (offsets & same[..., None]).float(), same[..., None].float()), -1)
    edge_s = edge_s * valid[..., None]
    edge_v = F.normalize(delta, dim=-1) * valid[..., None]
    others = valid[:, 1:]
    stats = {
        "scan_neighbors_mean": others.sum().float() / max(total, 1),
        "scan_cap_hit_fraction": capped.sum().float() / max(total, 1),
        "scan_isolated_fraction": (~others.any(-1)).sum().float() / max(total, 1),
        "scan_missing_coords_fraction": (~finite).sum().float() / max(total, 1),
        "scan_cross_chain_edge_fraction": ((~same[:, 1:]) & others).sum().float() / others.sum().clamp_min(1),
    }
    return indices, valid, edge_s, edge_v, xyz, stats


class BoundedConditioner(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context):
        return 0.2 * self.net(context).tanh()


class LocalGVPBlock(nn.Module):
    def __init__(self, conditioned, dropout):
        super().__init__()
        self.message = nn.ModuleList((GVP((533, 33), (256, 16)), GVP((256, 16), (256, 16))))
        self.update = GVP((512, 32), (256, 16), activations=(None, None))
        self.norm = nn.LayerNorm(256)
        self.drop = Dropout(dropout)
        self.conditioner = BoundedConditioner(528) if conditioned else None

    def forward(self, h, v, centers, neighbors, valid, edge_s, edge_v, context):
        n = neighbors.shape[1]
        ms = torch.cat((h[centers, None].expand(-1, n, -1), h[neighbors], edge_s), -1)
        mv = torch.cat((v[centers, None].expand(-1, n, -1, -1), v[neighbors], edge_v[..., None, :]), -2)
        for layer in self.message:
            ms, mv = layer(ms, mv)
        if self.conditioner is not None:
            gs, bs, gv = self.conditioner(context).split((256, 256, 16), -1)
            ms = (1 + gs[:, None]) * ms + bs[:, None]
            mv = (1 + gv[:, None, :, None]) * mv
        ms, mv = self.drop(ms, mv)
        denom = valid.sum(-1).clamp_min(1)
        ms = (ms * valid[..., None]).sum(1) / denom[:, None]
        mv = (mv * valid[..., None, None]).sum(1) / denom[:, None, None]
        ds, dv = self.update(torch.cat((h[centers], ms), -1), torch.cat((v[centers], mv), -2))
        ds, dv = self.drop(ds, dv)
        active = valid.any(-1)
        return (torch.where(active[:, None], self.norm(h[centers] + ds), h[centers]),
                torch.where(active[:, None, None], normalize_vectors(v[centers] + dv), v[centers]))


class LocalTriangleBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(565, 64), nn.GELU(), nn.LayerNorm(64))
        self.conditioner = BoundedConditioner(128)
        self.qkv = nn.Linear(64, 192)
        self.bias = nn.Linear(16, 4, bias=False)
        self.out = nn.Linear(64, 64)
        self.pair_norm = nn.LayerNorm(64)
        self.pool = nn.Linear(64, 256)
        self.norm = nn.LayerNorm(256)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, v, centers, neighbors, valid, edge_s, neighbor_xyz, context):
        count, n = neighbors.shape
        inv = vector_norm(v)
        z = self.pair(torch.cat((h[centers, None].expand(-1, n, -1), h[neighbors],
                                inv[centers, None].expand(-1, n, -1), inv[neighbors], edge_s), -1))
        gain, shift = self.conditioner(context).chunk(2, -1)
        z = (1 + gain[:, None]) * z + shift[:, None]
        q, key, value = self.qkv(z).reshape(count, n, 3, 4, 16).unbind(2)
        q, key, value = (t.transpose(1, 2) for t in (q, key, value))
        distances = (neighbor_xyz[:, :, None] - neighbor_xyz[:, None, :]).float().norm(dim=-1)
        bias = self.bias(rbf_encode(distances, D_max=24.0, num_rbf=16)).permute(0, 3, 1, 2)
        scores = (q.float() @ key.float().transpose(-2, -1)) / 4.0 + bias.float()
        # Finite sentinel plus explicit mask also handles a center with no keys.
        weights = scores.masked_fill(~valid[:, None, None], -1e9).softmax(-1)
        weights = weights * valid[:, None, None]
        read = (weights.to(value.dtype) @ value).transpose(1, 2).reshape(count, n, 64)
        z = self.pair_norm(z + self.drop(self.out(read)))
        pooled = (z * valid[..., None]).sum(1) / valid.sum(-1).clamp_min(1)[:, None]
        updated = self.norm(h[centers] + self.drop(self.pool(pooled)))
        return torch.where(valid.any(-1)[:, None], updated, h[centers]), v[centers]


class PocketLocalSpatialScan(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.variant = config["scan_variant"]
        if self.variant not in ("S0_local_gvp", "S1_cond_gvp", "T1_cond_triangle"):
            raise ValueError(self.variant)
        self.chunk = int(config.get("scan_chunk", 128))
        self.checkpoint = config.get("use_gradient_checkpointing", True)
        p = config.get("dropout", 0.15)
        self.esm_proj = nn.Sequential(nn.Linear(config.get("d_esm", 1280), 128), nn.LayerNorm(128))
        self.physchem_encoder = nn.Sequential(nn.Linear(62, 32), nn.GELU(), nn.LayerNorm(32))
        self.spatial_proj = nn.Sequential(nn.Linear(89, 96), nn.GELU(), nn.LayerNorm(96))
        self.vector_proj = nn.Linear(12, 16, bias=False)
        self.ligand_encoder = Ligand3DEncoder(node_in_dim=109, node_h_dim=128, out_dim=128,
            edge_in_dim=6 + config.get("ligand_edge_rbf_bins", 8), n_layers=3, dropout=p)
        self.local54_proj = nn.Sequential(nn.Linear(54, 128), nn.LayerNorm(128))
        self.ligand_merge = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128))
        self.ligand_pool = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128))
        self.spatial_blocks = nn.ModuleList(
            LocalTriangleBlock(p) if self.variant == "T1_cond_triangle" else
            LocalGVPBlock(self.variant == "S1_cond_gvp", p) for _ in range(2))
        self.atom_q = nn.Linear(256, 128)
        self.atom_k = nn.Linear(128, 128)
        self.atom_v = nn.Linear(128, 128)
        self.classifier = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 128), nn.GELU(),
                                        nn.Dropout(config.get("classifier_dropout", 0.2)), nn.Linear(128, 3))
        self.dist_head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1), nn.Softplus())

    def _drop_source(self, x, probability):
        if not self.training or probability <= 0:
            return x
        keep = (torch.rand((len(x),) + (1,) * (x.ndim - 1), device=x.device) >= probability)
        return x * keep

    def _run(self, fn, *args):
        if self.training and self.checkpoint and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _ligand(self, x, pos, edges, edge_attr, batch_ids, mask):
        b, m = mask.shape
        raw = x.new_zeros(b, m, 54)
        graph = x.new_zeros(b, m, 128)
        if len(x):
            if not bool((batch_ids[1:] >= batch_ids[:-1]).all()):
                raise ValueError("Flat ligand atoms must be contiguous by sample")
            counts = torch.bincount(batch_ids, minlength=b)
            if bool((counts > m).any()):
                raise ValueError("Ligand mask is smaller than the flat graph")
            offset = F.pad(counts.cumsum(0), (1, 0))
            local = torch.arange(len(x), device=x.device) - offset[batch_ids]
            d = (pos[edges[0]] - pos[edges[1]]).float().norm(dim=-1)
            attr = torch.cat((edge_attr, rbf_encode(d, D_max=5.0,
                             num_rbf=self.config.get("ligand_edge_rbf_bins", 8))), -1)
            encoded, _ = self._run(self.ligand_encoder, x, pos, edges, attr, batch_ids)
            graph = graph.to(encoded.dtype)
            graph[batch_ids, local] = encoded
            raw[batch_ids, local] = x[:, :54]
        if self.training:
            keep = (torch.rand(b, 1, 1, device=x.device) >= self.config.get("ligand_source_dropout_p", 0.03))
            graph = self._drop_source(graph * keep, self.config.get("ligand_dropout_hidden_p", 0.04))
            raw = self._drop_source(raw * keep, self.config.get("ligand_dropout_local54_p", 0.04))
        z = self.ligand_merge(torch.cat((graph, self.local54_proj(raw)), -1))
        z = z * mask[..., None]
        mean = z.sum(1) / mask.sum(1).clamp_min(1)[:, None]
        maximum = z.masked_fill(~mask[..., None], -1e4).amax(1) if m else mean
        maximum = torch.where(mask.any(-1)[:, None], maximum, 0.0)
        context = self.ligand_pool(torch.cat((mean, maximum), -1)) * mask.any(-1)[:, None]
        return z, context

    def _read(self, h, z, ligand_mask, context):
        q, key, value = self.atom_q(h), self.atom_k(z), self.atom_v(z)
        if z.shape[1]:
            scores = (q.float() @ key.float().transpose(-2, -1)) / (128 ** 0.5)
            weights = scores.masked_fill(~ligand_mask[:, None], -1e9).softmax(-1)
            weights = weights * ligand_mask[:, None]
            read = weights.to(value.dtype) @ value
        else:
            read = q * 0
        x = torch.cat((h, read, context[:, None].expand(-1, h.shape[1], -1)), -1)
        return self.classifier(x), self.dist_head(x).squeeze(-1)

    def forward(self, esm_embeddings, protein_physchem, protein_spatial_scalar,
                protein_spatial_vector, protein_gvp_vectors, protein_coords, protein_mask,
                ligand_x, ligand_pos, ligand_edge_index, ligand_edge_attr, ligand_batch,
                ligand_mask, chain_ids=None, esm_valid_mask=None, residue_positions=None,
                unimol_embed=None, esm3_embeddings=None, esm3_valid_mask=None, ligand_to_protein=None):
        mask = protein_mask.bool()
        b, n = mask.shape
        esm = self.esm_proj(esm_embeddings)
        if esm_valid_mask is not None:
            esm = esm * esm_valid_mask[..., None]
        esm = self._drop_source(esm, self.config.get("esm_dropout_prob", 0.05))
        phys = self._drop_source(self.physchem_encoder(protein_physchem), self.config.get("physchem_dropout_prob", 0.05))
        geo = self.spatial_proj(protein_spatial_scalar)
        vectors = torch.cat((protein_gvp_vectors, protein_spatial_vector), -2)
        v = normalize_vectors(self.vector_proj(vectors.transpose(-1, -2)).transpose(-1, -2))
        if self.training:
            keep = self._drop_source(torch.ones(b, 1, 1, device=geo.device), self.config.get("gvp_dropout_prob", 0.05))
            geo, v = geo * keep, v * keep[..., None]
        h = torch.cat((esm, phys, geo), -1)[mask]
        v = v[mask]
        batch_ids = torch.arange(b, device=mask.device)[:, None].expand(b, n)[mask]
        coords = protein_coords[mask]
        if self.training and self.config.get("coord_noise_std", 0.2):
            coords = coords + torch.randn_like(coords) * self.config.get("coord_noise_std", 0.2)
        chains = torch.zeros_like(protein_mask, dtype=torch.long) if chain_ids is None else chain_ids
        positions = torch.arange(n, device=mask.device)[None].expand(b, n) if residue_positions is None else residue_positions
        indices, valid, es, ev, xyz, stats = local_neighbors(coords, batch_ids, chains[mask], positions[mask],
            radius=self.config.get("scan_radius", 12.0), k=self.config.get("scan_k", 32), chunk=self.chunk)
        z, context = self._ligand(ligand_x, ligand_pos, ligand_edge_index, ligand_edge_attr,
                                  ligand_batch, ligand_mask.bool())
        centers = torch.arange(len(h), device=h.device)
        for layer in self.spatial_blocks:
            hs, vs = [], []
            for c in centers.split(self.chunk):
                geometry = xyz[indices[c]] if self.variant == "T1_cond_triangle" else ev[c]
                hc, vc = self._run(layer, h, v, c, indices[c], valid[c], es[c], geometry, context[batch_ids[c]])
                hs.append(hc)
                vs.append(vc)
            if hs:
                h, v = torch.cat(hs), torch.cat(vs)
        padded = h.new_zeros(b, n, 256)
        padded[mask] = h
        logits, distances = [], []
        for start in range(0, n, self.chunk):
            lc, dc = self._run(self._read, padded[:, start:start + self.chunk], z, ligand_mask.bool(), context)
            logits.append(lc)
            distances.append(dc)
        return {"logits": torch.cat(logits, 1) * mask[..., None],
                "pred_dist": torch.cat(distances, 1) * mask, **{k: val.detach() for k, val in stats.items()}}
