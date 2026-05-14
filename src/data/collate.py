"""
Batch collation utilities for variable-length proteins and ligands.

Handles padding and mask creation for efficient batching.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from torch_geometric.data import Batch, Data


def pocket_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for PocketDataset.
    
    Pads protein sequences and ligand atoms to create uniform batches.
    Creates attention masks to ignore padding during training.
    
    Args:
        batch: List of sample dictionaries from PocketDataset
        
    Returns:
        Batched dictionary with padded tensors and masks
    """
    if len(batch) == 0:
        return {}
    
    # Extract info
    pdb_ids = [sample['pdb_id'] for sample in batch]
    n_residues_list = [sample['n_residues'] for sample in batch]
    n_atoms_list = [sample['n_atoms'] for sample in batch]
    
    # Find max lengths in this batch
    max_residues = max(n_residues_list)
    max_atoms = max(n_atoms_list)
    
    B = len(batch)

    # Detect ESM embedding dimension from first sample
    esm_dim = batch[0]['esm'].shape[-1]

    # Detect protein scalar feature dimension from first sample
    protein_scalar_dim = batch[0]['protein_scalar'].shape[-1]

    # Initialize padded tensors
    esm_embeddings = torch.zeros(B, max_residues, esm_dim)
    protein_scalar = torch.zeros(B, max_residues, protein_scalar_dim)
    protein_coords = torch.zeros(B, max_residues, 3)
    protein_vectors = torch.zeros(B, max_residues, 4, 3)
    protein_mask = torch.zeros(B, max_residues, dtype=torch.bool)
    labels = torch.zeros(B, max_residues, dtype=torch.long)

    # Check if labels_original exists in any sample (V5: for comparison)
    has_labels_original = any('labels_original' in sample for sample in batch)
    if has_labels_original:
        labels_original = torch.zeros(B, max_residues, dtype=torch.long)

    # Detect ligand feature dimension from first sample
    ligand_dim = batch[0]['ligand_features'].shape[-1]
    ligand_features = torch.zeros(B, max_atoms, ligand_dim)
    ligand_coords = torch.zeros(B, max_atoms, 3)
    ligand_mask = torch.zeros(B, max_atoms, dtype=torch.bool)
    ligand_edge_index_list = []
    ligand_edge_attr_list = []

    # Fill in data
    for i, sample in enumerate(batch):
        n_res = sample['n_residues']
        n_atm = sample['n_atoms']

        esm_embeddings[i, :n_res] = sample['esm'][:n_res]
        protein_scalar[i, :n_res] = sample['protein_scalar'][:n_res]
        protein_coords[i, :n_res] = sample['gvp_coords'][:n_res]
        protein_vectors[i, :n_res] = sample['gvp_vectors'][:n_res]
        protein_mask[i, :n_res] = True
        labels[i, :n_res] = sample['labels'][:n_res]

        # V5: Fill labels_original if available
        if has_labels_original and 'labels_original' in sample:
            # Handle case where labels_original might be shorter than n_res
            labels_orig = sample['labels_original']
            n_orig = min(len(labels_orig), n_res)
            labels_original[i, :n_orig] = labels_orig[:n_orig]

        ligand_features[i, :n_atm] = sample['ligand_features'][:n_atm]
        ligand_coords[i, :n_atm] = sample['ligand_coords'][:n_atm]
        ligand_mask[i, :n_atm] = True
        ligand_edge_index_list.append(sample.get('ligand_edge_index', torch.zeros((2, 0), dtype=torch.long)))
        ligand_edge_attr_list.append(sample.get('ligand_edge_attr', torch.zeros((0, 6), dtype=torch.float32)))

    result = {
        'pdb_ids': pdb_ids,
        'esm_embeddings': esm_embeddings,
        'protein_scalar': protein_scalar,
        'protein_coords': protein_coords,
        'protein_vectors': protein_vectors,
        'protein_mask': protein_mask,
        'ligand_features': ligand_features,
        'ligand_coords': ligand_coords,
        'ligand_mask': ligand_mask,
        'ligand_edge_index_list': ligand_edge_index_list,
        'ligand_edge_attr_list': ligand_edge_attr_list,
        'labels': labels,
        'n_residues': torch.tensor(n_residues_list),
        'n_atoms': torch.tensor(n_atoms_list),
        'batch_size': B,
        'max_residues': max_residues,
        'max_atoms': max_atoms
    }

    # V5: Add labels_original if available
    if has_labels_original:
        result['labels_original'] = labels_original

    return result


def build_ligand_graphs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Build ligand molecular graphs for GINEConv from batched features.
    
    This constructs edge indices based on distance (assuming bonded atoms are close).
    For a proper implementation, edges should be pre-computed from RDKit molecules.
    
    Args:
        batch: Batched data dictionary
        
    Returns:
        Updated batch with ligand graph components:
            - ligand_x: [N_total, 64] node features
            - ligand_edge_index: [2, E_total] edge indices
            - ligand_edge_attr: [E_total, edge_dim] edge features
            - ligand_batch: [N_total] batch assignment
    """
    B = batch['batch_size']
    ligand_features = batch['ligand_features']  # [B, M, 64]
    ligand_coords = batch['ligand_coords']      # [B, M, 3]
    ligand_mask = batch['ligand_mask']          # [B, M]
    
    # Collect all atoms across batch
    all_nodes = []
    all_coords = []  # V6m6: Collect 3D coordinates
    all_edges_src = []
    all_edges_dst = []
    all_edge_attrs = []
    all_batch = []

    node_offset = 0

    for b in range(B):
        mask_b = ligand_mask[b]  # [M]
        n_atoms = mask_b.sum().item()

        if n_atoms == 0:
            continue

        # Get valid atoms
        nodes = ligand_features[b, mask_b]  # [n_atoms, 64]
        coords = ligand_coords[b, mask_b]   # [n_atoms, 3]

        all_nodes.append(nodes)
        all_coords.append(coords)  # V6m6: Store coordinates
        all_batch.extend([b] * n_atoms)
        
        # Build edges based on distance (< 2Å suggests bond)
        # This is a simple heuristic; proper edges should come from RDKit
        if n_atoms > 1:
            dist_matrix = torch.cdist(coords, coords)  # [n_atoms, n_atoms]
            
            # Connect atoms within bonding distance
            bond_mask = (dist_matrix < 2.0) & (dist_matrix > 0.1)
            src, dst = torch.where(bond_mask)
            
            # Offset by current node count
            all_edges_src.extend((src + node_offset).tolist())
            all_edges_dst.extend((dst + node_offset).tolist())
            
            # Simple edge features: distance
            for s, d in zip(src.tolist(), dst.tolist()):
                dist = dist_matrix[s, d].item()
                # [bond_type placeholder (4), is_close, distance_normalized]
                edge_feat = [1.0, 0.0, 0.0, 0.0, 1.0, dist / 2.0]
                all_edge_attrs.append(edge_feat)
        
        node_offset += n_atoms
    
    # Stack into tensors
    if len(all_nodes) > 0:
        ligand_x = torch.cat(all_nodes, dim=0)  # [N_total, 64]
        ligand_pos = torch.cat(all_coords, dim=0)  # [N_total, 3] - V6m6
        ligand_batch_idx = torch.tensor(all_batch, dtype=torch.long)
    else:
        ligand_x = torch.zeros(1, 64)
        ligand_pos = torch.zeros(1, 3)  # V6m6
        ligand_batch_idx = torch.zeros(1, dtype=torch.long)

    if len(all_edges_src) > 0:
        edge_index = torch.tensor([all_edges_src, all_edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(all_edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 6, dtype=torch.float32)

    # Add to batch
    batch['ligand_x'] = ligand_x
    batch['ligand_pos'] = ligand_pos  # V6m6: Add 3D coordinates
    batch['ligand_edge_index'] = edge_index
    batch['ligand_edge_attr'] = edge_attr
    batch['ligand_batch'] = ligand_batch_idx

    return batch


def prepare_model_inputs(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Prepare batch for model forward pass.
    
    Moves tensors to device and builds ligand graphs.
    
    Args:
        batch: Collated batch dictionary
        device: Target device
        
    Returns:
        Model-ready input dictionary
    """
    # Build ligand graphs
    batch = build_ligand_graphs(batch)
    
    # Move to device
    model_inputs = {
        'esm_embeddings': batch['esm_embeddings'].to(device),
        'protein_scalar': batch['protein_scalar'].to(device),
        'protein_vectors': batch['protein_vectors'].to(device),
        'protein_coords': batch['protein_coords'].to(device),
        'protein_mask': batch['protein_mask'].to(device),
        'ligand_x': batch['ligand_x'].to(device),
        'ligand_pos': batch['ligand_pos'].to(device),  # V6m6: 3D coordinates
        'ligand_edge_index': batch['ligand_edge_index'].to(device),
        'ligand_edge_attr': batch['ligand_edge_attr'].to(device),
        'ligand_batch': batch['ligand_batch'].to(device),
        'ligand_to_protein': batch['ligand_features'].to(device),  # Padded version for cross-attn
        'ligand_mask': batch['ligand_mask'].to(device),
    }

    return model_inputs


def compute_class_weights(
    dataloader,
    n_classes: int = 3,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Compute class weights from training data for balanced training.
    
    Args:
        dataloader: Training DataLoader
        n_classes: Number of classes
        device: Target device
        
    Returns:
        Class weights tensor [n_classes]
    """
    class_counts = torch.zeros(n_classes)
    
    for batch in dataloader:
        labels = batch['labels']
        mask = batch['protein_mask']
        
        for c in range(n_classes):
            class_counts[c] += ((labels == c) & mask).sum().item()
    
    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (n_classes * class_counts + 1e-6)
    
    # Normalize to mean of 1
    weights = weights / weights.mean()
    
    return weights.to(device)


def pocket_collate_fn_v8(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for PocketDatasetV7 (V3 features).

    V8 Changes:
    - Handles protein_physchem (62D)
    - Handles protein_spatial_scalar (72D)
    - Handles protein_spatial_vector (6×3D)
    - Keeps protein_vectors (4×3D) for GVP vectors

    Args:
        batch: List of sample dictionaries from PocketDatasetV7

    Returns:
        Batched dictionary with padded tensors and masks
    """
    if len(batch) == 0:
        return {}

    # Extract info
    pdb_ids = [sample['pdb_id'] for sample in batch]
    n_residues_list = [sample['n_residues'] for sample in batch]
    n_atoms_list = [sample['n_atoms'] for sample in batch]

    # Find max lengths in this batch
    max_residues = max(n_residues_list)
    max_atoms = max(n_atoms_list)

    B = len(batch)

    # Detect dimensions from first sample
    esm_dim = batch[0]['esm'].shape[-1]
    physchem_dim = batch[0]['protein_physchem'].shape[-1]  # V8: 62D
    spatial_scalar_dim = batch[0]['protein_spatial_scalar'].shape[-1]  # V8: 72D

    # Initialize padded tensors
    esm_embeddings = torch.zeros(B, max_residues, esm_dim)
    protein_physchem = torch.zeros(B, max_residues, physchem_dim)  # V8: 62D
    protein_spatial_scalar = torch.zeros(B, max_residues, spatial_scalar_dim)  # V8: 72D
    protein_spatial_vector = torch.zeros(B, max_residues, 6, 3)  # V8: 6×3D
    protein_coords = torch.zeros(B, max_residues, 3)
    protein_vectors = torch.zeros(B, max_residues, 4, 3)  # GVP vectors
    protein_mask = torch.zeros(B, max_residues, dtype=torch.bool)
    labels = torch.zeros(B, max_residues, dtype=torch.long)

    # Check if labels_original exists
    has_labels_original = any('labels_original' in sample for sample in batch)
    if has_labels_original:
        labels_original = torch.zeros(B, max_residues, dtype=torch.long)

    # Detect ligand feature dimension
    ligand_dim = batch[0]['ligand_features'].shape[-1]
    ligand_features = torch.zeros(B, max_atoms, ligand_dim)
    ligand_coords = torch.zeros(B, max_atoms, 3)
    ligand_mask = torch.zeros(B, max_atoms, dtype=torch.bool)
    ligand_edge_index_list = []
    ligand_edge_attr_list = []

    # Fill in data
    for i, sample in enumerate(batch):
        n_res = sample['n_residues']
        n_atm = sample['n_atoms']

        esm_embeddings[i, :n_res] = sample['esm'][:n_res]
        protein_physchem[i, :n_res] = sample['protein_physchem'][:n_res]  # V8
        protein_spatial_scalar[i, :n_res] = sample['protein_spatial_scalar'][:n_res]  # V8
        protein_spatial_vector[i, :n_res] = sample['protein_spatial_vector'][:n_res]  # V8
        protein_coords[i, :n_res] = sample['gvp_coords'][:n_res]
        protein_vectors[i, :n_res] = sample['gvp_vectors'][:n_res]
        protein_mask[i, :n_res] = True
        labels[i, :n_res] = sample['labels'][:n_res]

        if has_labels_original and 'labels_original' in sample:
            labels_orig = sample['labels_original']
            n_orig = min(len(labels_orig), n_res)
            labels_original[i, :n_orig] = labels_orig[:n_orig]

        ligand_features[i, :n_atm] = sample['ligand_features'][:n_atm]
        ligand_coords[i, :n_atm] = sample['ligand_coords'][:n_atm]
        ligand_mask[i, :n_atm] = True
        ligand_edge_index_list.append(sample.get('ligand_edge_index', torch.zeros((2, 0), dtype=torch.long)))
        ligand_edge_attr_list.append(sample.get('ligand_edge_attr', torch.zeros((0, 6), dtype=torch.float32)))

    result = {
        'pdb_ids': pdb_ids,
        'esm_embeddings': esm_embeddings,
        'protein_physchem': protein_physchem,  # V8
        'protein_spatial_scalar': protein_spatial_scalar,  # V8
        'protein_spatial_vector': protein_spatial_vector,  # V8
        'protein_coords': protein_coords,
        'protein_vectors': protein_vectors,  # GVP vectors
        'protein_mask': protein_mask,
        'ligand_features': ligand_features,
        'ligand_coords': ligand_coords,
        'ligand_mask': ligand_mask,
        'ligand_edge_index_list': ligand_edge_index_list,
        'ligand_edge_attr_list': ligand_edge_attr_list,
        'labels': labels,
        'n_residues': torch.tensor(n_residues_list),
        'n_atoms': torch.tensor(n_atoms_list),
    }

    if has_labels_original:
        result['labels_original'] = labels_original

    return result


def prepare_model_inputs_v8(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Prepare model inputs for V8 model from batched data.

    V8 Changes:
    - Adds protein_physchem (62D)
    - Adds protein_spatial_scalar (72D)
    - Adds protein_spatial_vector (6×3D)
    - Renames protein_vectors to protein_gvp_vectors for clarity

    Args:
        batch: Batched data from pocket_collate_fn_v8
        device: Target device

    Returns:
        Dictionary of model inputs ready for forward pass
    """
    # Build ligand graph from pre-computed bond graph if available.
    ligand_data_list = []
    B = len(batch['pdb_ids'])

    for i in range(B):
        n_atoms = batch['n_atoms'][i].item()
        if n_atoms == 0:
            continue

        x = batch['ligand_features'][i, :n_atoms]
        pos = batch['ligand_coords'][i, :n_atoms]
        if 'ligand_edge_index_list' in batch and 'ligand_edge_attr_list' in batch:
            edge_index = batch['ligand_edge_index_list'][i]
            edge_attr = batch['ligand_edge_attr_list'][i]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 6), dtype=torch.float32)

        ligand_data_list.append(Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr
        ))

    # Batch ligand graphs
    if len(ligand_data_list) > 0:
        ligand_batch_graph = Batch.from_data_list(ligand_data_list)
    else:
        # Empty batch fallback
        ligand_batch_graph = Data(
            x=torch.zeros((0, batch['ligand_features'].shape[-1])),
            pos=torch.zeros((0, 3)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 6)),
            batch=torch.zeros((0,), dtype=torch.long)
        )

    # Prepare model inputs
    model_inputs = {
        # Protein features
        'esm_embeddings': batch['esm_embeddings'].to(device),
        'protein_physchem': batch['protein_physchem'].to(device),  # V8: 62D
        'protein_spatial_scalar': batch['protein_spatial_scalar'].to(device),  # V8: 72D
        'protein_spatial_vector': batch['protein_spatial_vector'].to(device),  # V8: 6×3D
        'protein_gvp_vectors': batch['protein_vectors'].to(device),  # 4×3D GVP vectors
        'protein_coords': batch['protein_coords'].to(device),
        'protein_mask': batch['protein_mask'].to(device),

        # Ligand features (batched graph)
        'ligand_x': ligand_batch_graph.x.to(device),
        'ligand_pos': ligand_batch_graph.pos.to(device),
        'ligand_edge_index': ligand_batch_graph.edge_index.to(device),
        'ligand_edge_attr': ligand_batch_graph.edge_attr.to(device),
        'ligand_batch': ligand_batch_graph.batch.to(device),
        'ligand_to_protein': torch.zeros(B, batch['ligand_features'].shape[1], 512).to(device),  # Placeholder
        'ligand_mask': batch['ligand_mask'].to(device),
    }

    return model_inputs


def pocket_collate_fn_v10(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for protein complete features v4 + ligand features v5.

    Differences from v8:
    - protein_spatial_scalar dimension is inferred dynamically (v4 default: 89D)
    - protein_spatial_vector channel count is inferred dynamically (v4 default: 8x3)
    - ligand atom feature dimension is inferred dynamically (v5 default: 109D)
    """
    if len(batch) == 0:
        return {}

    pdb_ids = [sample['pdb_id'] for sample in batch]
    n_residues_list = [sample['n_residues'] for sample in batch]
    n_atoms_list = [sample['n_atoms'] for sample in batch]

    max_residues = max(n_residues_list)
    max_atoms = max(n_atoms_list)
    B = len(batch)

    esm_dim = batch[0]['esm'].shape[-1]
    physchem_dim = batch[0]['protein_physchem'].shape[-1]
    spatial_scalar_dim = batch[0]['protein_spatial_scalar'].shape[-1]
    spatial_vector_channels = batch[0]['protein_spatial_vector'].shape[1]
    ligand_dim = batch[0]['ligand_features'].shape[-1]

    esm_embeddings = torch.zeros(B, max_residues, esm_dim)
    protein_physchem = torch.zeros(B, max_residues, physchem_dim)
    protein_spatial_scalar = torch.zeros(B, max_residues, spatial_scalar_dim)
    protein_spatial_vector = torch.zeros(B, max_residues, spatial_vector_channels, 3)
    protein_rsa_raw  = torch.zeros(B, max_residues, 1)   # raw RSA [0,1] for SASA gate
    protein_rsa_mask = torch.zeros(B, max_residues, 1)   # 1=real RSA present, 0=absent
    protein_coords = torch.zeros(B, max_residues, 3)
    protein_vectors = torch.zeros(B, max_residues, 4, 3)
    protein_mask = torch.zeros(B, max_residues, dtype=torch.bool)
    chain_ids = torch.zeros(B, max_residues, dtype=torch.long)
    labels = torch.zeros(B, max_residues, dtype=torch.long)

    has_labels_original = any('labels_original' in sample for sample in batch)
    if has_labels_original:
        labels_original = torch.zeros(B, max_residues, dtype=torch.long)

    # V10m8: raw distances for distance regression auxiliary head
    has_raw_distances = any('raw_distances' in sample for sample in batch)
    if has_raw_distances:
        raw_distances = torch.full((B, max_residues), 999.0)  # pad with large distance

    ligand_features = torch.zeros(B, max_atoms, ligand_dim)
    ligand_coords = torch.zeros(B, max_atoms, 3)
    ligand_mask = torch.zeros(B, max_atoms, dtype=torch.bool)
    ligand_data_list = []  # PyG Data list for batched graph (assembled in worker)

    for i, sample in enumerate(batch):
        n_res = sample['n_residues']
        n_atm = sample['n_atoms']

        esm_embeddings[i, :n_res] = sample['esm'][:n_res]
        protein_physchem[i, :n_res] = sample['protein_physchem'][:n_res]
        protein_spatial_scalar[i, :n_res] = sample['protein_spatial_scalar'][:n_res]
        protein_spatial_vector[i, :n_res] = sample['protein_spatial_vector'][:n_res]
        if 'protein_rsa_raw' in sample:
            protein_rsa_raw[i, :n_res]  = sample['protein_rsa_raw'][:n_res]
        if 'protein_rsa_mask' in sample:
            protein_rsa_mask[i, :n_res] = sample['protein_rsa_mask'][:n_res]
        protein_coords[i, :n_res] = sample['gvp_coords'][:n_res]
        protein_vectors[i, :n_res] = sample['gvp_vectors'][:n_res]
        protein_mask[i, :n_res] = True
        if 'chain_ids' in sample:
            chain_ids[i, :n_res] = sample['chain_ids'][:n_res]
        labels[i, :n_res] = sample['labels'][:n_res]

        if has_labels_original and 'labels_original' in sample:
            labels_orig = sample['labels_original']
            n_orig = min(len(labels_orig), n_res)
            labels_original[i, :n_orig] = labels_orig[:n_orig]

        # V10m8: raw distances
        if has_raw_distances and 'raw_distances' in sample:
            rd = sample['raw_distances']
            n_rd = min(len(rd), n_res)
            raw_distances[i, :n_rd] = rd[:n_rd]

        ligand_features[i, :n_atm] = sample['ligand_features'][:n_atm]
        ligand_coords[i, :n_atm] = sample['ligand_coords'][:n_atm]
        ligand_mask[i, :n_atm] = True

        # Build per-sample PyG Data here (in DataLoader worker, not main thread).
        # Keep empty ligand slots so PyG batch indices stay aligned with B.
        if n_atm > 0:
            lig_x_i = sample['ligand_features'][:n_atm]
            lig_pos_i = sample['ligand_coords'][:n_atm]
        else:
            lig_x_i = torch.zeros((0, ligand_dim), dtype=sample['ligand_features'].dtype)
            lig_pos_i = torch.zeros((0, 3), dtype=sample['ligand_coords'].dtype)
        ligand_data_list.append(Data(
            x=lig_x_i,
            pos=lig_pos_i,
            edge_index=sample.get('ligand_edge_index', torch.zeros((2, 0), dtype=torch.long)),
            edge_attr=sample.get('ligand_edge_attr', torch.zeros((0, 6), dtype=torch.float32)),
        ))

    # Assemble batched ligand graph in worker (avoids main-thread Python loop overhead)
    if len(ligand_data_list) > 0:
        ligand_batch_graph = Batch.from_data_list(ligand_data_list)
        ligand_x = ligand_batch_graph.x
        ligand_pos = ligand_batch_graph.pos
        ligand_edge_index = ligand_batch_graph.edge_index
        ligand_edge_attr = ligand_batch_graph.edge_attr
        ligand_batch_idx = ligand_batch_graph.batch
    else:
        ligand_x = torch.zeros((0, ligand_dim))
        ligand_pos = torch.zeros((0, 3))
        ligand_edge_index = torch.zeros((2, 0), dtype=torch.long)
        ligand_edge_attr = torch.zeros((0, 6), dtype=torch.float32)
        ligand_batch_idx = torch.zeros((0,), dtype=torch.long)

    result = {
        'pdb_ids': pdb_ids,
        'esm_embeddings': esm_embeddings,
        'protein_physchem': protein_physchem,
        'protein_spatial_scalar': protein_spatial_scalar,
        'protein_spatial_vector': protein_spatial_vector,
        'protein_rsa_raw':  protein_rsa_raw,
        'protein_rsa_mask': protein_rsa_mask,
        'protein_coords': protein_coords,
        'protein_vectors': protein_vectors,
        'protein_mask': protein_mask,
        'chain_ids': chain_ids,
        'ligand_features': ligand_features,
        'ligand_coords': ligand_coords,
        'ligand_mask': ligand_mask,
        'ligand_x': ligand_x,
        'ligand_pos': ligand_pos,
        'ligand_edge_index': ligand_edge_index,
        'ligand_edge_attr': ligand_edge_attr,
        'ligand_batch': ligand_batch_idx,
        'labels': labels,
        'n_residues': torch.tensor(n_residues_list),
        'n_atoms': torch.tensor(n_atoms_list),
    }

    if has_labels_original:
        result['labels_original'] = labels_original

    if has_raw_distances:
        result['raw_distances'] = raw_distances

    return result


def prepare_model_inputs_v10(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Prepare model inputs for v10-compatible models.

    Ligand graph (ligand_x/pos/edge_index/edge_attr/batch) is pre-assembled
    in pocket_collate_fn_v10 (DataLoader workers) and forwarded here directly.
    The legacy ligand_to_protein placeholder is kept for older v10/v10m*
    models and scripts that still read it for shape information.
    """
    B, max_atoms = batch['ligand_mask'].shape
    model_inputs = {
        'esm_embeddings': batch['esm_embeddings'].to(device, non_blocking=non_blocking),
        'protein_physchem': batch['protein_physchem'].to(device, non_blocking=non_blocking),
        'protein_spatial_scalar': batch['protein_spatial_scalar'].to(device, non_blocking=non_blocking),
        'protein_spatial_vector': batch['protein_spatial_vector'].to(device, non_blocking=non_blocking),
        'protein_gvp_vectors': batch['protein_vectors'].to(device, non_blocking=non_blocking),
        'protein_coords': batch['protein_coords'].to(device, non_blocking=non_blocking),
        'protein_mask': batch['protein_mask'].to(device, non_blocking=non_blocking),
        'chain_ids': batch['chain_ids'].to(device, non_blocking=non_blocking),
        'ligand_x': batch['ligand_x'].to(device, non_blocking=non_blocking),
        'ligand_pos': batch['ligand_pos'].to(device, non_blocking=non_blocking),
        'ligand_edge_index': batch['ligand_edge_index'].to(device, non_blocking=non_blocking),
        'ligand_edge_attr': batch['ligand_edge_attr'].to(device, non_blocking=non_blocking),
        'ligand_batch': batch['ligand_batch'].to(device, non_blocking=non_blocking),
        'ligand_to_protein': torch.zeros(B, max_atoms, 512).to(device, non_blocking=non_blocking),
        'ligand_mask': batch['ligand_mask'].to(device, non_blocking=non_blocking),
    }
    return model_inputs


def prepare_model_inputs_v11(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Prepare model inputs for v11-family models.

    This intentionally omits the legacy ligand_to_protein placeholder.  V11
    reads ligand batch shape from ligand_mask and consumes the flat PyG ligand
    graph produced by pocket_collate_fn_v10.
    """
    model_inputs = {
        'esm_embeddings': batch['esm_embeddings'].to(device, non_blocking=non_blocking),
        'protein_physchem': batch['protein_physchem'].to(device, non_blocking=non_blocking),
        'protein_spatial_scalar': batch['protein_spatial_scalar'].to(device, non_blocking=non_blocking),
        'protein_spatial_vector': batch['protein_spatial_vector'].to(device, non_blocking=non_blocking),
        'protein_gvp_vectors': batch['protein_vectors'].to(device, non_blocking=non_blocking),
        'protein_coords': batch['protein_coords'].to(device, non_blocking=non_blocking),
        'protein_mask': batch['protein_mask'].to(device, non_blocking=non_blocking),
        'chain_ids': batch['chain_ids'].to(device, non_blocking=non_blocking),
        'ligand_x': batch['ligand_x'].to(device, non_blocking=non_blocking),
        'ligand_pos': batch['ligand_pos'].to(device, non_blocking=non_blocking),
        'ligand_edge_index': batch['ligand_edge_index'].to(device, non_blocking=non_blocking),
        'ligand_edge_attr': batch['ligand_edge_attr'].to(device, non_blocking=non_blocking),
        'ligand_batch': batch['ligand_batch'].to(device, non_blocking=non_blocking),
        'ligand_mask': batch['ligand_mask'].to(device, non_blocking=non_blocking),
    }
    return model_inputs


def prepare_model_inputs_v12(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = True,
    unimol_embed: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Prepare model inputs for V12 (Interleaved Co-Attention + Uni-Mol FiLM).

    Adds protein_rsa_raw  [B, L, 1] raw RSA values for SASA residual gate.
    Adds protein_rsa_mask [B, L, 1] validity mask (1=real RSA, 0=absent/old .pt).
    Adds unimol_embed     [B, 512]  for FiLM conditioning (may be None).
    """
    return {
        'esm_embeddings':         batch['esm_embeddings'].to(device, non_blocking=non_blocking),
        'protein_physchem':       batch['protein_physchem'].to(device, non_blocking=non_blocking),
        'protein_spatial_scalar': batch['protein_spatial_scalar'].to(device, non_blocking=non_blocking),
        'protein_spatial_vector': batch['protein_spatial_vector'].to(device, non_blocking=non_blocking),
        'protein_gvp_vectors':    batch['protein_vectors'].to(device, non_blocking=non_blocking),
        'protein_coords':         batch['protein_coords'].to(device, non_blocking=non_blocking),
        'protein_mask':           batch['protein_mask'].to(device, non_blocking=non_blocking),
        'protein_rsa_raw':        batch['protein_rsa_raw'].to(device, non_blocking=non_blocking),
        'protein_rsa_mask':       batch['protein_rsa_mask'].to(device, non_blocking=non_blocking),
        'chain_ids':              batch['chain_ids'].to(device, non_blocking=non_blocking),
        'ligand_x':               batch['ligand_x'].to(device, non_blocking=non_blocking),
        'ligand_pos':             batch['ligand_pos'].to(device, non_blocking=non_blocking),
        'ligand_edge_index':      batch['ligand_edge_index'].to(device, non_blocking=non_blocking),
        'ligand_edge_attr':       batch['ligand_edge_attr'].to(device, non_blocking=non_blocking),
        'ligand_batch':           batch['ligand_batch'].to(device, non_blocking=non_blocking),
        'ligand_mask':            batch['ligand_mask'].to(device, non_blocking=non_blocking),
        'unimol_embed':           unimol_embed.to(device, non_blocking=non_blocking) if unimol_embed is not None else None,
    }


def pocket_collate_fn_auto(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Automatically dispatch to the appropriate collate function.

    Routing:
    - legacy samples with `protein_scalar` -> `pocket_collate_fn`
    - v8/v9 samples with `protein_physchem` + 72D scalar + 6x3 vector + <=96D ligand -> `pocket_collate_fn_v8`
    - newer samples (e.g. complete_features_v4 / ligand_features_v5) -> `pocket_collate_fn_v10`
    """
    if len(batch) == 0:
        return {}

    sample = batch[0]
    if 'protein_scalar' in sample:
        return pocket_collate_fn(batch)

    if 'protein_physchem' not in sample or 'protein_spatial_scalar' not in sample or 'protein_spatial_vector' not in sample:
        raise KeyError("Unknown batch schema for automatic collate dispatch")

    spatial_scalar_dim = int(sample['protein_spatial_scalar'].shape[-1])
    spatial_vector_channels = int(sample['protein_spatial_vector'].shape[1])
    ligand_dim = int(sample['ligand_features'].shape[-1])

    if spatial_scalar_dim == 72 and spatial_vector_channels == 6 and ligand_dim <= 96:
        return pocket_collate_fn_v8(batch)

    return pocket_collate_fn_v10(batch)


def prepare_model_inputs_auto(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Automatically dispatch to the appropriate model-input preparation function.
    """
    if 'protein_scalar' in batch:
        return prepare_model_inputs(batch, device)

    if 'protein_physchem' not in batch or 'protein_spatial_scalar' not in batch or 'protein_spatial_vector' not in batch:
        raise KeyError("Unknown batched schema for automatic input preparation")

    spatial_scalar_dim = int(batch['protein_spatial_scalar'].shape[-1])
    spatial_vector_channels = int(batch['protein_spatial_vector'].shape[2])
    ligand_dim = int(batch['ligand_features'].shape[-1])

    if spatial_scalar_dim == 72 and spatial_vector_channels == 6 and ligand_dim <= 96:
        return prepare_model_inputs_v8(batch, device)

    return prepare_model_inputs_v10(batch, device)
