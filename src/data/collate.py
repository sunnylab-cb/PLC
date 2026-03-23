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
    # Build ligand graph
    ligand_data_list = []
    B = len(batch['pdb_ids'])

    for i in range(B):
        n_atoms = batch['n_atoms'][i].item()
        if n_atoms == 0:
            continue

        x = batch['ligand_features'][i, :n_atoms]
        pos = batch['ligand_coords'][i, :n_atoms]

        # Build edges (fully connected within ligand)
        edge_index = []
        edge_attr = []
        for src in range(n_atoms):
            for dst in range(n_atoms):
                if src != dst:
                    edge_index.append([src, dst])
                    # Edge features: [bond_type (4), is_close, distance_normalized]
                    # 与原始 build_ligand_graphs 保持一致
                    vec = pos[dst] - pos[src]
                    dist = torch.norm(vec).item()
                    edge_feat = [1.0, 0.0, 0.0, 0.0, 1.0, dist / 2.0]
                    edge_attr.append(edge_feat)

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # V8: 使用 tensor 而不是 stack
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

