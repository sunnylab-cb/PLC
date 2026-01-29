"""
Training dynamics diagnostics for monitoring model health.

Implements tracking for:
- GVP output norms (scalar and vector features)
- Transformer gate openness (for gated attention)
- Attention entropy (attention distribution health)
- Module health checks
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math


def compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute entropy of attention distribution.
    
    Entropy = -sum(p * log(p))
    
    Higher entropy = more uniform attention (diffuse)
    Lower entropy = more focused attention (peaked)
    
    Healthy range: 1.5 - 3.5 for typical sequence lengths
    
    Args:
        attn_weights: Attention weights [B, H, L, S] after softmax
        eps: Small value for numerical stability
        
    Returns:
        Mean entropy across all heads and positions
    """
    # Clamp for numerical stability
    attn_weights = attn_weights.clamp(min=eps)
    
    # Compute entropy per position: -sum(p * log(p)) over keys
    entropy = -(attn_weights * attn_weights.log()).sum(dim=-1)  # [B, H, L]
    
    # Mean across all dimensions
    return entropy.mean()


def compute_gate_openness(gate_values: torch.Tensor) -> torch.Tensor:
    """
    Compute mean gate openness.
    
    For gated attention: gate = sigmoid(gate_score)
    
    Healthy range: 0.3 - 0.7
    - If ~1.0: Model ignoring residual connection (unstable)
    - If ~0.0: Attention layer is dead/useless
    
    Args:
        gate_values: Gate values after sigmoid [*, d_model]
        
    Returns:
        Mean gate openness
    """
    return gate_values.mean()


def compute_feature_norm(features: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute L2 norm of features.
    
    Args:
        features: Feature tensor
        dim: Dimension along which to compute norm
        
    Returns:
        Mean norm across all other dimensions
    """
    return features.norm(dim=dim).mean()


class DiagnosticsHook:
    """
    Hook manager for collecting training diagnostics from model modules.
    
    Usage:
        hook = DiagnosticsHook()
        hook.register(model)
        
        # Forward pass
        outputs = model(inputs)
        
        # Get diagnostics
        diagnostics = hook.get_diagnostics()
        
        # Clear for next iteration
        hook.clear()
    """
    
    def __init__(self):
        self.handles = []
        self.diagnostics = {}
        
    def register(self, model: nn.Module):
        """Register hooks on relevant model modules."""
        
        # Find GVP encoder
        if hasattr(model, 'gvp_encoder'):
            self._register_gvp_hook(model.gvp_encoder)
            
        # Find transformer blocks with gated attention
        if hasattr(model, 'protein_context_encoder'):
            self._register_transformer_hooks(model.protein_context_encoder)
            
        # Find cross-attention (support both old single layer and new multi-layer)
        if hasattr(model, 'cross_attention_layers'):
            for layer in model.cross_attention_layers:
                self._register_cross_attn_hook(layer)
        elif hasattr(model, 'cross_attention'):
            self._register_cross_attn_hook(model.cross_attention)
    
    def _register_gvp_hook(self, gvp_encoder):
        """Register forward hook on GVP encoder output."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                h_s, h_v = output
                self.diagnostics['gvp_scalar_norm'] = compute_feature_norm(h_s)
                self.diagnostics['gvp_vector_norm'] = compute_feature_norm(h_v, dim=-1).mean()
        
        handle = gvp_encoder.register_forward_hook(hook_fn)
        self.handles.append(handle)
        
    def _register_transformer_hooks(self, encoder):
        """Register hooks on transformer blocks."""
        gate_values = []
        attn_entropies = []
        
        # Recursively find attention modules
        for name, module in encoder.named_modules():
            # Look for gated attention modules
            if hasattr(module, '_last_gate_openness'):
                def make_gate_hook(m):
                    def hook_fn(mod, inp, out):
                        if hasattr(mod, '_last_gate_openness'):
                            gate_values.append(mod._last_gate_openness)
                    return hook_fn
                handle = module.register_forward_hook(make_gate_hook(module))
                self.handles.append(handle)
                
            if hasattr(module, '_last_attn_weights'):
                def make_entropy_hook(m):
                    def hook_fn(mod, inp, out):
                        if hasattr(mod, '_last_attn_weights'):
                            entropy = compute_attention_entropy(mod._last_attn_weights)
                            attn_entropies.append(entropy)
                    return hook_fn
                handle = module.register_forward_hook(make_entropy_hook(module))
                self.handles.append(handle)
        
        # Store references for aggregation
        self._gate_values = gate_values
        self._attn_entropies = attn_entropies
                
    def _register_cross_attn_hook(self, cross_attn):
        """Register hook on cross-attention module."""
        def hook_fn(module, input, output):
            if hasattr(module, 'attn') and hasattr(module.attn, '_last_attn_weights'):
                weights = module.attn._last_attn_weights
                self.diagnostics['cross_attn_entropy'] = compute_attention_entropy(weights)
        
        handle = cross_attn.register_forward_hook(hook_fn)
        self.handles.append(handle)
        
    def get_diagnostics(self) -> Dict[str, float]:
        """Get collected diagnostics as dictionary."""
        result = dict(self.diagnostics)
        
        # Aggregate gate values
        if hasattr(self, '_gate_values') and self._gate_values:
            result['gate_openness'] = torch.stack(self._gate_values).mean().item()
            
        # Aggregate attention entropies
        if hasattr(self, '_attn_entropies') and self._attn_entropies:
            result['attn_entropy'] = torch.stack(self._attn_entropies).mean().item()
            
        # Convert tensors to floats
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.item()
                
        return result
        
    def clear(self):
        """Clear collected diagnostics for next iteration."""
        self.diagnostics = {}
        if hasattr(self, '_gate_values'):
            self._gate_values.clear()
        if hasattr(self, '_attn_entropies'):
            self._attn_entropies.clear()
            
    def remove(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []


def collect_model_diagnostics(model: nn.Module) -> Dict[str, float]:
    """
    Collect diagnostics directly from model attributes.
    
    Alternative to hooks - uses cached values stored in modules.
    
    Args:
        model: The model after a forward pass
        
    Returns:
        Dict of diagnostic values
    """
    diagnostics = {}
    
    # GVP encoder
    if hasattr(model, 'gvp_encoder'):
        gvp = model.gvp_encoder
        if hasattr(gvp, '_last_scalar_norm'):
            diagnostics['gvp_scalar_norm'] = gvp._last_scalar_norm
        if hasattr(gvp, '_last_vector_norm'):
            diagnostics['gvp_vector_norm'] = gvp._last_vector_norm
    
    # Transformer gates and attention
    gate_values = []
    attn_entropies = []
    
    for name, module in model.named_modules():
        if hasattr(module, '_last_gate_openness'):
            val = module._last_gate_openness
            if isinstance(val, torch.Tensor):
                gate_values.append(val.item())
            else:
                gate_values.append(val)
                
        if hasattr(module, '_last_attn_entropy'):
            val = module._last_attn_entropy
            if isinstance(val, torch.Tensor):
                attn_entropies.append(val.item())
            else:
                attn_entropies.append(val)
    
    if gate_values:
        diagnostics['gate_openness'] = sum(gate_values) / len(gate_values)
    if attn_entropies:
        diagnostics['attn_entropy'] = sum(attn_entropies) / len(attn_entropies)
        
    return diagnostics


def check_health_warnings(diagnostics: Dict[str, float]) -> List[str]:
    """
    Check diagnostics for potential issues.
    
    Args:
        diagnostics: Dict of diagnostic values
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # GVP norms
    if 'gvp_scalar_norm' in diagnostics:
        norm = diagnostics['gvp_scalar_norm']
        if norm < 0.01:
            warnings.append(f"GVP scalar norm vanishing: {norm:.4f}")
        elif norm > 100:
            warnings.append(f"GVP scalar norm exploding: {norm:.4f}")
            
    if 'gvp_vector_norm' in diagnostics:
        norm = diagnostics['gvp_vector_norm']
        if norm < 0.001:
            warnings.append(f"GVP vector norm vanishing: {norm:.4f}")
        elif norm > 50:
            warnings.append(f"GVP vector norm exploding: {norm:.4f}")
    
    # Gate openness
    if 'gate_openness' in diagnostics:
        gate = diagnostics['gate_openness']
        if gate < 0.1:
            warnings.append(f"Gate nearly closed (attention dead?): {gate:.4f}")
        elif gate > 0.9:
            warnings.append(f"Gate nearly open (ignoring residual?): {gate:.4f}")
    
    # Attention entropy
    if 'attn_entropy' in diagnostics:
        entropy = diagnostics['attn_entropy']
        if entropy < 0.5:
            warnings.append(f"Attention entropy low (collapsing?): {entropy:.4f}")
        elif entropy > 5.0:
            warnings.append(f"Attention entropy high (too diffuse?): {entropy:.4f}")
    
    return warnings


def compute_feature_statistics(
    esm_embeddings: torch.Tensor,
    gvp_scalar: torch.Tensor,
    mask: torch.Tensor,
    esm_mask: Optional[torch.Tensor] = None,
    gvp_vector: Optional[torch.Tensor] = None,
    h_geo: Optional[torch.Tensor] = None,
    raw_esm_embeddings: Optional[torch.Tensor] = None,
    h_gvp_proj: Optional[torch.Tensor] = None,
    h_esm_proj: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute statistics about ESM and GVP features.
    
    Tracks:
    - ESM feature norm (raw and effective)
    - GVP scalar feature norm
    - GVP vector feature norm (if provided)
    - h_geo feature norm (final geometric embedding, if provided)
    - Projected feature norms (h_gvp_proj, h_esm_proj) for balance tracking
    - Relative contribution ratio (h_geo vs ESM, excluding dropped samples)
    - Modality dropout rate (if esm_mask provided)
    
    Args:
        esm_embeddings: [B, L, 1280] ESM features (after modality dropout)
        gvp_scalar: [B, L, 256] GVP scalar features
        mask: [B, L] padding mask
        esm_mask: [B, 1, 1] modality dropout mask (1=keep, 0=drop)
        gvp_vector: [B, L, 16, 3] GVP vector features (optional)
        h_geo: [B, L, 320] Final geometric embedding after bridge (optional)
        raw_esm_embeddings: [B, L, 1280] ESM features BEFORE dropout (optional)
        h_gvp_proj: [B, L, d_half] Projected GVP features (optional)
        h_esm_proj: [B, L, d_half] Projected ESM features (optional)
        
    Returns:
        Dict of feature statistics
    """
    stats = {}
    B, L, _ = esm_embeddings.shape
    device = esm_embeddings.device
    
    # Get valid features only (padding mask)
    valid_gvp_scalar = gvp_scalar[mask]  # [N_valid, 256]
    gvp_scalar_norm = valid_gvp_scalar.norm(dim=-1).mean().item()
    stats['gvp_scalar_feature_norm'] = gvp_scalar_norm
    
    # Vector norm (if provided)
    if gvp_vector is not None:
        valid_gvp_vector = gvp_vector[mask]  # [N_valid, 16, 3]
        gvp_vector_norm = valid_gvp_vector.norm(dim=-1).mean().item()
        stats['gvp_vector_feature_norm'] = gvp_vector_norm
    
    # h_geo norm (final geometric embedding, if provided)
    if h_geo is not None:
        valid_h_geo = h_geo[mask]  # [N_valid, 320]
        h_geo_norm = valid_h_geo.norm(dim=-1).mean().item()
        stats['h_geo_feature_norm'] = h_geo_norm
    
    # Projected feature norms (IMPORTANT for balance tracking)
    if h_gvp_proj is not None:
        valid_gvp_proj = h_gvp_proj[mask]
        gvp_proj_norm = valid_gvp_proj.norm(dim=-1).mean().item()
        stats['h_gvp_proj_norm'] = gvp_proj_norm
    
    if h_esm_proj is not None:
        # For ESM proj, we might want to consider modality dropout
        if esm_mask is not None:
            esm_batch_mask = esm_mask.squeeze(-1).squeeze(-1)
            esm_valid_mask = mask & esm_batch_mask.unsqueeze(1).bool()
            if esm_valid_mask.sum() > 0:
                valid_esm_proj = h_esm_proj[esm_valid_mask]
                esm_proj_norm = valid_esm_proj.norm(dim=-1).mean().item()
            else:
                esm_proj_norm = 0.0
        else:
            valid_esm_proj = h_esm_proj[mask]
            esm_proj_norm = valid_esm_proj.norm(dim=-1).mean().item()
        stats['h_esm_proj_norm'] = esm_proj_norm
        
        # Compute projected feature balance ratio
        if h_gvp_proj is not None and (gvp_proj_norm + esm_proj_norm) > 0:
            total_proj_norm = gvp_proj_norm + esm_proj_norm
            stats['proj_gvp_ratio'] = gvp_proj_norm / total_proj_norm
            stats['proj_esm_ratio'] = esm_proj_norm / total_proj_norm
    
    # ESM norm handling - need to account for modality dropout
    if esm_mask is not None:
        # esm_mask is [B, 1, 1], expand to [B, L] for per-residue masking
        esm_batch_mask = esm_mask.squeeze(-1).squeeze(-1)  # [B]
        
        # Create combined mask: padding AND not ESM-dropped
        # [B, 1] * [B, L] -> [B, L]
        esm_valid_mask = mask & esm_batch_mask.unsqueeze(1).bool()  # [B, L]
        
        # Effective ESM norm (only non-dropped samples)
        if esm_valid_mask.sum() > 0:
            valid_esm_effective = esm_embeddings[esm_valid_mask]
            esm_norm_effective = valid_esm_effective.norm(dim=-1).mean().item()
            stats['esm_feature_norm_effective'] = esm_norm_effective
        else:
            esm_norm_effective = 0.0
            stats['esm_feature_norm_effective'] = 0.0
        
        # Raw ESM norm (if provided - before dropout)
        if raw_esm_embeddings is not None:
            valid_esm_raw = raw_esm_embeddings[mask]
            esm_norm_raw = valid_esm_raw.norm(dim=-1).mean().item()
            stats['esm_feature_norm_raw'] = esm_norm_raw
        
        # Dropout rate
        dropout_rate = 1.0 - esm_mask.float().mean().item()
        stats['esm_dropout_rate'] = dropout_rate
        
        # Contribution ratio: use effective ESM norm (excluding dropped samples)
        # This gives true comparison of feature magnitudes when both are active
        esm_norm_for_ratio = esm_norm_effective
    else:
        # No modality dropout - use all samples
        valid_esm = esm_embeddings[mask]
        esm_norm = valid_esm.norm(dim=-1).mean().item()
        stats['esm_feature_norm'] = esm_norm
        esm_norm_for_ratio = esm_norm
    
    # Contribution ratio using h_geo (full structural pathway) vs ESM
    if h_geo is not None and esm_norm_for_ratio > 0:
        h_geo_norm_normalized = h_geo_norm / (320 ** 0.5)
        esm_norm_normalized = esm_norm_for_ratio / (1280 ** 0.5)
        
        total_norm = h_geo_norm_normalized + esm_norm_normalized
        if total_norm > 0:
            stats['h_geo_contribution_ratio'] = h_geo_norm_normalized / total_norm
            stats['esm_contribution_ratio'] = esm_norm_normalized / total_norm
    elif h_geo is None and esm_norm_for_ratio > 0:
        # Fallback: use just scalar if h_geo not available
        gvp_norm_normalized = gvp_scalar_norm / (256 ** 0.5)
        esm_norm_normalized = esm_norm_for_ratio / (1280 ** 0.5)
        
        total_norm = gvp_norm_normalized + esm_norm_normalized
        if total_norm > 0:
            stats['gvp_contribution_ratio'] = gvp_norm_normalized / total_norm
            stats['esm_contribution_ratio'] = esm_norm_normalized / total_norm
    
    return stats

