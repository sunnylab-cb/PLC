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
    h_geo: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute statistics about ESM and GVP features.
    
    Tracks:
    - ESM feature norm
    - GVP scalar feature norm
    - GVP vector feature norm (if provided)
    - h_geo feature norm (final geometric embedding, if provided)
    - Relative contribution ratio (h_geo vs ESM)
    - Modality dropout rate (if esm_mask provided)
    
    Args:
        esm_embeddings: [B, L, 1280] ESM features (possibly masked)
        gvp_scalar: [B, L, 256] GVP scalar features
        mask: [B, L] padding mask
        esm_mask: [B, 1, 1] modality dropout mask (1=keep, 0=drop)
        gvp_vector: [B, L, 16, 3] GVP vector features (optional)
        h_geo: [B, L, 320] Final geometric embedding after bridge (optional)
        
    Returns:
        Dict of feature statistics
    """
    stats = {}
    
    # Get valid features only
    valid_esm = esm_embeddings[mask]  # [N_valid, 1280]
    valid_gvp_scalar = gvp_scalar[mask]  # [N_valid, 256]
    
    # Feature norms
    esm_norm = valid_esm.norm(dim=-1).mean().item()
    gvp_scalar_norm = valid_gvp_scalar.norm(dim=-1).mean().item()
    
    stats['esm_feature_norm'] = esm_norm
    stats['gvp_scalar_feature_norm'] = gvp_scalar_norm
    
    # Vector norm (if provided)
    if gvp_vector is not None:
        valid_gvp_vector = gvp_vector[mask]  # [N_valid, 16, 3]
        # Compute norm per vector, then average over vectors and residues
        gvp_vector_norm = valid_gvp_vector.norm(dim=-1).mean().item()
        stats['gvp_vector_feature_norm'] = gvp_vector_norm
    
    # h_geo norm (final geometric embedding, if provided)
    if h_geo is not None:
        valid_h_geo = h_geo[mask]  # [N_valid, 320]
        h_geo_norm = valid_h_geo.norm(dim=-1).mean().item()
        stats['h_geo_feature_norm'] = h_geo_norm
        
        # Contribution ratio using h_geo (full structural pathway) vs ESM
        # Normalize by dimension
        h_geo_norm_normalized = h_geo_norm / (320 ** 0.5)
        esm_norm_normalized = esm_norm / (1280 ** 0.5)
        
        total_norm = h_geo_norm_normalized + esm_norm_normalized
        if total_norm > 0:
            stats['h_geo_contribution_ratio'] = h_geo_norm_normalized / total_norm
            stats['esm_contribution_ratio'] = esm_norm_normalized / total_norm
    else:
        # Fallback: use just scalar if h_geo not available
        gvp_norm_normalized = gvp_scalar_norm / (256 ** 0.5)
        esm_norm_normalized = esm_norm / (1280 ** 0.5)
        
        total_norm = gvp_norm_normalized + esm_norm_normalized
        if total_norm > 0:
            stats['gvp_contribution_ratio'] = gvp_norm_normalized / total_norm
            stats['esm_contribution_ratio'] = esm_norm_normalized / total_norm
    
    # Modality dropout statistics
    if esm_mask is not None:
        # esm_mask is [B, 1, 1], compute fraction of batches with ESM enabled
        dropout_rate = 1.0 - esm_mask.float().mean().item()
        stats['esm_dropout_rate'] = dropout_rate
    
    return stats

