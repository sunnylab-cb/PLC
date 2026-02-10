# Pocket-GVP-Fusion v2: Model Architecture Documentation

## Overview

**Pocket-GVP-Fusion v2** is a hybrid deep learning model for protein-ligand binding site prediction. It combines geometric, sequential, and topological information to predict which protein residues interact with ligand molecules.

**Task**: Multi-class classification (per-residue)
- **Contact** (<3.5Å): Direct binding interaction
- **Near** (3.5-6.0Å): Proximal interaction
- **Far** (>6.0Å): No interaction

**Key Innovation**: Dual-pathway architecture that separately processes geometric (GVP) and sequential (ESM) features before fusion, preventing ESM from dominating early training.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT FEATURES                            │
├─────────────────┬──────────────────┬────────────────────────────┤
│   Protein       │    Protein       │       Ligand               │
│   Sequence      │    Structure     │       Graph                │
│                 │                  │                            │
│ ESM-3 (1536D)   │ Complete (82D)   │ Atom Features (35D)        │
│                 │ GVP Vectors (4×3)│ Edge Features (6D)         │
│                 │ Coordinates (3D) │                            │
└────────┬────────┴────────┬─────────┴──────────┬─────────────────┘
         │                 │                    │
         │                 ▼                    ▼
         │        ┌────────────────┐   ┌───────────────┐
         │        │  Module 2:     │   │  Module 5:    │
         │        │  GVP Encoder   │   │  Ligand GNN   │
         │        │  (3 layers)    │   │  (GINEConv)   │
         │        └────────┬───────┘   └────────┬──────┘
         │                 │                    │
         │                 ▼                    │
         │        ┌────────────────┐            │
         │        │  Module 3:     │            │
         │        │  Pairwise      │            │
         │        │  Bridge        │            │
         │        └────────┬───────┘            │
         │                 │                    │
         │        ┌────────┴───────┐            │
         │        │ Geometric Path │            │
         │        │ (GVP+Bridge+   │            │
         │        │  Physchem)     │            │
         │        └────────┬───────┘            │
         │                 │                    │
         ▼                 ▼                    │
┌────────────────┐ ┌──────────────┐             │
│  ESM Dropout   │ │  GVP Proj    │             │
│  (Training)    │ │  427→256     │             │
└────────┬───────┘ └──────┬───────┘             │
         │                 │                    │
         ▼                 ▼                    │
┌────────────────┐ ┌──────────────┐             │
│  ESM Proj      │ │              │             │
│  1536→256      │ │              │             │
└────────┬───────┘ └──────┬───────┘             │
         │                 │                    │
         └────────┬────────┘                    │
                  ▼                             │
         ┌────────────────┐                     │
         │  Concatenate   │                     │
         │  256 + 256     │                     │
         │  = 512         │                     │
         └────────┬───────┘                     │
                  ▼                             │
         ┌────────────────┐                     │
         │  Module 4:     │                     │
         │  Gated Trans-  │                     │
         │  former (4L)   │                     │
         └────────┬───────┘                     │
                  │                             │
                  ▼                             ▼
         ┌────────────────────────────────────────┐
         │         Module 6:                      │
         │         Cross-Attention Fusion         │
         │         (Protein ← Ligand)             │
         └────────────────┬───────────────────────┘
                          ▼
                 ┌────────────────┐
                 │  Module 7:     │
                 │  Late Injection│
                 │  Concat(H_fused│
                 │  + Physchem)   │
                 └────────┬───────┘
                          ▼
                 ┌────────────────┐
                 │  Classifier    │
                 │  594→256→3     │
                 └────────┬───────┘
                          ▼
                 ┌────────────────┐
                 │  Output Logits │
                 │  [B, L, 3]     │
                 └────────────────┘
```

---

## Module-by-Module Breakdown

### Module 1: Input Featurizer

**Location**: Handled in data loading (`src/data/pocket_dataset_v3.py`, `src/data/collate.py`)

**Inputs**:
- **ESM-3 embeddings**: `[B, L, 1536]` - Pre-computed protein language model embeddings
- **Complete features**: `[B, L, 82]` - Physicochemical (42D) + Surface (14D) + Geometry (26D)
- **GVP vectors**: `[B, L, 4, 3]` - Geometric vector features (normals, tangents, etc.)
- **Coordinates**: `[B, L, 3]` - Cα atom positions
- **Ligand atom features**: `[N_atoms, 35]` - Atom type, hybridization, charge, etc.
- **Ligand edge features**: `[E, 6]` - Bond type (4D) + conjugated + in_ring

**Code Reference**: `src/data/pocket_dataset_v3.py:95-150`

---

### Module 2: Protein Geometric Encoder (GVP)

**Purpose**: Encode local 3D protein geometry using Geometric Vector Perceptrons

**Architecture**:
```
Input: protein_scalar [N, 82], protein_vectors [N, 4, 3], coords [N, 3]
  ↓
k-NN Graph Construction (k=20, radius=10Å)
  ↓
Initial Projection:
  - Scalar: 82 → 256
  - Vector: 4 → 16
  ↓
GVPConv Layer 1 (256, 16) → (256, 16)
  ↓
GVPConv Layer 2 (256, 16) → (256, 16)
  ↓
GVPConv Layer 3 (256, 16) → (256, 16)
  ↓
LayerNorm on scalars
  ↓
Output: h_s_gvp [N, 256], h_v_gvp [N, 16, 3]
```

**Key Features**:
- **Vector Gating**: Learned gates control vector feature flow
- **Safe Norm**: Prevents gradient explosion with ε=1e-8
- **Residual Connections**: Within each GVPConv layer
- **Per-Protein Processing**: Each protein in batch processed independently

**Dimensions**:
- Input scalar: 82D (complete features)
- Input vector: 4 channels × 3D
- Hidden scalar: 256D
- Hidden vector: 16 channels × 3D
- Edge scalar: 1D (distance)
- Edge vector: 1 channel × 3D (direction)

**Code Reference**: `src/models/gvp_layers.py:200-293`, `src/models/pocket_gvp_fusion.py:285-338`

---

### Module 3: Bridge & Early Injection

**Purpose**: Convert rotation-invariant vector features to scalars and create dual projection paths

**Step A: Pairwise Bridge**

Converts vector features to translation/rotation-invariant scalars via Gram matrix:

```python
# Compute pairwise inner products
gram = torch.einsum('bnci,bnji->bncj', h_v, h_v)  # [B, N, 16, 16]

# Extract upper triangle (unique pairs)
# 16 channels → 16×17/2 = 136 unique pairs
pairwise_out = gram[upper_triangle_indices]  # [B, N, 136]
```

**Step B: Geometric Path Construction**

```
h_geo = Concat([h_s_gvp, pairwise_out, protein_scalar])
      = [256 + 136 + 82] = 427D
```

**Step C: Modality Dropout (Training Only)**

```python
# Drop ESM with probability 0.3 to force GVP learning
if training:
    esm_mask = Bernoulli(1 - 0.3)  # [B, 1, 1]
    esm_embeddings = esm_embeddings * esm_mask
```

**Step D: Separate Projections**

```
GVP Path:  h_geo [427D] → Linear → LayerNorm → Dropout → [256D]
ESM Path:  esm [1536D] → Linear → LayerNorm → Dropout → [256D]

Combined: Concat([gvp_proj, esm_proj]) = [512D]
```

**Why Separate Paths?**
- Prevents ESM from dominating early training when GVP weights are random
- Allows balanced contribution from both modalities
- Enables modality dropout without dimension mismatch

**Code Reference**: `src/models/pocket_gvp_fusion.py:22-66, 340-390`

---

### Module 4: Protein Context Encoder (Gated Transformer)

**Purpose**: Capture long-range dependencies in protein sequence

**Architecture**:
```
Input: h_seq [B, L, 512]
  ↓
Sinusoidal Positional Encoding
  ↓
GatedTransformerBlock 1:
  - LayerNorm → GatedMultiHeadAttention (8 heads) → Residual
  - LayerNorm → SwiGLU FFN → Residual
  ↓
GatedTransformerBlock 2
  ↓
GatedTransformerBlock 3
  ↓
GatedTransformerBlock 4
  ↓
Output: h_prot_global [B, L, 512]
```

**Key Components**:

1. **Gated Multi-Head Attention** (Qwen-style):
   ```python
   attn_out = MultiHeadAttention(x)
   gate = Sigmoid(Linear(x))
   output = attn_out * gate
   ```

2. **SwiGLU FFN** (LLaMA-style):
   ```python
   gate = SiLU(W_gate(x))
   output = W_2(W_1(x) * gate)
   ```

**Dimensions**:
- d_model: 512
- n_heads: 8
- d_head: 64
- d_ffn: ~1365 (8/3 × 512, rounded to multiple of 64)

**Code Reference**: `src/models/transformer_layers.py:184-280`, `src/models/pocket_gvp_fusion.py:175-181`

---

### Module 5: Ligand Encoder (GINEConv)

**Purpose**: Encode ligand molecular topology using Graph Isomorphism Network with Edge features

**Architecture**:
```
Input: ligand_x [N_atoms, 35], edge_index [2, E], edge_attr [E, 6]
  ↓
Node Projection: 35 → 256
Edge Projection: 6 → 256
  ↓
GINEConv Layer 1:
  - MLP: 256 → 512 → 256
  - Aggregate neighbors with edge features
  - LayerNorm → ReLU → Dropout → Residual
  ↓
GINEConv Layer 2
  ↓
GINEConv Layer 3
  ↓
Output Projection: 256 → 512
LayerNorm
  ↓
Output: h_ligand_nodes [N_atoms, 512]
```

**Scatter to Batch Format**:
```python
# Convert from flat [N_atoms, 512] to batched [B, Max_atoms, 512]
for b in range(B):
    mask_b = (ligand_batch == b)
    h_ligand[b, :n_atoms_b] = h_ligand_nodes[mask_b]
```

**Code Reference**: `src/models/ligand_encoder.py:14-105`, `src/models/pocket_gvp_fusion.py:400-435`

---

### Module 6: Cross-Attention Fusion

**Purpose**: Fuse protein and ligand representations via cross-attention

**Architecture**:
```
Input: h_prot [B, L, 512], h_ligand [B, M, 512]
  ↓
CrossAttentionBlock:
  - LayerNorm(h_prot) → Q projection
  - LayerNorm(h_ligand) → K, V projections
  - MultiHeadAttention(Q=protein, K=ligand, V=ligand)
  - Residual connection
  ↓
Output: h_fused [B, L, 512]
```

**Attention Mechanism**:
```python
Q = protein_features  # [B, L, 512]
K, V = ligand_features  # [B, M, 512]

# Protein attends to ligand
attn = Softmax(Q @ K^T / sqrt(d_head))  # [B, 8, L, M]
output = attn @ V  # [B, L, 512]
```

**Masking**: Both protein_mask and ligand_mask applied to prevent attention to padding

**Code Reference**: `src/models/transformer_layers.py:300-400`, `src/models/pocket_gvp_fusion.py:440-447`

---

### Module 7: Late Injection & Classifier

**Purpose**: Final prediction with raw physicochemical features re-injected

**Architecture**:
```
Input: h_fused [B, L, 512], protein_scalar [B, L, 82]
  ↓
Concatenate: [h_fused, protein_scalar] = [B, L, 594]
  ↓
Linear: 594 → 256
ReLU
Dropout(0.1)
  ↓
Linear: 256 → 3
  ↓
Output: logits [B, L, 3]
```

**Why Late Injection?**
- Provides direct path for physicochemical features to influence final prediction
- Prevents information loss through deep transformations
- Allows model to learn when to rely on raw features vs learned representations

**Code Reference**: `src/models/pocket_gvp_fusion.py:455-457`

---

## Complete Feature Flow

### Dimension Tracking

| Stage | Feature | Dimension |
|-------|---------|-----------|
| **Input** | ESM-3 embeddings | [B, L, 1536] |
| | Complete features | [B, L, 82] |
| | GVP vectors | [B, L, 4, 3] |
| | Coordinates | [B, L, 3] |
| | Ligand atoms | [N, 35] |
| | Ligand edges | [E, 6] |
| **GVP Encoder** | Scalar output | [B, L, 256] |
| | Vector output | [B, L, 16, 3] |
| **Pairwise Bridge** | Invariant features | [B, L, 136] |
| **Geometric Path** | h_geo | [B, L, 427] |
| **Projections** | GVP proj | [B, L, 256] |
| | ESM proj | [B, L, 256] |
| | Combined | [B, L, 512] |
| **Transformer** | h_prot_global | [B, L, 512] |
| **Ligand GNN** | h_ligand | [B, M, 512] |
| **Cross-Attention** | h_fused | [B, L, 512] |
| **Late Injection** | Concatenated | [B, L, 594] |
| **Classifier** | Hidden | [B, L, 256] |
| **Output** | Logits | [B, L, 3] |

### Parameter Counts

Approximate parameter counts for default configuration:

| Module | Parameters |
|--------|------------|
| GVP Encoder | ~500K |
| Pairwise Bridge | 0 (no learnable params) |
| GVP Projection | ~110K |
| ESM Projection | ~393K |
| Transformer (4 layers) | ~8M |
| Ligand Encoder | ~1.5M |
| Cross-Attention | ~2M |
| Classifier | ~152K |
| **Total** | **~12.7M** |

---

## Training Configuration

### Loss Function

**Focal Loss** with class weights:

```python
focal_loss = -(1 - p_t)^gamma * w_c * log(p_t)
```

Where:
- `p_t`: Probability of true class
- `gamma`: Focusing parameter (default: 2.0, recommended: 5.0)
- `w_c`: Class weight

**Recommended Class Weights**:
```yaml
class_weights: [100, 50, 1]  # [Contact, Near, Far]
```

Rationale: Extreme class imbalance (0.77% Contact, 0.87% Near, 98.36% Far)

### Optimizer

**AdamW** with:
- Learning rate: 1e-4
- Weight decay: 0.01
- Betas: (0.9, 0.999)

### Learning Rate Schedule

**Cosine Annealing** with warmup:
- Warmup epochs: 5
- Max epochs: 100
- Min LR: 1e-6

### Regularization

1. **Dropout**: 0.1 throughout
2. **ESM Modality Dropout**: 0.3 (training only)
3. **Weight Decay**: 0.01
4. **Gradient Clipping**: max_norm=1.0

---

## Ablation Studies

The codebase includes `PocketGVPFusionAblation` class for component analysis:

### Ablation Flags

```yaml
use_gvp: true              # Enable/disable GVP encoder
use_esm: true              # Enable/disable ESM embeddings
use_physchem_early: true   # Physchem in early injection
use_physchem_late: true    # Physchem in late injection
use_self_attention: true   # Protein self-attention
```

### Example Configurations

**ESM-Only Baseline**:
```yaml
use_gvp: false
use_esm: true
use_physchem_early: true
use_physchem_late: true
use_self_attention: true
```

**GVP-Only Baseline**:
```yaml
use_gvp: true
use_esm: false
use_physchem_early: true
use_physchem_late: true
use_self_attention: true
```

**No Self-Attention**:
```yaml
use_gvp: true
use_esm: true
use_physchem_early: true
use_physchem_late: true
use_self_attention: false
```

**Code Reference**: `src/models/pocket_gvp_fusion.py:629-1061`

---

## Transfer Learning

### Loading Pre-trained GVP Weights

```python
model = PocketGVPFusion(config)
model.load_gvp_from_checkpoint('path/to/gvp_pretrained.pt')
```

Loads only GVP-related parameters:
- `gvp_encoder.*`
- `pairwise_bridge.*`
- `gvp_proj.*`

### Freezing GVP Parameters

```python
model.freeze_gvp(freeze=True)
```

Useful for:
- Fine-tuning ESM pathway while keeping GVP stable
- Transfer learning from GVP-only models
- Debugging gradient flow

**Code Reference**: `src/models/pocket_gvp_fusion.py:537-626`

---

## Data Format

### Input Files

**Nested Directory Structure**:
```
data/
├── complete_features/
│   ├── 1981-2000/
│   │   └── 1ABC/
│   │       └── 1abc_protein.pt
│   ├── 2001-2010/
│   └── 2011-2019/
├── esm3_latents/
│   └── [same structure]
└── ligand_features/
    └── [same structure]
```

### Feature File Contents

**complete_features/{pdb_id}_protein.pt**:
```python
{
    'complete_features': Tensor[N, 82],  # Physchem + Surface + Geometry
    'gvp_vectors': Tensor[N, 4, 3],      # Geometric vectors
    'coords': Tensor[N, 3],              # Cα coordinates
}
```

**esm3_latents/{pdb_id}_protein.pt**:
```python
{
    'esm': Tensor[N, 1536]  # ESM-3 embeddings
}
```

**ligand_features/{pdb_id}_ligand.pt**:
```python
{
    'atom_features': Tensor[M, 35],      # Ligand atom features
    'coords': Tensor[M, 3],              # Ligand atom coordinates
    'edge_index': Tensor[2, E],          # Bond connectivity
    'edge_attr': Tensor[E, 6],           # Bond features
    'distances': {
        'distances': Tensor[N, M]        # Protein-ligand distances
    }
}
```

### Labels

**Distance-based 3-class labels**:
```python
labels[dist < 3.5] = 0  # Contact
labels[(dist >= 3.5) & (dist < 6.0)] = 1  # Near
labels[dist >= 6.0] = 2  # Far
```

**Code Reference**: `src/data/pocket_dataset_v3.py:1-200`

---

## Usage Examples

### Training

```bash
python scripts/train_v3.py --config configs/pocket_gvp_v2_nested.yaml
```

### Inference

```python
from src.models.pocket_gvp_fusion import PocketGVPFusion

# Load model
model = PocketGVPFusion(config)
model.load_state_dict(torch.load('checkpoint.pt')['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    outputs = model(
        esm_embeddings=esm,
        protein_scalar=physchem,
        protein_vectors=vectors,
        protein_coords=coords,
        protein_mask=mask,
        ligand_x=lig_x,
        ligand_edge_index=lig_edge_idx,
        ligand_edge_attr=lig_edge_attr,
        ligand_batch=lig_batch,
        ligand_to_protein=lig_padded,
        ligand_mask=lig_mask
    )

    predictions = outputs['logits'].argmax(dim=-1)  # [B, L]
```

---

## Key Design Decisions

### 1. Dual-Pathway Architecture

**Problem**: ESM embeddings (pre-trained, high-quality) dominate random GVP features early in training

**Solution**: Separate projection paths with equal dimensions (256 each) before concatenation

**Result**: Balanced gradient flow to both pathways

### 2. Modality Dropout

**Problem**: Model may over-rely on ESM and ignore geometric features

**Solution**: Drop ESM with 30% probability during training

**Result**: Forces model to learn from GVP features

### 3. Late Injection

**Problem**: Deep transformations may lose important physicochemical information

**Solution**: Re-inject raw 82D features before classifier

**Result**: Direct path for domain knowledge to influence predictions

### 4. Pairwise Bridge

**Problem**: Vector features are not rotation-invariant

**Solution**: Compute Gram matrix (pairwise inner products) to create invariant scalars

**Result**: 136D rotation-invariant geometric features

### 5. Focal Loss with High Gamma

**Problem**: Extreme class imbalance (98.36% Far class)

**Solution**: Focal loss (γ=5.0) with class weights [100, 50, 1]

**Result**: Model focuses on hard-to-classify Contact/Near residues

---

## References

1. **GVP**: Jing et al. "Learning from Protein Structure with Geometric Vector Perceptrons" (ICLR 2021)
2. **ESM-3**: Hayes et al. "Simulating 500 million years of evolution" (2024)
3. **SwiGLU**: Shazeer "GLU Variants Improve Transformer" (2020)
4. **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
5. **GINEConv**: Hu et al. "Strategies for Pre-training Graph Neural Networks" (ICLR 2020)

---

## File Structure

```
PLC/
├── configs/
│   ├── pocket_gvp_v2.yaml          # Flat data structure config
│   └── pocket_gvp_v2_nested.yaml   # Nested data structure config
├── src/
│   ├── data/
│   │   ├── collate.py              # Batch collation
│   │   ├── pocket_dataset.py       # Original dataset (flat)
│   │   └── pocket_dataset_v3.py    # Nested dataset
│   └── models/
│       ├── pocket_gvp_fusion.py    # Main model
│       ├── gvp_layers.py           # GVP implementation
│       ├── transformer_layers.py   # Transformer components
│       └── ligand_encoder.py       # Ligand GNN
├── scripts/
│   ├── train.py                    # Training script (flat)
│   └── train_v3.py                 # Training script (nested)
└── docs/
    ├── MODEL_ARCHITECTURE.md       # This file
    ├── LABEL_ANALYSIS.md           # Label distribution analysis
    ├── PROCESSING_COMPARISON.md    # PoC vs PLC comparison
    └── TRAINING_GUIDE.md           # Training instructions
```

---

## Contact & Support

For questions or issues:
- Check existing documentation in `docs/`
- Review code comments in `src/models/`
- Examine config files in `configs/`

---

*Last Updated: 2026-02-05*
