# Ligand Encoding Review for PLC

Date: 2026-05-26

This note records the side investigation on ligand encoding choices for the current
ProtBind/PLC V11/V12 line.  It is intended to guide the next ligand-side
experiments without changing the stable protein trunk prematurely.

## 1. Current Evidence in PLC

### Current ligand input structure

The current V11m-series ligand path is effectively:

```text
local54 atom chemistry
+ global55 broadcast ligand descriptor / shape summary
+ UniMol adapter feature
-> ligand node projection / ligand encoder
```

The important observation is that these groups have different semantic levels:

- `local54`: atom-level chemistry, including atom type, local bonding context,
  charge/aromatic/ring/HBD/HBA-like information, flexibility and local
  contribution descriptors.
- `global55`: molecule-level and shape-like descriptors broadcast to every atom.
  This is not residue-specific evidence; it behaves more like calibration.
- `UniMol`: pretrained molecular prior.  Current ablations suggest it is not yet
  the dominant source of ligand dependency in this pipeline.

### Dependency evidence so far

Existing ligand dependency experiments support this ordering:

```text
local54 / ligand graph contribution > global55 contribution >> current UniMol-only contribution
```

Representative conclusion from V11m3/V11m4-style ablations:

- Zeroing `local54` drops F1 much more than zeroing `global55`.
- Zeroing / shuffling UniMol alone has small effect under the current injection.
- `global55` changes precision/recall balance more than it creates residue-level
  ligand evidence.

Interpretation:

```text
The model is not purely ligand-blind.
The useful ligand signal is mainly local atom chemistry.
The current weakness is that ligand evidence enters at a lower semantic level
than protein evidence and can be underused or mis-calibrated.
```

## 2. What Peer Models Commonly Do

### Atom graph remains the standard base

Mainstream protein-ligand and molecular models do not usually collapse the ligand
into a single global vector.  They keep atom-level tokens / graph nodes as the
base representation.

Common atom features include:

- atom type / atomic number
- degree / valence / hydrogen count
- formal charge
- hybridization
- aromatic / ring flags
- chirality / stereochemistry
- donor / acceptor / pharmacophore flags
- bond type and ligand-internal geometry / distance features

This supports keeping atom-level local features in PLC.

### Global molecule features are useful but not sufficient

Molecular descriptors, fingerprints, global shape features and pretrained CLS
embeddings are commonly used as graph-level context or calibration.  They are not
a good replacement for atom-level evidence when the output is residue-level.

For PLC, this supports:

```text
global55 / UniMol CLS -> calibration or global prior token
local54 / ligand graph -> residue-level evidence source
```

### Hierarchical ligand views are increasingly common

Recent molecular representation work often combines multiple levels:

```text
atom-level tokens
+ pharmacophore / functional group tokens
+ graph-level / molecule-level token
+ pretrained molecular prior
```

This is the most relevant direction for PLC because protein residue tokens have a
rich semantic prior from ESM/GVP/physchem/spatial features, while ligand atom
tokens are comparatively low-level.  Functional-group or pharmacophore tokens can
bridge this semantic-level gap.

## 3. Encoding Choices

### One-hot / categorical embeddings

Verdict: keep, but organize by field.

One-hot for categorical atom fields is still appropriate.  For a categorical
field, `one_hot -> Linear` is equivalent to a learned embedding table.  This is
standard for atom type, hybridization, chirality, aromatic/ring flags and similar
non-ordinal features.

Recommended refinement:

```text
atom categorical fields -> learned embeddings or grouped one-hot adapter
atom continuous fields  -> normalized scalar / RBF / small MLP
global descriptors      -> global token / calibration gate
UniMol                  -> separate adapter or token
```

The issue is not that one-hot is wrong.  The issue is raw mixed-group concat into
one projection layer.

### Thermometer / ordinal encoding

Verdict: use only for ordered discrete fields.

Good candidates:

- degree
- valence
- hydrogen count
- formal charge bins
- ring size bins
- rotatable-bond count bins

Bad candidates:

- atom type
- hybridization if treated as a nominal category
- chirality labels
- pharmacophore class labels

Reason:

```text
Thermometer encoding assumes monotonic order.
C/N/O/S/halogen atom identities do not have a single meaningful order.
```

For continuous local descriptors such as charge, local SASA, RMSF, logP/TPSA
contribution, RBF or soft-binning is more appropriate than thermometer encoding.

### Target encoding

Verdict: do not use in the main model.

Target encoding is risky here because the target is residue-level
contact/near/far and depends on both protein and ligand.  A ligand-side target
mean can leak dataset frequency, ligand identity and protein-family shortcut
signals.

Acceptable use:

- offline diagnostic only
- strict out-of-fold computation
- not used as a production feature

Not recommended:

```text
atom type -> empirical contact rate
ligand scaffold -> empirical positive rate
functional group -> empirical residue-label prior
```

These would likely strengthen dataset shortcut rather than ligand-conditioned
interaction reasoning.

### Single whole-ligand embedding

Verdict: not as the only ligand representation.

A single ligand vector can help global calibration, but it cannot represent
which ligand substructure should affect which residue.  Since the current task is
residue-level labeling, a single global ligand vector loses the residue-specific
interaction channel.

Useful role:

```text
whole-ligand embedding -> global token / delta gate / contact threshold calibration
```

Not useful as sole representation:

```text
whole-ligand embedding only -> residue-level contact evidence
```

### Functional group / pharmacophore tokens

Verdict: highest-value next ligand representation upgrade.

Motivation:

- PLC ablations show local atom chemistry matters.
- Protein tokens are semantically rich; ligand atom tokens are still primitive.
- Functional group tokens provide a middle layer: more meaningful than atoms,
  more residue-specific than a whole-ligand vector.

Candidate tokens:

- carboxylate / phosphate / sulfonamide / amide / urea / guanidine
- aromatic ring / heteroaromatic ring / fused ring system
- donor group / acceptor group / cationic group / anionic group
- metal chelator
- halogenated aromatic group
- hydrophobic/aromatic pharmacophore group

Recommended use:

```text
ligand memory = atom tokens + functional-group tokens + global prior token
protein residue queries ligand memory
```

This keeps atom-level specificity while raising ligand semantic density.

## 4. Input Balance: What to Measure

Dimensional compression is helpful but not sufficient.

Examples:

```text
ESM 1280 -> 256
UniMol 512 -> 64
```

These prevent pretrained embeddings from dominating purely by dimension.  But
they do not prove balanced contribution.  The correct measurement is projected
contribution RMS:

```text
RMS(W_local54  x_local54)
RMS(W_global55 x_global55)
RMS(W_unimol   x_unimol)
```

For protein:

```text
RMS(h_gvp_proj)
RMS(h_esm_proj)
RMS(gate * h_gvp_proj)
RMS((1 - gate) * h_esm_proj)
```

Goal:

```text
Same order of magnitude at initialization / early training.
Not permanently equal contribution.
```

This is why `V11m4tp` and `V11m5t-balanced` are useful:

- `V11m4tp`: records the old entrance contribution without changing logits.
- `V11m5t-balanced`: uses group-wise adapters to make local/global/UniMol
  projected RMS more controlled.

## 5. Recommended Roadmap

### P0: Finish current balance probes

Wait for:

- `V11m4tp`: old input projection contribution probe.
- `V11m5t-balanced`: group-wise adapter experiment.

Evaluate:

- full F1 / Dice
- dependency drops: shuffle / zero / zero_local54 / zero_global55 / zero_unimol
- TB health:
  - `Health/ligand_group_*_proj_rms`
  - `Health/ligand_*_nodeproj_contrib_rms`
  - ligand adapter gradients
  - protein ESM/GVP projection RMS

Decision:

```text
If entrance was imbalanced and balanced adapter improves or preserves F1:
  keep group-wise adapters as V11m5t mainline.

If entrance was already balanced or balanced adapter gives no benefit:
  stop focusing on input scaling and move to functional-group tokens.
```

### P1: Keep group-wise ligand adapters if validated

The clean adapter form is:

```text
local54  -> LN/MLP/LN -> local hidden
global55 -> LN/MLP/LN -> global hidden
UniMol   -> LN/MLP/LN -> unimol hidden
concat -> ligand encoder
```

This is an initialization/optimization fix, not a new interaction mechanism.

### P2: Design V11m6 functional-group / pharmacophore tokens

Constraints:

- keep V11m4t/V11m5t stable protein trunk
- no protein-ligand cross-distance
- no holo leakage
- no target encoding
- functional group tokens are ligand-only

Initial design:

```text
atom tokens from local54
functional-group tokens from RDKit SMARTS / pharmacophore detection
global token from global55 + UniMol CLS
protein residue reads ligand memory
```

Minimal ablation:

```text
atom-only
atom + FG tokens
atom + FG tokens + global gate
```

Metrics:

- full F1 / Dice
- shuffle drop
- zero_local54 drop
- zero_global55 drop
- FG-token ablation drop
- precision/recall balance

### P3: Only after P2, consider richer ligand priors

Candidates:

- UniMol atom-level adapter as stronger memory
- ChemBERTa / MoLFormer CLS token
- graph-pretrained token such as KPGT/GROVER-style graph prior
- conformer ensemble / microstate features

These are higher-cost and should wait until functional-group tokens are tested.

## 6. Parked Directions

Do not prioritize for now:

- target encoding as train feature
- single whole-ligand embedding replacing atom tokens
- more pair-bias attention injection before proving stable benefit
- more co-attention depth as a first response
- adding many pretrained ligand priors before resolving current source-level
  dependency

## 7. References / Search Notes

Links used in the side search:

- DiffDock featurization sensitivity:
  https://www.blopig.com/blog/2025/04/featurisation-is-key-one-version-change-that-halved-diffdocks-performance/
- Atomic featurization study:
  https://link.springer.com/article/10.1186/s13321-023-00751-7
- MONN / compound atom by protein residue interaction:
  https://www.sciencedirect.com/science/article/pii/S2405471220300818
- PharmHGT hierarchical atom/pharmacophore/junction representation:
  https://www.nature.com/articles/s42004-023-00857-x
- KPGT knowledge-guided molecular pretraining:
  https://www.nature.com/articles/s41467-023-43214-1
- Uni-Mol repository:
  https://github.com/deepmodeling/Uni-Mol

Summary of peer practice:

```text
Atom-level graph remains the base.
Global ligand representation is useful as prior/calibration, not a replacement.
Functional group / pharmacophore / graph-level tokens are the most relevant
next level for PLC.
```
