# LiBRe on the Locked Joint TEST: Feasibility and SI Audit

Checked 2026-10-23. This is an input/provenance audit, not a new performance
evaluation. No prediction scores or TEST labels were used to select anything.
No GPU job, training, checkpoint update or new large download was launched.

## Feasibility

The released checkpoint, pinned source and ESM2 weights are already on HPC2:

```text
/hpc2hdd/home/shuang886/PLC/external_baselines/libre/
```

The official repository now resolves to
[keumseok/LiBRe](https://github.com/keumseok/LiBRe); the current upstream commit
still matches the previously pinned
`e9151a85bab2d05b67b953bc55a74c780aa2bf24`.
Checkpoint SHA256 is unchanged:
`ba40bdf78ba4e596e7429e56d2c955c42b6debc67fb159a7438dfe6b9993c871`.

A fresh CPU `weights_only=True` load matched all 68 state-dict entries with
`strict=True`; the downstream model has 1,348,481 parameters. No forward pass
was run. Prior [external benchmark inference](libre_benchmark_20261010.md)
established that this pinned implementation can run in the existing environment.

The existing engine is reusable, but is not yet a complete joint-TEST adapter:

- Require raw ESM2 L33 residue embeddings and the official 40D
  SMILES-to-graph features, not PLC-normalized ESM or local54 input.
- Use the existing exact observed chain sequences and residue keys. Predict
  all represented chains, not an oracle GT binding chain, and reassemble
  onto the locked PLC residue axis.
- Preserve the current 17,395-row TEST and labels d_min<5 A. Do not adopt
  LiBRe's paper preprocessing filters after inspecting performance.
- Document chain-window handling above LiBRe's original 1500-residue input
  length. Match the original PLC metric axis (max1800), not a new hidden cap.
- Fresh chemistry and feature coverage QC is necessary. The two sampled
  ligand caches load safely, but this is not all-row SMILES acceptance.
  The sampled PDBbind ligand metadata still contains a local historical
  SDF path, so canonical ligand input must be located rather than assumed.
- For a released-checkpoint reference, keep its fixed p>0.6 rule and
  prespecified Top-K budgets; also report AP/AUROC/MCC and matching PLC
  metrics on the identical accepted population. No TEST threshold search.

This means technically feasible, not already submitted or fully input-ready.

## Does LiBRe Guarantee SI30?

**No such guarantee is reported or implemented in the inspected materials.**

The [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02883), Methods /
Data Preprocessing, reports training data from PDBbind2020 and sc-PDB2017,
CD-HIT sequence-identity threshold0.9 for training redundancy reduction,
removal of shared train/test PDB IDs, and removal of test proteins with
Foldseek structural similarity>0.4 to training. It also filters length>1500,
fewer than10 binding residues, and unsuccessful SMILES conversions.
Foldseek structural similarity0.4 is **not** pairwise sequence identity0.3.

The official [train/val helper](https://github.com/keumseok/LiBRe/blob/e9151a85bab2d05b67b953bc55a74c780aa2bf24/src/utils.py#L227)
uses a seeded row-level random permutation. The inspected path does not
group homology components or audit pairwise SI across TRAIN/VAL. This code
observation is not proof about every unpublished experiment by the authors.

In contrast, our current split audits global SI<0.30 across TRAIN/VAL,
TRAIN/TEST and VAL/TEST for our supervised cohort's observed chain sequences.
That certificate does not cover another model's historical training data.
Neither protocol proves absence of all local/domain homology or pretrained
language-model exposure.

The paper XML was re-fetched from the official
[Europe PMC full-text API](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13080968/fullTextXML);
its SHA256 matches the archived source:
`52d687b9ea0662267a035d237abf698214147ce576a8bf44401dede4522ac435`.

## Concrete Public-Example Overlap

The current official repository exposes only a
[50-row train example CSV](https://github.com/keumseok/LiBRe/blob/e9151a85bab2d05b67b953bc55a74c780aa2bf24/data/train/train_example.csv)
and a25-row test example CSV, not a complete checkpoint-linked supervised
training manifest. We compared those50 training-example sequences with all
8,380 exact unique observed chain sequences represented in our17,395 TEST rows.

| Check against public train examples | Unique matches | Current TEST rows | Source |
| --- | ---: | ---: | --- |
| Same PDB ID | 3 PDBs | 7 | 5 BioLiP, 2 PDBbind |
| Exactly identical full observed chain sequence | 3 sequences | 21 | BioLiP |
| Union of these checks | Not an SI clustering result | 23 | 21 BioLiP, 2 PDBbind |

The7 same-PDB rows and21 exact-sequence rows overlap; they must not be added.

| Public example | Sequence length | Exactly matching TEST rows |
| --- | ---: | ---: |
| 5jar chainA, ligand6HV | 371 | 17 |
| 5v1b chainA, ligand8UY | 226 | 1 |
| 2bjh chainA, ligandFER | 260 | 3 |

For example, the5jar example sequence is exactly shared by TEST entries
5jad,5jah,5jal,5jan,5jao,5jap,5jar,5jas,5jat,5jau,5lyy,5lz2,5lz4,5lz5,
5lz7,5lz8 and5yea. This illustrates why excluding shared PDB IDs alone
does not certify protein novelty.

**These are verified overlaps with public training examples, not a verified
leakage rate of the released checkpoint.** The complete supervised training
membership and checkpoint provenance are unavailable in the inspected public
assets. No full SI30 cross-audit against that unknown membership is possible.
Removing just these23 rows would not certify the remainder as clean.

The [machine-readable audit](results/libre_joint_test_overlap_preflight_20261023.json)
retains matched sample keys, sequence hashes, source counts and input digests.
It contains no raw feature tensors or predictions. Exact sequence matching
uses the full observed chain strings, not model caps or local alignments.

## Comparison Options

1. **Released checkpoint reference:** technically feasible after the adapter
   and full input QC. Label it external pretrained inference with unaudited
   training overlap. Do not call it strict-SI30 generalization or an
   architecture-only superiority comparison.

2. **Strict matched supervised comparison:** randomly initialize the LiBRe
   task model (reuse the same frozen pretrained ESM2 as an explicit shared
   resource), train only on our existing TRAIN, select checkpoint/threshold
   only on existing VAL, and evaluate the locked TEST once. Do not resume
   the public supervised checkpoint, rerandomize its train/val rows, or
   silently remove short-contact TEST cases. Adaptation choices and any
   chemistry failures must be recorded independently of measured performance.

No change to the existing split or PLC model is justified by this audit.
This note addresses comparison validity only; TEST remains reference-only.
