# PDBbind cached tags versus heavy-atom geometry

Date: 2026-10-13. Read-only audit; no training scores or new split certification.
The active protocol remains native heavy-atom center distance **d_min <5.0 A**
and >=3 positive residues per sample. The user deferred the vdW alternative.
Fixed5 jobs 13007974/13007975 were submitted before this follow-up analysis.

## What The Cached Tags Actually Contain

Input: `data/residue_ligand_distances_dcenter.pt`.
SHA256: `2cbcc609d836a66ca03a194244a7f8cd5d8557ce7bf6bb1a9e8a32b4fdb00479`.

This PLC-generated file stores continuous per-residue nearest heavy-atom center
distances, not a PDBbind-supplied vdW binding-residue flag. Each entry contains
`distances`, `n_residues`, and minimum/maximum/mean summaries. The historical
loader creates class0 using `<5.0 A`; it does not subtract atomic radii.

| Quantity | Measured value |
| --- | ---: |
| Complexes | 19,037 |
| Residues | 10,922,448 |
| Nonfinite cached distances | 0 |
| Positive residues, d_min <5 A | 378,874 (3.4688% of all residues) |
| Positive-distance p10 / p25 | 2.8288 / 3.2006 A |
| Positive-distance median / p75 | 3.5705 / 3.9825 A |
| Positive-distance p90 / p95 / p99 | 4.5117 / 4.7566 / 4.9517 A |
| Maximum positive distance | 4.9999814 A |
| Contacts per complex, p50 / p90 / maximum | 19 / 28 / 58 |
| Complexes with fewer than three cached positives | 2: 3rul and 4z46, two each |
| Residues at <=5.5 A (diagnostic only) | 423,034 |

Of the <5 A positives, 16.65% lie in [2,3), 58.74% in [3,4), and 24.28%
in [4,5) A; 1,255 residues (0.331%) are below 2 A. This distribution does
not establish that 5 A is an optimal biological cutoff: that threshold is
the rule used to select these positives. New cohorts recompute labels on their
validated input axes and rerun feature QC; these counts are not final pool sizes.

## Verified Example: 3ddq

Source: `New/2001-2010/3ddq/{3ddq_protein.pdb,3ddq_ligand.sdf}`.
Using the historical residue enumeration and native SDF heavy coordinates,
all 1,134 cached distances agree with recomputation, maximum absolute error
`6.50703e-6 A` (tolerance `1e-4 A`). This certifies this sample, not every cached
residue axis. Protein-feature alignment remains a separate input check.

| Rule | Positive residues |
| --- | ---: |
| Active center distance <5 A | 22 |
| Diagnostic center distance <=5.5 A | 25 |
| Deferred vdW sum +0.5 A, at least one atom pair | 13 |
| Deferred vdW sum +0.5 A, at least two atom pairs | 7 |

Here `d_min = min(d)` and `gap_min = min(d-r_protein-r_ligand)` are independently
minimized over all heavy-atom pairs in the residue. Their minimizing atom pair
need not be the same. Radii are the explicit project Bondi/extended table;
these calculations do not certify exact reproduction of BioLiP annotations.

| Residue | Center d_min (A) | Minimum vdW surface gap (A) | Atom pairs within vdW sum +0.5 A |
| --- | ---: | ---: | ---: |
| C:84 LEU | 2.7881 | -0.2819 | 4 |
| C:82 GLU | 3.1462 | -0.0738 | 1 |
| C:32 ALA | 3.3365 | -0.0635 | 3 |
| C:135 LEU | 3.3869 | -0.0131 | 8 |
| C:81 PHE | 3.3958 | -0.0042 | 4 |
| C:133 ASN | 3.4656 | 0.2456 | 1 |
| C:11 ILE | 3.5000 | 0.1608 | 4 |
| C:85 HIS | 3.5042 | 0.2842 | 1 |
| C:132 GLN | 3.6218 | 0.4018 | 1 |
| C:83 PHE | 3.6321 | 0.2321 | 2 |
| C:19 VAL | 3.6539 | 0.2539 | 2 |
| C:9 GLU | 3.7742 | 0.5542 | 0 |
| C:65 VAL | 3.7788 | 0.3788 | 1 |
| C:13 GLU | 3.7962 | 0.7262 | 0 |
| C:86 GLN | 3.8312 | 0.4312 | 1 |
| C:12 GLY | 3.8685 | 0.5956 | 0 |
| C:87 ASP | 3.9440 | 0.6940 | 0 |
| C:14 GLY | 3.9442 | 0.6942 | 0 |
| C:146 ASP | 4.0088 | 0.6088 | 0 |
| C:145 ALA | 4.0295 | 0.6295 | 0 |
| C:90 LYS | 4.3818 | 1.0099 | 0 |
| C:34 LYS | 4.7029 | 1.3029 | 0 |

All 22 are positive under the active rule. A shorter center distance does not
by itself determine vdW contact status: C:9 GLU at 3.7742 A fails the vdW
test, whereas C:65 VAL at 3.7788 A passes the one-pair test. Two-pair positivity
cannot be recovered from a single minimum-distance value alone.

The supplied `3ddq_pocket.pdb` contains 70 ATOM residues: 22 have d_min <5 A,
25 have d_min <=5.5 A, and 62 have d_min <10 A; the maximum is 11.9161 A.
Thus membership in this provided pocket file is NOT equivalent to our binding
label. This single example does not establish an official dataset-wide cutoff.

## Separate QC Finding: Coincident Source Atoms

There are **232 exactly-zero cached distances in 208 complexes**, and 327
distances below 1 A in 273 complexes. Do not interpret zero center distance as
an ordinary favorable intermolecular contact, or as a vdW surface gap.

Three source checks reproduce the cache within 3e-6 A:

| Complex | Source residue(s) with d_min <0.01 A | Source minimum (A) |
| --- | --- | ---: |
| 1jrs | A:195 SER | 1.62e-6 |
| 1e34 | B:195 SER | 1.15e-6 |
| 6g8i | B:3,4,6,7,8,9,10,11 | 1.27e-7 |

Near-identical protein/ligand coordinates are already present in these source
files; this is not explained by choosing 5 versus 5.5 A. Shared covalent-adduct
atoms or peptide-ligand/protein overlap are hypotheses requiring chemical/chain
inspection, not established classifications for all 208 cases. This audit did
not silently change eligibility or cancel the new fixed5 jobs. Review any such
cases admitted by the new axis-aligned QC before calling the dataset certified.
