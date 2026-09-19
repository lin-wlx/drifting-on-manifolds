# Generative Modelling via Drifting on Manifolds

Code accompanying the MSc dissertation *Generative Modelling via Drifting on
Manifolds* (MSc Statistical Science, University of Oxford).

The project asks how much a Drifting Model benefits from knowing the geometry of
the space its data lives on, and what happens when that geometry has to be
estimated from the data instead. Four levels of geometric information are
compared on fifteen target distributions across three manifold families —
the sphere $S^2$, the flat torus $T^2$, and the rotation group $SO(3)$.

| Model | Support | Distance & displacement | Transport | Regression objective |
|---|---|---|---|---|
| **Exact** | analytic $P_\mathcal{M}$ | geodesic $d_\mathcal{M}$, $\mathrm{Log}_x$ | $\mathrm{Exp}_x$ | intrinsic $d_\mathcal{M}(x,\tilde x)^2$ |
| **Partial** | analytic $P_\mathcal{M}$ | graph $\hat d_G$, $\hat\Delta_x$ | $P_\mathcal{M}(x+v)$ | ambient $\lVert x-\tilde x\rVert_2^2$ |
| **Full** | learned $\hat\Pi$ | graph $\hat d_G$, $\hat\Delta_x$ | $\hat\Pi(x+v)$ | ambient $\lVert x-\tilde x\rVert_2^2$ |
| **Euclidean** | none | ambient | additive | ambient $\lVert x-\tilde x\rVert_2^2$ |

> **Exact uses a different regression objective from the other three.** This is a
> deliberate design choice, not an inconsistency: the intrinsic objective is the
> geodesic counterpart of the ambient squared error, and using it is part of what
> "exact geometry" means (dissertation §3.6). It is why Exact is trained in its
> own notebook, `experiments/03_exact_geodesic_retune.ipynb`, with its own
> hyperparameter selection under that objective.

## Repository layout

```
data_preparation/
  prepare_amass_so3.py           AMASS -> SO(3), global whole-sequence split
  protein_data_prep.ipynb        Top500 -> Ramachandran angles, PDB-level split
  archive/                       superseded variant, kept for provenance only
experiments/
  01_tuning_and_pilot.ipynb      hyperparameter tuning (15k budget, early stopping)
  02_final_fitting_50k.ipynb     final fitting, 50k steps, all four models
  03_exact_geodesic_retune.ipynb Exact retuned and fitted under the intrinsic loss
  utils/progress_monitor.ipynb   read-only progress monitor for long runs
analysis/
  04_combine_and_report.ipynb    aggregation, tables, figures, audits
results/                         CSV outputs of notebook 04
archive/                         superseded experiment revisions
```

## Which notebook produced which numbers

Run order matters, and the stages pass state through the results directory
rather than through imports.

```
prepare_amass_so3.py  ─┐
protein_data_prep     ─┼─→  01_tuning_and_pilot
                       │      └→ <dataset>/<method>/tuning/selected_hyperparameters.json
                       │                    │ (frozen; never re-run)
                       │                    ↓
                       └─→  02_final_fitting_50k
                                  └→ <dataset>/<method>/final_v3_50k_cosine/
                                                    │
    03_exact_geodesic_retune ───────────────────────┤
      └→ <dataset>/exact/final_v4_50k_cosine_geodesic_loss/
                                                    ↓
                                        04_combine_and_report
                                                    ↓
                                   tables, figures, appendix audits
```

Notebook 01 also fits a 15,000-step pilot into `<method>/final/`. Those results
are **not** reported anywhere; only its `tuning/` output feeds the next stage.

Notebook 04 sources Exact **only** from notebook 03 and Partial, Full and
Euclidean **only** from notebook 02. The Exact results produced by notebooks 01
and 02 are ignored by construction.

## Experimental protocol

Generator: four-hidden-layer MLP, width 256, SiLU, LayerNorm, latent
$\epsilon\sim\mathcal N(0,I_6)$. Adam at $10^{-3}$, EMA decay 0.999, gradient
clipping at norm 1, drift step capped at 0.25, effective batch capped at 256.

| | |
|---|---|
| Tuning budget | 15,000 steps max, early stopping from step 5,000, patience 10 |
| Tuning seeds | 43, 44, 45 |
| Bandwidth grid | $c \in \{1/32, 1/16, 1/8, 1/4, 1/2, 1\}$, $\tau = c\,m$ |
| Graph grid | $k_{\mathrm{graph}} \in \{4, 8, 16, 32, 64\}$ (Partial, Full only) |
| Selection criterion | raw ambient Gaussian MMD² on held-out validation data |
| Final budget | 50,000 steps, early stopping disabled, cosine to $10^{-5}$ at 50k |
| Final seeds | 101, 102, 103, 104, 105 |
| Geometry reference | ≤2,500 points, farthest-point subsampled, mutual $k$-NN graph |
| Evaluation size | $N_{\mathrm{eval}} = \min\{2048, |D|\}$ |

Model selection is geometry-blind: every method is scored on the raw samples it
actually produces, under a common ambient metric, so no method is judged by its
own geometry. Geometry is estimated from training observations only.

Cost: 236 trainings per dataset (23 Exact, 23 Euclidean, 95 Partial, 95 Full),
3,540 across the suite, plus the Exact geodesic retune.

## Data

Raw data is **not** redistributed here. To reproduce:

| Target family | Source |
|---|---|
| Sphere (Volcano, Earthquake, Fire, Flood) | See dissertation Appendix E.2 for provenance |
| Torus (General, Glycine, Proline, Pre-Pro) | Top500 via `protein_data_prep.ipynb` |
| $SO(3)$ (BMLhandball, DanceDB, BMLmovi, CMU) | AMASS — https://amass.is.tue.mpg.de/ (registration required) |
| Spiral, checkerboard, three-mode mixture | generated in-notebook from fixed seeds |

The notebooks expect a Google Drive layout (they were run in Colab):

```
MyDrive/
  earth_dataset/{volc,quakes,fire,flood}.csv
  protein_dataset/top500_angles_master.tsv
  AMASS_dataset/{BMLhandball,DanceDB,BMLmovi,CMU}_SO3_ready_10hz.npz
  dissertation/final_four_model_all_manifolds_v1/     <- results written here
```

To run outside Colab, set `DISSERTATION_DRIVE_ROOT` to a directory with the same
structure.

### Hold-out construction

Splits are grouped, not row-level, wherever observations are correlated:

- **Proteins** — split by PDB identifier, so residues from one structure stay in
  one partition.
- **AMASS** — split by whole motion sequence: all sequences in a collection are
  permuted once with seed 42 and assigned 80/10/10. Frames from one motion
  trajectory therefore cannot appear in more than one partition. Subjects are
  not stratified across partitions, so a given subject's sequences may fall
  entirely within one split.
- **Earth catalogues** — a plain row-level 80/10/10 permutation with seed 42.
  These catalogues repeat coordinates, so identical points can appear in both
  training and test. This is a known limitation, recorded in the dissertation.

## Requirements

```
pip install -r requirements.txt
```

A CUDA GPU is assumed. The full suite is expensive; notebooks 01–03 support
resuming from cached checkpoints, and `ACTIVE_DATASETS` / `ACTIVE_METHODS` at the
top of each restrict a session to a subset.

## Modifications relative to the notebooks as run

See `PATCHES.md`. Two changes were made for reproducibility; neither alters any
computation.

## Citation

See `CITATION.cff`.
