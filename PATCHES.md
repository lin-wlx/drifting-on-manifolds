# Changes relative to the notebooks as they were actually run

Two edits were made when assembling this repository. Neither changes any
computation, model, hyperparameter or reported number.

## 1. `experiments/03_exact_geodesic_retune.ipynb` — final output directory

**As run**, the notebook wrote final checkpoints and test evaluations to

```
<manifold>/<dataset>/exact/geodesic_loss_retuned/final_50k_cosine/
```

**Notebook 04** reads Exact from

```
<manifold>/<dataset>/exact/final_v4_50k_cosine_geodesic_loss/
```

The gap was closed by hand in Drive after the runs finished, so a clean
re-execution of the original notebooks would not have reproduced the analysis.
`geodesic_final_dir()` now returns the path notebook 04 expects:

```python
GEODESIC_FINAL_DIRNAME = "final_v4_50k_cosine_geodesic_loss"

def geodesic_final_dir(spec):
    return method_dir(spec, "exact") / GEODESIC_FINAL_DIRNAME
```

Tuning output remains nested under `geodesic_loss_retuned/tuning/` so it can
never collide with the ambient-loss Exact tuning written by notebook 01.

## 2. `experiments/01_tuning_and_pilot.ipynb` — corrected a design note

A markdown cell stated that all four variants share the ambient squared-error
regression objective. That is true **of that notebook**, but it contradicts
dissertation §3.6, where Exact is defined with the intrinsic objective
`d_M(x_theta(eps), sg(x_tilde))^2`. The cell now says so and points to
notebook 03, which trains the reported Exact model.

## 3. `data_preparation/` — canonical AMASS script corrected

The repository originally presented `prepare_amass_so3.py` (subject-stratified)
as canonical, following dissertation Appendix E.4. Inspecting the prepared data
showed otherwise: all four `*_SO3_ready_10hz.npz` files carry

```
split_note: "Fixed global 80/10/10 split over complete motion sequences.
             Frames from one motion sequence never appear in multiple splits."
```

and none contains the `<split>_subject` arrays the stratified script writes. The
global-split script is therefore the one that produced the experimental data, and
the two files have been swapped: the global script is now `prepare_amass_so3.py`
and the stratified one is archived.

No code or result changed — only which file the repository presents as canonical.
Appendix E.4 of the dissertation describes the archived script and needs
correcting; see the project notes.
