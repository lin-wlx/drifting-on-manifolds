# Superseded experiment revisions

Kept for provenance. Neither produced any reported number. Do not run them.

| File | What it was | Why it was superseded |
|---|---|---|
| `four_model_v2_50k_SUPERSEDED.ipynb` | First 50k final-fitting attempt. Cosine schedule stretched to 50k. Wrote to `RESULTS_ROOT/final_v2/`. | Replaced by the V3 revision, which also saves each seed's test evaluation incrementally. |
| `four_model_FINAL_V2_50k_SUPERSEDED.ipynb` | 50k final fitting, but the cosine schedule still **ended at step 15,000**, leaving the learning rate at its minimum for the last 35,000 steps. Wrote to `<method>/final_v2/`. | The schedule bug that `experiments/02_final_fitting_50k.ipynb` fixes. |

The filenames are chronologically misleading: despite the "FINAL_V2" label,
`four_model_v2_50k` already had the stretched schedule. Both predate
`02_final_fitting_50k.ipynb`, which is the only 50k notebook whose output
notebook 04 reads.
