# Results

`analysis/04_combine_and_report.ipynb` writes its aggregated outputs here:

```
completion_audit.csv                      per dataset/method/seed completeness
combined_test_metrics_by_seed.csv         one row per trained seed
combined_test_metrics_long.csv            long-form metrics
combined_test_results_table_mean_sd.csv   mean +/- SD over the five final seeds
dissertation_primary_results.csv          compact main-text table
real_vs_real_reference.csv                finite-sample reference per dataset
best_validation_steps.csv                 checkpoint locations
```

Per-seed model checkpoints (`.pt`) are not committed — they are large and
regenerable from `experiments/`.
