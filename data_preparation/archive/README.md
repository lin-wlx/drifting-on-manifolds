# Superseded data preparation

`prepare_amass_so3_subject_stratified.py` splits AMASS sequences **within** each
subject/session directory, so every subject appears in all three partitions.

It was **not** used. The prepared `*_SO3_ready_10hz.npz` files consumed by the
experiments were produced by `../prepare_amass_so3.py`, which applies a single
global 80/10/10 permutation over complete motion sequences. This is recorded
inside the data files themselves:

```python
import numpy as np
with np.load(path, allow_pickle=True) as d:
    print(str(d["split_note"].item()))   # "Fixed global 80/10/10 split ..."
    print("train_subject" in d)          # False for the global script
```

Both scripts split at the level of whole motion sequences, so neither permits
frames from one trajectory to appear in more than one partition. They differ
only in whether subjects are forced to appear in every partition.

Kept for provenance. Do not run it: it writes the same output filename as the
canonical script and would silently overwrite prepared data.
