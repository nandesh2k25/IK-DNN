# Notebook Map — `august_thesis_claude` (Colab, Model A)

A cached index of this Colab notebook's cells, so edits can target a cell by ID directly (`update_cell`/`run_code_cell`) instead of fetching the whole notebook first. Colab cell IDs are stable across a session — they don't change when a cell's content is edited, only when cells are inserted/deleted/moved.

**This is a cache, not ground truth.** Before trusting an ID here for an edit, especially after time has passed or the notebook may have been touched outside this map's tracking, do a cheap targeted `get_cells` (small `cellIndexStart`/`cellIndexEnd` range, `includeOutputs: false`) to confirm the cell still matches — not a full-notebook fetch.

**Keep this in sync:** update the relevant row(s) immediately after any `update_cell`/`add_code_cell`/`add_text_cell`/`delete_cell`/`move_cell` call, in the same turn as the edit. See `.claude/skills/notebook_map/SKILL.md`. If the user edits the notebook directly in the browser, they'll say so and this file needs a manual re-sync (re-fetch full notebook, rewrite this table).

**⚠️ The committed `.ipynb` file has one MORE cell than the live notebook.** When Colab saves to GitHub it prepends an "Open In Colab" badge cell to the *committed file only* — it is **not** in the live session. Verified 2026-08-08: live notebook = 29 cells starting at `lIYdn1woOS1n`; committed file = 30 cells starting with the badge. So:

- **The index numbers in the table below refer to the LIVE notebook** (what `get_cells`/`update_cell` see). That's what matters for editing.
- If you're ever reading the `.ipynb` file on disk instead, **file index = live index + 1**.

Don't "fix" this offset by renumbering the table — the table is correct for the tool you actually edit with.

**Git workflow (as of 2026-08-08):** the notebook is version-controlled at
`IK_MDN(3R in 3D)_tuned/notebooks/august_thesis_claude.ipynb` on GitHub
(`nandesh2k25/IK-DNN`, branch `master`). **Colab is the sole writer of this file** —
the user saves via *File → Save a copy in GitHub*, then it's pulled locally.
Don't edit the notebook file locally and commit it; that reintroduces the
divergence this workflow exists to prevent. Note Colab's save dialog wants the
path *within* the repo — including the repo name or a stray trailing space
silently creates a duplicate at a new path instead of updating this one (this
already happened once).

**Last full sync:** 2026-08-08, verified live against Colab after the user's *Save a copy in GitHub* (commit `594870f` on `master`). Live cell IDs and indices unchanged from the previous sync — only the committed file gained the badge cell.

---

## Cell index

| # | Cell ID | Type | Section | Summary |
|---|---|---|---|---|
| 0 | `lIYdn1woOS1n` | code | — | Imports: `time`, `numpy as np`, `matplotlib.pyplot as plt` |
| 1 | `QlmXjFurKqHB` | markdown | Title | Notebook overview, current-state roadmap bullets (v3 elbow_config, v5 mini-batch) |
| 2 | `S3_6o4eiKxOs` | markdown | 1. Data Generation | FK equations (r, z, x, y) |
| 3 | `Tmq9n1WQbmGw` | markdown | 1. Data Generation | Angle encoding: sin/cos for theta1 rationale |
| 4 | `F0VoUj8mv6WM` | markdown | 1. Data Generation | Elbow Configuration Input (v3) rationale |
| 5 | `pTiMiasMK0Yy` | code | 1. Data Generation | `forward_kinematics`, `encode/decode_theta1_sincos`, `encode_elbow_config`, `generate_dataset`; builds `X_train_norm`/`Y_train_norm`/`X_dev_*`/`X_test_*`, `N_SAMPLES=30000`, `X_mean`/`X_std`/`Y_mean`/`Y_std` |
| 6 | `NTQfvvvTK15d` | markdown | 2. Helper Functions | Section intro |
| 7 | `b4wYU4tNK4EO` | code | 2. Helper Functions | `relu`, `linear`, `relu_backward`, `linear_backward`, `mae_degrees` (circular), `pose_error_mm` |
| 8 | `pDK7f71dK5aU` | markdown | 3. NN Model | Section intro |
| 9 | `wEya-3d5K6is` | markdown | 3.1 | Parameter init rationale (He init) |
| 10 | `0ttwxl42K8Ky` | code | 3.1 | `initialize_parameters` + sanity check |
| 11 | `RkchUJHXK_TD` | markdown | 3.2 | Forward prop rationale |
| 12 | `_ttp6vDWLCVq` | code | 3.2 | `linear_activation_forward`, `forward_propagation` + sanity check |
| 13 | `AW7tcm2RLDXR` | markdown | 3.3 | Cost function rationale |
| 14 | `B86mTo8eLEiu` | code | 3.3 | `compute_cost` + sanity check |
| 15 | `Y31iTFfnLFvb` | markdown | 3.4 | Backward prop rationale, `dAL` bug note |
| 16 | `KdMjmVOjLHnq` | code | 3.4 | `linear_activation_backward`, `backward_propagation` + sanity check |
| 17 | `Jv4zNPH2LIwM` | markdown | 3.5 | Update parameters rationale (plain GD) |
| 18 | `kaBeI0RlLKae` | code | 3.5 | `update_parameters` |
| 19 | `FW2XHbjDBBW_` | markdown | 3.6 | Mini-batch GD rationale (v5) |
| 20 | `s4BUtqqiBDb3` | code | 3.6 | `random_mini_batches` + sanity check |
| 21 | `BbN5bi1QLL9H` | markdown | 3.7 | `nn_model` docs — full-batch vs mini-batch modes |
| 22 | `sN2abBUaLNT8` | code | 3.7 | `nn_model` — trains, times itself internally, returns `(parameters, costs, train_time, final_cost)` |
| 23 | `rjaj4jROLOu0` | markdown | 4. Analysis | Section intro; mentions built-in timer + `model_a_training_log.csv` |
| 24 | `4-ONOXF2LQjR` | code | 4. Analysis | Training cell: `VERSION`/`NOTES`/`layer_dims`/`BATCH_SIZE`/`LEARNING_RATE`/`NUM_ITERATIONS`, calls `nn_model`, plots cost curve |
| 25 | `-xmJjm1oLSrj` | code | 4. Analysis | `predict_ik`, `evaluate` (returns `config_accuracy` too), `log_run_metrics`, calls both + logs the run to `model_a_training_log.csv` |
| 26 | `e7qfFh-89CiB` | markdown | 5. Diagnostic | Singularity-vs-precision-ceiling intro |
| 27 | `qq7hQwgaLUau` | code | 5. Diagnostic | Scatter plot: predicted vs. true angle, per joint |
| 28 | `5aBqP7IEd_VH` | code | 5. Diagnostic | Singularity proximity correlations, worst-decile stratification, branch-selection accuracy near each singularity |

---

## Current key variables (after running top to bottom)

`parameters`, `costs`, `train_time`, `final_cost` — from cell 24's `nn_model` call.
`VERSION`, `NOTES`, `layer_dims`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_ITERATIONS` — cell 24's run config (edit these to start a new experiment; bump `VERSION` for a new logged run).
`dev_pred_theta`/`test_pred_theta`, `*_joint_mae`, `*_pose_err`, `*_config_acc` — from cell 25's `evaluate()` calls.
`model_a_training_log.csv` — written by cell 25, lives in Colab's ephemeral session (`/content`). **Not in the repo, and it does not survive a runtime restart** — unlike the notebook, saving to GitHub does not carry it along. Currently holds only the v5 row. If the version-to-version comparison history matters, it needs downloading (or writing to mounted Drive) before the runtime recycles.
