# Notebook Map — `august_thesis_claude` (Colab, Model A)

A cached index of this Colab notebook's cells, so edits can target a cell by ID directly (`update_cell`/`run_code_cell`) instead of fetching the whole notebook first. Colab cell IDs are stable across a session — they don't change when a cell's content is edited, only when cells are inserted/deleted/moved.

**This is a cache, not ground truth.** Before trusting an ID here for an edit, especially after time has passed or the notebook may have been touched outside this map's tracking, do a cheap targeted `get_cells` (small `cellIndexStart`/`cellIndexEnd` range, `includeOutputs: false`) to confirm the cell still matches — not a full-notebook fetch.

**Keep this in sync:** update the relevant row(s) immediately after any `update_cell`/`add_code_cell`/`add_text_cell`/`delete_cell`/`move_cell` call, in the same turn as the edit. See `.claude/skills/notebook_map/SKILL.md`. If the user edits the notebook directly in the browser, they'll say so and this file needs a manual re-sync (re-fetch full notebook, rewrite this table).

**⚠️ The committed `.ipynb` file has one MORE cell than the live notebook.** When Colab saves to GitHub it prepends an "Open In Colab" badge cell to the *committed file only* — it is **not** in the live session. Verified 2026-08-08: live notebook = 33 cells starting at `lIYdn1woOS1n`; committed file gains the badge at the front. So:

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

**Last full sync:** 2026-08-08, after adding the v6 Adam optimizer. **Four cells were inserted**, shifting every index from 21 onward: `lozV7gLneGX7`/`TFeZsHyxeMlU` (§3.7 Adam) at 21–22, and `Od3DQJzrfqVE`/`Q_qHfieOfvY3` (§4.2 LR sweep) at 28–29. Live notebook is now 33 cells.

---

## Cell index

| # | Cell ID | Type | Section | Summary |
|---|---|---|---|---|
| 0 | `lIYdn1woOS1n` | code | — | Imports: `time`, `numpy as np`, `matplotlib.pyplot as plt` |
| 1 | `QlmXjFurKqHB` | markdown | Title | Notebook overview, current-state roadmap bullets (v3 elbow_config, v5 mini-batch, v6 Adam) |
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
| 17 | `Jv4zNPH2LIwM` | markdown | 3.5 | Update parameters rationale (plain GD); points forward to 3.7 for Adam |
| 18 | `kaBeI0RlLKae` | code | 3.5 | `update_parameters` — plain GD, deliberately untouched by v6 |
| 19 | `FW2XHbjDBBW_` | markdown | 3.6 | Mini-batch GD rationale (v5) |
| 20 | `s4BUtqqiBDb3` | code | 3.6 | `random_mini_batches` + sanity check |
| 21 | `lozV7gLneGX7` | markdown | 3.7 | **(v6)** Adam rationale: momentum + RMSprop + bias correction; the `t`-counts-updates and lr-doesn't-carry-over warnings |
| 22 | `TFeZsHyxeMlU` | code | 3.7 | **(v6)** `initialize_adam`, `update_parameters_with_adam` + sanity checks (β₁=β₂=0 reduces to a signed step; bias-corrected first step is lr-sized) |
| 23 | `BbN5bi1QLL9H` | markdown | 3.8 | `nn_model` docs — batching mode × optimizer mode, and the shared `_step` helper |
| 24 | `sN2abBUaLNT8` | code | 3.8 | `nn_model` — takes `optimizer`/`beta1`/`beta2`/`epsilon`; inner `_step` owns the global update counter `t`; returns `(parameters, costs, train_time, final_cost)` |
| 25 | `rjaj4jROLOu0` | markdown | 4. Analysis | Section intro; mentions built-in timer + `model_a_training_log.csv` |
| 26 | `4-ONOXF2LQjR` | code | 4. Analysis | Training cell: `VERSION`/`NOTES`/`layer_dims`/`BATCH_SIZE`/`NUM_ITERATIONS`/`OPTIMIZER`/`LEARNING_RATE`/`BETA1`/`BETA2`/`EPSILON`, calls `nn_model`, plots cost curve |
| 27 | `-xmJjm1oLSrj` | code | 4. Analysis | `predict_ik`, `evaluate` (returns `config_accuracy` too), `log_run_metrics` (now logs optimizer/betas/epsilon, archives the CSV on schema change, NaN-tolerant rounding), calls both |
| 28 | `Od3DQJzrfqVE` | markdown | 4.2 | **(v6)** Why the LR sweep exists — α isn't comparable across optimizers |
| 29 | `Q_qHfieOfvY3` | code | 4.2 | **(v6)** Sweep over `SWEEP_CONFIGS` (gd@0.05 as v5-repro + Adam at 3e-4/1e-3/3e-3/1e-2); logs each, survives divergence, prints ranked summary. Does **not** overwrite `parameters` |
| 30 | `e7qfFh-89CiB` | markdown | 5. Diagnostic | Singularity-vs-precision-ceiling intro |
| 31 | `qq7hQwgaLUau` | code | 5. Diagnostic | Scatter plot: predicted vs. true angle, per joint |
| 32 | `5aBqP7IEd_VH` | code | 5. Diagnostic | Singularity proximity correlations, worst-decile stratification, branch-selection accuracy near each singularity |

---

## Current key variables (after running top to bottom)

`parameters`, `costs`, `train_time`, `final_cost` — from cell 26's `nn_model` call.
`VERSION`, `NOTES`, `layer_dims`, `BATCH_SIZE`, `NUM_ITERATIONS`, `OPTIMIZER`, `LEARNING_RATE`, `BETA1`, `BETA2`, `EPSILON` — cell 26's run config (edit these to start a new experiment; bump `VERSION` for a new logged run).
`dev_pred_theta`/`test_pred_theta`, `*_joint_mae`, `*_pose_err`, `*_config_acc` — from cell 27's `evaluate()` calls.
`sweep_results` — from cell 29; a list of `(tag, optimizer, lr, cost, angle_mae, pose_mm, elbow_acc, seconds)`. Cell 29 uses `_`-prefixed locals throughout specifically so it does **not** clobber the canonical run's globals.
`model_a_training_log.csv` — written by cells 27 and 29, lives in Colab's ephemeral session (`/content`). **Not in the repo, and it does not survive a runtime restart** — unlike the notebook, saving to GitHub does not carry it along. The v5 row from the previous session was already lost this way. Currently holds 6 rows (`v5-repro`, `v6-lr3e-04`, `v6-lr1e-03`, `v6-lr3e-03`, `v6-lr1e-02`, `v6`). If the version-to-version comparison history matters, it needs downloading (or writing to mounted Drive) before the runtime recycles.

Note the log's schema changed in v6 (added `optimizer`, `beta1`, `beta2`, `epsilon`). `log_run_metrics` compares the existing header against its field list and renames the old file to `model_a_training_log_schema_<timestamp>.csv` on a mismatch rather than appending misaligned rows — so a stale log won't silently corrupt a new one.
