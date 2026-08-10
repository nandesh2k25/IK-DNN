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

**Last full sync:** 2026-08-10 (branch `error_diagnosis`), after adding §4.5 (oracle-conditioned diagnostic, 2 cells at 30–31) and §4.6 (elbow-branch ceiling diagnostic, 2 more cells at 32–33), shifting everything from the old §4.2 onward down by 4 total. Live notebook is now 41 cells. The runtime had recycled at the start of this session (fresh `execution_count`s) and was re-run top-to-bottom through cell 31 before either diagnostic's numbers could be trusted — see `CLAUDE.md`'s Colab-flakiness note.

Prior sync (2026-08-09): promoted v8 (α₀=2e-3, inverse decay → ~3e-4) as the canonical model and deleted the six §4.4 exploratory sweep cells (coarse grid, fine grid, even-larger-α₀ grid — all logged in `thesis_doc.md` §4.4 instead).

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
| 23 | `EmVG158lDD33` | markdown | 3.8 | **(v7)** LR decay rationale: orthogonal to Adam, inverse-time + staircase formulas, epoch-vs-`t` gotcha warning |
| 24 | `vFfzoTn8DG6e` | code | 3.8 | **(v7)** `update_lr`, `schedule_lr_decay` + sanity checks (epoch-0 undecayed, closed-form at epoch 10, staircase constant-then-drops, `time_interval=1` reduces to `update_lr`) |
| 25 | `BbN5bi1QLL9H` | markdown | 3.9 | `nn_model` docs — batching mode × optimizer mode × decay mode, and the shared `_step` helper |
| 26 | `sN2abBUaLNT8` | code | 3.9 | `nn_model` — takes `optimizer`/`beta1`/`beta2`/`epsilon`/`decay`/`decay_rate`/`time_interval`; inner `_step` owns the global update counter `t`; `_epoch_lr` recomputes α once per epoch; returns `(parameters, costs, train_time, final_cost)` |
| 27 | `rjaj4jROLOu0` | markdown | 4. Analysis | Section intro; mentions built-in timer + `model_a_training_log.csv` |
| 28 | `4-ONOXF2LQjR` | code | 4. Analysis | **(v8, canonical)** Training cell: `VERSION="v8"`, `LEARNING_RATE=2e-3`, `DECAY="inverse"`, `DECAY_RATE=0.0572` (→ ~3e-4 by epoch 99) — the winner of the §4.4 sweep (deleted from the notebook after promotion; full grid in `thesis_doc.md` §4.4). Test pose error 22.61mm |
| 29 | `-xmJjm1oLSrj` | code | 4. Analysis | `predict_ik`, `evaluate` (returns `config_accuracy` too), `log_run_metrics` (logs decay/decay_rate/time_interval, archives the CSV on schema change, NaN-tolerant rounding), calls both |
| 30 | `uFzr7tWgLTEk` | markdown | 4.5 | **(NEW, error-diagnosis branch)** Oracle-conditioned evaluation rationale — isolates regression error from branch-selection error by feeding the true `elbow_config` instead of best-of-2 |
| 31 | `xMzyNJ1vLWrf` | code | 4.5 | **(NEW)** `evaluate_oracle()` — same as `evaluate()` but conditions on true `elbow_config`; prints dev/test results + a best-of-2-vs-oracle comparison table. Result: oracle mean angle MAE 1.19° vs. best-of-2's 16.20° (confirms branch-selection artifact); oracle pose error 30.16mm is *worse* than best-of-2's 22.61mm (best-of-2's min-of-two-candidates acts as implicit denoising) |
| 32 | `SayL9p-3PD_4` | markdown | 4.6 | **(NEW)** Elbow-branch ceiling diagnostic rationale — analytic 2-link law-of-cosines check for whether both elbow branches are geometrically valid per target |
| 33 | `tDV7fjDMPHZJ` | code | 4.6 | **(NEW)** `elbow_branch_ceiling()` — solves both branches analytically (no network), counts unique-vs-ambiguous targets, computes a predicted accuracy ceiling. Result: 84.1% of test targets ambiguous (both branches valid) → predicted ceiling 57.9%, matching v8's observed 58.8% almost exactly. Confirms selection accuracy is at its ceiling, not a modelling gap |
| 34 | `Od3DQJzrfqVE` | markdown | 4.2 | **(v6)** Why the LR sweep exists — α isn't comparable across optimizers |
| 35 | `Q_qHfieOfvY3` | code | 4.2 | **(v6)** Sweep over `SWEEP_CONFIGS` (gd@0.05 as v5-repro + Adam at 3e-4/1e-3/3e-3/1e-2); logs each, survives divergence, prints ranked summary. Does **not** overwrite `parameters` |
| 36 | `dj1b2_LGD_pr` | markdown | 4.3 | **(v7)** Why the decay sweep exists — α₀ pinned to v6's winner, single-variable vs. v6, `v6-repro` doubles as regression test |
| 37 | `f1jC4InREEN6` | code | 4.3 | **(v7)** Sweep over `DECAY_SWEEP_CONFIGS` (none as v6-repro + inverse@0.01/0.03/0.1/0.3 + staircase@0.1/0.3/1.0); logs each, survives divergence, prints ranked summary by pose error. Does **not** overwrite `parameters`. Result: **every decay rate was worse than no decay** |
| 38 | `e7qfFh-89CiB` | markdown | 5. Diagnostic | Singularity-vs-precision-ceiling intro |
| 39 | `qq7hQwgaLUau` | code | 5. Diagnostic | Scatter plot: predicted vs. true angle, per joint |
| 40 | `5aBqP7IEd_VH` | code | 5. Diagnostic | Singularity proximity correlations, worst-decile stratification, branch-selection accuracy near each singularity |

---

## Current key variables (after running top to bottom)

`parameters`, `costs`, `train_time`, `final_cost` — from cell 28's `nn_model` call (v8 canonical config).
`VERSION`, `NOTES`, `layer_dims`, `BATCH_SIZE`, `NUM_ITERATIONS`, `OPTIMIZER`, `LEARNING_RATE`, `BETA1`, `BETA2`, `EPSILON`, `DECAY`, `DECAY_RATE`, `TIME_INTERVAL` — cell 28's run config (edit these to start a new experiment; bump `VERSION` for a new logged run).
`dev_pred_theta`/`test_pred_theta`, `*_joint_mae`, `*_pose_err`, `*_config_acc` — from cell 29's `evaluate()` calls.
`dev_oracle_pred`/`test_oracle_pred`, `*_oracle_mae`, `*_oracle_pose` — from cell 31's `evaluate_oracle()` calls (§4.5 diagnostic).
`model_a_training_log.csv` — written by cells 29 and 31, lives in Colab's ephemeral session (`/content`). **Not in the repo, and it does not survive a runtime restart** — unlike the notebook, saving to GitHub does not carry it along. As of this session it holds the v6/v7 rows plus the full v8 sweep history (coarse/fine/xlarge grids), even though the sweep cells themselves were deleted from the notebook after v8 was promoted — the log is currently the *only* place that raw per-run data survives outside `thesis_doc.md`'s summarized tables. Download it (or write to mounted Drive) before the runtime recycles if that raw data matters.

Note the log's schema changed in v6 (added `optimizer`/`beta1`/`beta2`/`epsilon`) and again in v7 (added `decay`/`decay_rate`/`time_interval`). `log_run_metrics` compares the existing header against its field list and renames the old file to `model_a_training_log_schema_<timestamp>.csv` on a mismatch rather than appending misaligned rows — so a stale log won't silently corrupt a new one.
