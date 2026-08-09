# Handoff: IK_MDN_3R_3D — Thesis IK Models (Model A: from-scratch NumPy, Model B: PyTorch MDN)

## Who this is for
A fresh agent picking up work in this conversation/workspace. Read this, then `CLAUDE.md` (durable project facts, gotchas already hit) and `thesis_doc.md` (all equations + full results history), before doing anything — don't re-derive context that's already written down.

**Save location note:** this file lives in the project directory (`handoff.md` at the project root), not the OS temp dir the `handoff` skill defaults to. The user explicitly chose that location for this project (also recorded in cross-session memory) — keep saving updates here, don't move it to temp, don't re-ask.

## Project context
Working directory: `D:\Research\thesis code\IK_MDN(3R in 3D)_tuned-20260504T200213Z-3-001\`

Thesis project: inverse kinematics for a 3R robot arm in 3D. **Two parallel model tracks — see `CLAUDE.md` for the full split, don't conflate them:**
1. **Model B** (main thesis model): PyTorch Mixture Density Network, local notebook `IK_MDN(3R in 3D)_tuned\notebooks\IK_MDN_3R_3D_Colab.ipynb`. **Untouched this session** — all work below is on Model A. Its architecture, equations, and latest diagnostics are documented in `thesis_doc.md` §5, not repeated here.
2. **Model A** (from-scratch learning track, the active thread this session): pure NumPy, manual backprop, in the Google Colab notebook `august_thesis_claude.ipynb` (live-edited via `colab-proxy-mcp`). **This is where all this session's changes happened.**

The `colab-proxy-mcp` connection can silently disconnect between sessions — if its tools time out/404, ask the user to run `/mcp` to reconnect, then retry `open_colab_browser_connection` (this happened repeatedly at the start of this session — took ~9-10 reconnect attempts before it succeeded; don't be surprised if it needs several tries, see [[feedback_colab_mcp_flaky]] in cross-session memory). Don't open a new Colab tab — connect to whichever one is already open (currently `august_thesis_claude.ipynb`).

## Where everything actually lives (don't duplicate — read these instead)
- **`thesis_doc.md`** (project root) — the authoritative record: every equation for both models, a versioned results table for Model A (v0 through **v8, the current best**), Model B's latest diagnostics, known limitations, and a dated change log. **Read this first for any technical/numeric detail** — §4.4 in particular is now the *only* place the full v8 sweep data lives (the notebook cells that generated it were deleted after promotion, see below).
- **`QUESTIONS.md`** (project root) — dated log of conceptual Q&A the user has asked, with short answers. Standing instruction: log every question they ask here going forward. No new conceptual questions this session (the v8 exploration was directive, not Q&A).
- **`CLAUDE.md`** (project root) — durable facts: robot/FK parameters, the two-track split, known limitations, and three gotchas already hit (the `dAL`/m double-division bug, Adam's `t`-counts-updates-not-epochs bug, the decay-epoch-vs-Adam's-`t` counter-swap risk).
- **`.claude/skills/thesis_check/SKILL.md`** — project-scoped checklist of do's/don'ts the user has explicitly given. **Consult before any architecture decision or notebook edit.**
- **`notebook_map.md`** — cached index of the live Colab notebook's cell IDs; kept in sync via `.claude/skills/notebook_map/SKILL.md`. Fully up to date as of this session (37 live cells — net unchanged from the post-v7 state, since v8's exploration cells were added then removed).

## What happened this session (Model A only) — v7 (decay, negative result) then v8 (decay + larger α₀, new best)

### v7 — Learning-Rate Decay (initial implementation + negative result)
1. Implemented both Course 2 decay schedules in the live notebook: `update_lr` (§3.8, inverse-time) and `schedule_lr_decay` (staircase), plus a `decay`/`decay_rate`/`time_interval` extension to `nn_model` (§3.9) defaulting to no-op so v0–v6 stay bit-identical. A new `_epoch_lr` helper recomputes α once per epoch — deliberately never from Adam's `t`, the new gotcha now in `CLAUDE.md`.
2. Verified via a `v6-repro` regression run (exact match to v6: cost 0.003168, pose 31.61mm) before trusting anything else.
3. §4.3 sweep (8 runs, α₀=3e-4 pinned to v6's winner): **every decay rate tested was worse than no decay**, monotonically so. Logged as **v7**, whose numbers equal v6's since `decay=None` won. This was a genuine, well-isolated negative result, not a failed experiment — see `thesis_doc.md` §4.3 for the full mechanism (α₀ was already the smallest/best-tuned rate, so decay just under-trained it further).

### v8 — Decay paired with a larger α₀ (user-directed follow-up, new best result)
The user explicitly asked to test §4.3's untested prediction: pair decay with a *larger* α₀, where it's actually supposed to help. Ran as a three-pass sweep, escalating in scope on the user's instruction each time ("explore further first", "even larger learning rate", "don't lose the current result"):

1. **Coarse grid** (§4.4 pass 1): α₀∈{1e-3, 3e-3, 1e-2} × {none, mild decay, targeted decay →~3e-4}. **Decay won at every α₀** — the opposite of v7. Best: α₀=3e-3 targeted, 24.35mm (vs. v6/v7's 31.61mm).
2. **Fine grid** (§4.4 pass 2, user asked for this to refine around the optimum): Grid A (α₀ 1e-3–5e-3, 8 steps, all →~3e-4) found a shallow, well-behaved optimum spanning α₀∈[2e-3, 3.5e-3]; **new best: α₀=2e-3, 22.54mm.** Grid B (decaying further to ~1e-4 instead) was uniformly worse — confirms ~3e-4 is specifically the right value to settle near, not arbitrary.
3. **Even-larger-α₀ grid** (§4.4 pass 3, user asked for this too): α₀ up to 5e-2, all →~3e-4. Degradation continued monotonically and became catastrophic (112mm at α₀=5e-2) — a *different* failure mode from v7's (early instability decay arrives too late to fix, vs. v7's under-training). No second good regime found. This closes the search: the α₀∈[2e-3,3.5e-3] region is a genuine, bracketed optimum, not a lucky point.
4. **Promotion** (user's explicit instruction: "promote... and revoke the notebook changes to this best"): asked the user to clarify scope via AskUserQuestion (delete sweep cells vs. keep them), user chose delete-all. Updated the canonical training cell (`4-ONOXF2LQjR`) to `VERSION="v8"`, α₀=2e-3, decay="inverse", decay_rate=0.0572 (→~3e-4 by epoch 99); re-ran training, evaluate/log, and both §5 diagnostic cells. **Final v8 result: test pose error 22.61mm, angle MAE 16.20°, selection 58.8% — a 29% improvement over v6/v7.** Singularity diagnostic re-confirmed the standing conclusion for a 4th version (correlations still weak: 0.054 elbow, 0.059 base).
5. **Deleted all 6 exploratory sweep cells** (3 markdown+code pairs covering the coarse/fine/xlarge grids) from the notebook per the user's request — the notebook now shows only the clean path to v8, not the search that found it. The full sweep data (25 runs across 3 passes) is preserved **only** in `thesis_doc.md` §4.4, which is now the authoritative record — there is no other copy.
6. Documentation fully updated: `thesis_doc.md` (§4.4 full write-up, v8 results row, §4.1's standing-conclusion note extended to 4 versions, §6 item 8 updated), `notebook_map.md` (re-synced to reflect deletions — net 37 cells, same as post-v7), this file.

## Key result to carry forward
**v8 is the new best model in the entire from-scratch track: α₀=2×10⁻³, inverse decay to ≈3×10⁻⁴ by epoch 99, test pose error 22.61mm** (vs. v6/v7's 31.61mm, a 29% improvement; vs. v0's 3230mm at project start). This came directly from testing a prediction that a prior negative result (v7) made explicit — a clean example of a null finding motivating the experiment that superseded it, worth highlighting in the thesis narrative.

**Important asymmetry to know about:** the notebook itself only shows v8's canonical training cell — none of the 25-run search that found it. If anyone (including a future session) wants to see the sweep mechanics, the *only* source is `thesis_doc.md` §4.4's tables and prose. Don't assume the notebook documents its own history the way it did for v6's §4.2 sweep (which is still in the notebook, untouched).

## Not yet done
- **`model_a_training_log.csv` was not downloaded from Colab's ephemeral `/content`** this session — it currently holds the full v6/v7 rows plus all 25 v8 sweep rows (coarse/fine/xlarge), and does not survive a runtime restart. Since the notebook cells that could regenerate this data are now deleted, **this CSV is the only remaining raw-data copy of the v8 sweep** (thesis_doc.md has the summarized/rounded tables, not raw per-run CSV rows). Flag this to the user with some urgency if the raw data matters for the thesis appendix.
- **The user has not yet saved the notebook to GitHub** (*File → Save a copy in GitHub*, path `IK_MDN(3R in 3D)_tuned/notebooks/august_thesis_claude.ipynb`, no repo-name prefix, no trailing space) — the local `.ipynb` on disk is still the pre-v7 version, missing both v7 and v8 entirely. Ask the user to do this, then pull locally, before treating the local file as current.
- Everything else outstanding from the prior handoff is unchanged: re-verify the v3 branch-confusion finding directly on **Model B**; Model B's LR-sweep cell (its own notebook, Section 10) has never been run; the `/teach` lesson track's status was never resolved.

## Suggested skills for the next session
- **`thesis_check`** — consult before any further architecture/scope decision on Model A.
- **`notebook_map`** — apply immediately after any further Colab cell edits.
- Do not invoke `codebase-memory`, `security-review`, or unrelated skills — out of scope for this notebook/ML work.

## Redactions
None required — no credentials, API keys, or secrets appeared in this session.
