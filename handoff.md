# Handoff: IK_MDN_3R_3D — Thesis IK Models (Model A: from-scratch NumPy, Model B: PyTorch MDN)

## Who this is for
A fresh agent picking up work in this conversation/workspace. Read this, then `CLAUDE.md` (durable project facts, gotchas already hit) and `thesis_doc.md` (all equations + full results history), before doing anything — don't re-derive context that's already written down.

**Save location note:** this file lives in the project directory (`handoff.md` at the project root), not the OS temp dir the `handoff` skill defaults to. The user explicitly chose that location for this project (also recorded in cross-session memory) — keep saving updates here, don't move it to temp, don't re-ask.

## Project context
Working directory: `D:\Research\thesis code\IK_MDN(3R in 3D)_tuned-20260504T200213Z-3-001\`

Thesis project: inverse kinematics for a 3R robot arm in 3D. **Two parallel model tracks — see `CLAUDE.md` for the full split, don't conflate them:**
1. **Model B** (main thesis model): PyTorch Mixture Density Network, local notebook `IK_MDN(3R in 3D)_tuned\notebooks\IK_MDN_3R_3D_Colab.ipynb`. **Untouched this session** — all work below is on Model A. Its architecture, equations, and latest diagnostics are documented in `thesis_doc.md` §5, not repeated here.
2. **Model A** (from-scratch learning track, the active thread this session): pure NumPy, manual backprop, plain gradient descent, in the Google Colab notebook `August_thesis.ipynb`, edited live via the `colab-proxy-mcp` MCP server. **This is where all this session's changes happened.**

The `colab-proxy-mcp` connection can silently disconnect between sessions — if its tools time out/fail, ask the user to run `/mcp` to reconnect, then retry `open_colab_browser_connection` (see `CLAUDE.md`'s Working Conventions). Don't open a new Colab tab — connect to whichever one is already open (currently `August_thesis.ipynb`).

## Where everything actually lives (don't duplicate — read these instead)
- **`thesis_doc.md`** (project root) — the authoritative record: every equation for both models (FK, Model A's init/forward/cost/backward/update, Model B's mixture-density formulation), a versioned results table for Model A (v0 through v3, see below), Model B's latest diagnostics, known limitations, and a dated change log. **Read this first for any technical/numeric detail** — this handoff intentionally doesn't repeat its tables.
- **`QUESTIONS.md`** (project root) — dated log of conceptual Q&A the user has asked, with short answers. Standing instruction from the user: **log every question they ask here going forward**, not just at session boundaries.
- **`CLAUDE.md`** (project root) — durable facts for any new session: robot/FK parameters, the two-track split, known limitations, and gotchas already hit (notably: a backward-prop `1/m` double-division bug, documented so it isn't reintroduced).
- **`.claude/skills/thesis_check/SKILL.md`** — project-scoped (only visible in this project folder) checklist of do's/don'ts the user has explicitly given: don't decide unilaterally on open technical forks, follow the exact framing requested, Model A's standing build constraints, Colab workflow rules, documentation habits. **Consult before any architecture decision or notebook edit.**

## What happened this session (Model A only)
1. Built Model A from scratch per the user's spec: parameter init, forward prop, cost, backward prop, update, full training loop — all hand-derived NumPy, no autograd. See `thesis_doc.md` §3 for every equation.
2. Found and fixed a real bug: `dA^[L]` was divided by `m` at the top of backprop *and again* inside the per-layer `dW`/`db` formulas, silencing gradients by an extra factor of ~`m`. Fixed to `dAL = AL - Y` (single division, inside `dW`/`db` only). Documented in `CLAUDE.md` and `thesis_doc.md` §3.5 as a pitfall to watch for if a "from scratch" NN in this repo trains suspiciously slowly.
3. **v2**: added sin/cos encoding for `theta1` only (removes its ±180° wraparound discontinuity; `theta2`/`theta3` don't wrap so stay raw). Result: `theta1` MAE 15.3°→1.6°; pose error 528mm→194mm; `theta2`/`theta3` unaffected as expected (`thesis_doc.md` §4).
4. Created `QUESTIONS.md` and `.claude/skills/thesis_check/SKILL.md` (see above).
5. **v3**: ran a `/grilling`-style decision session with the user over how to remove the `theta2`/`theta3` elbow-branch ambiguity. Landed on: `elbow_config` (±1, derived from `sign(theta3)`) as an **oracle input at training time only**; at inference, `predict_ik` queries the network with both `elbow_config` values and returns whichever decoded solution reprojects (via FK) closer to the requested `(x,y,z)` — caller never supplies the branch. Implemented in the notebook (input layer 3→4 dims).
6. **v3 result — important, non-obvious, fully written up in `thesis_doc.md` §4 v2→v3 interpretation:** training cost dropped ~10x (0.470→0.059), pose error nearly halved (193.9mm→101.6mm), but `theta2`/`theta3` angle-MAE against the *sampled test label* barely moved, and best-of-2 branch-selection accuracy was only 59% (near chance). Mechanism: once both branches are learned well, both reproject close to the target, so "pick smaller reprojection error" can't reliably recover the *specific* label's branch — it's finding *a* valid solution, just not always the recorded one. This is treated as evidence (not just hypothesis) for something already flagged as an open question about Model B: some of its "branch confusion" may be the same metric-framing artifact, not pure model error.
7. User reframed the priority explicitly: **the thesis cares about end-effector (pose) error, not angle-MAE-per-label.** Under that framing, v3 is a real win (pose error nearly halved) even though angle-MAE didn't move — this reframing is now recorded in `thesis_doc.md` §6 as a new open question (“is angle-MAE-per-label even the right metric for an intrinsically one-to-many problem?”).
8. Ran a singularity diagnostic on v3's remaining ~100mm pose error (elbow proxy `|sin(theta3)|`, base proxy `r`). **Result: neither singularity drives it** — correlations near zero/weak, worst-error decile not concentrated near either. Pose error is actually slightly *lower* near the elbow singularity (a real, explainable effect: the singular direction is exactly where angle changes barely move the end-effector — angle uncertainty grows there while position sensitivity shrinks; this does not contradict Model B's earlier angle-MAE-based finding, it's a different axis). Full writeup: `thesis_doc.md` §4.1. Conclusion: the remaining pose error looks like a **general regression precision ceiling** (plain GD, 3000 iterations, cost curve still decreasing at the end), not a structural problem.
9. Asked the user "what would you recommend next?" and gave a recommendation: **add Adam optimizer + longer training budget** (targeting the diagnosed under-convergence, not capacity or singularities), to be logged as **v4**. **The user has not yet confirmed this — the conversation ended (with this `/handoff` invocation) right after the recommendation was given, before they said go/no-go.**

## Blocking / immediate next step
**Get the user's go-ahead on implementing Adam (+ longer training) as v4 before touching the notebook.** Don't assume yes and don't pick a different direction unilaterally (per `thesis_check`). If they confirm, implement in the Colab tab (per their standing instruction: code changes always go into the open Colab tab), keep the existing plain-GD `update_parameters` intact for comparison (add Adam as a new path/function), retrain, re-run the evaluate + singularity-diagnostic cells, and log v4's results into `thesis_doc.md` §4 in the same table/interpretation format as v0–v3.

## Suggested skills for the next session
- **`thesis_check`** (project-scoped skill, this project folder only) — consult before implementing v4 or any other architecture/scope decision.
- **`mattpocock-skills:grilling`** — if another open-ended design fork comes up (as it did for the `elbow_config` decision this session), this walked the decision tree one question at a time with recommendations and worked well; reuse the pattern.
- **`dataviz`** — the singularity-correlation and branch-selection-accuracy diagnostics (`thesis_doc.md` §4.1) are currently printed numbers only; if the user wants them as scatter/box plots for the thesis figures, this skill is a good fit.
- Do not invoke `codebase-memory`, `security-review`, or unrelated skills — out of scope for this notebook/ML work.

## Open threads / next actions (in priority order)
1. **Blocking**: user's go/no-go on Adam + longer training as v4 (see above).
2. Not yet done: re-verify the v3 branch-confusion finding directly on **Model B** (the sign-flipped/pose-error filter check flagged in `thesis_doc.md` §6 item 4) — v3's result makes this more urgent/interesting to actually run, not just hypothesize about.
3. Low priority, still outstanding from earlier sessions: Model B's LR-sweep cell (Section 10 of its notebook) has never been run; the stateful `/teach` lesson track (`lessons/0002-*.html`) has been superseded by direct diagnostics work and its status with the user was never resolved (still an open question if that track gets revisited).

## Redactions
None required — no credentials, API keys, or secrets appeared in this session. No personally identifiable information appeared in any user-facing artifact.
