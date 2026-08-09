# Project: IK_MDN_3R_3D — Thesis Inverse Kinematics Models

## What this is
A thesis project solving inverse kinematics for a 3R robot arm in 3D: predict joint angles `[theta1, theta2, theta3]` from a target end-effector position `[x, y, z]`. `theta1` is base yaw (full circle); `theta2`, `theta3` move a 2-link planar arm in the `(r, z)` plane swept out by that yaw.

```
r = L1 + L2*cos(theta2) + L3*cos(theta2 + theta3)
z = L2*sin(theta2) + L3*sin(theta2 + theta3)
x = r*cos(theta1), y = r*sin(theta1)
L1=1.0, L2=1.0, L3=0.5
theta1 in (-pi, pi), theta2/theta3 in (-pi/2, pi/2)
```

Thesis accuracy target: mean joint-angle error < 5°, end-effector pose error < 5mm.

## Two parallel model tracks — don't conflate them
1. **Main thesis model** (PyTorch, Mixture Density Network): local notebook
   `IK_MDN(3R in 3D)_tuned\notebooks\IK_MDN_3R_3D_Colab.ipynb`. Predicts a
   mixture of Gaussians over sin/cos-encoded angles (6-dim output), trained
   with NLL + FK-consistency loss. Full detail, architecture, and latest
   diagnostic numbers are in `handoff.md` — read that before touching this
   notebook.
2. **From-scratch learning track** (pure NumPy, manual backprop, no
   autograd/optimizer): Google Colab notebook `August_thesis.ipynb`, edited
   live via the `colab-proxy-mcp` MCP server. This mirrors the Andrew Ng
   Course 1/2 assignment structure (`initialize_parameters` →
   `forward_propagation` → `compute_cost` → `backward_propagation` →
   `update_parameters` → `nn_model`) applied to this exact IK problem, as a
   deliberately simplified single-output-head, plain-gradient-descent
   baseline the user builds up in stages before adding complexity.

## Known limitations of the from-scratch (NumPy) track — see QUESTIONS.md
These are intentional simplifications, not bugs, unless stated otherwise:
- **Sin/cos encoding for `theta1` only** (added) — target is
  `[sin(theta1), cos(theta1), theta2, theta3]`, decoded via `atan2`.
  `theta2`/`theta3` stay raw since they're limited to (-90°, 90°) and never
  wrap. The angle-error metric (`mae_degrees`) uses circular distance
  (wraps to `(-pi, pi]`) — don't revert it to a plain difference, or error
  near the `theta1` wrap boundary will be wildly overstated.
- **No singularity-aware sampling/loss** — data is sampled uniformly; error
  is expected to grow near the elbow singularity (`theta3 ≈ 0`, full
  extension) and near the base/shoulder singularity (`r ≈ 0`).
- **Single deterministic output head, plain MSE** — this cannot represent
  the elbow-up/elbow-down branch ambiguity in the IK problem; it will
  average conflicting solutions where two branches produce nearly the same
  end-effector position. This is the core reason the main thesis model uses
  an MDN instead.

## Gotcha already hit once — check backward-prop math carefully
In `backward_propagation`, the top-level gradient must be `dAL = AL - Y`
(no `/m`). The `1/m` averaging belongs only inside `linear_activation_backward`'s
`dW`/`db` computation. Putting `/m` in `dAL` as well double-counts it at
every layer during backprop, silencing gradients by roughly another factor
of `m` — this happened once and produced a model that looked like it was
training (smoothly decreasing loss) but was actually barely moving from
random init. If a "from scratch" NN in this repo trains suspiciously slowly
or converges worse than a constant-mean predictor, check this first.

## Second gotcha of the same class — Adam's `t` counts updates, not epochs
`update_parameters_with_adam`'s bias correction divides by `1 - beta**t`, so
`t` must be the **global parameter-update counter**, incremented once per
mini-batch. Under mini-batching (batch 256 over 24,000 examples) that is 94
increments per epoch — passing the epoch index instead leaves the correction
wrong by that factor for the whole run. Like the `dAL` bug above, this trains
*smoothly* while being wrong. `nn_model` owns `t` inside its `_step` helper and
prints the final count; a 100-epoch run must report **9,400 parameter updates**.
Also note Adam's learning rate is not comparable to plain GD's (its step is
magnitude-normalized to roughly ±α per parameter) — don't carry α across
optimizers; see `thesis_doc.md` §3.11/§4.2.

## Third gotcha of the same class — learning-rate decay indexes epoch, not Adam's `t`
Decay (`update_lr`/`schedule_lr_decay`, v7) recomputes α once per **epoch**;
Adam's bias correction uses `t`, the global **parameter-update counter**. Both
counters are live in the same training loop and are trivially swappable —
passing `t` into a decay schedule instead of the epoch index decays α roughly
94x too fast at batch 256/24,000 examples (t reaches ~9,400 over 100 epochs vs.
99 epochs), collapsing the effective step to near zero almost immediately.
Like the two bugs above, this still produces a smooth, plausible-looking loss
curve. `nn_model`'s final print line reports α₀ -> α_final so this is checkable
at a glance; see `thesis_doc.md` §3.12. Separately, note that in the v7 sweep
**every decay rate tested made pose error worse**, not better — this is not a
bug, just a real (and reported) result: v6's α₀ was already the smallest value
in the §4.2 sweep and not in the "bouncing near the minimum" regime decay helps
with, see `thesis_doc.md` §4.3 for the mechanism.

## Where to look for more
- `handoff.md` — detailed session handoff: architecture history, the
  notebook-corruption-and-rebuild incident (avoid incremental `NotebookEdit`
  calls on large notebooks — cell-id resolution can drift after inserts;
  prefer rebuilding from clean JSON if things get inconsistent), latest
  diagnostic numbers, and open next steps. Save updates here at the project
  root, not the OS temp dir the `handoff` skill defaults to.
- `QUESTIONS.md` — running, dated log of conceptual Q&A about the models
  (append new sections here, don't duplicate into this file).
- `thesis_doc.md` — the thesis-writing reference: every equation used in
  implementation (FK, dataset generation, Model A's from-scratch NN math,
  Model B's mixture-density math), a results table per experiment version,
  and a dated change log. Update this whenever a new equation, result, or
  methodology change happens — it's meant to be pasted from directly into
  the thesis, so keep it accurate and don't duplicate its content into
  `handoff.md`/`QUESTIONS.md`.
- `lessons/`, `MISSION.md`, `RESOURCES.md`, `NOTES.md` — a stateful `/teach`
  track that has partially diverged from direct notebook work; check
  `handoff.md`'s "Teaching workspace" section for current status.
- `.claude/skills/thesis_check/SKILL.md` — project-scoped checklist of
  do's and don'ts the user has explicitly given (decision-making,
  from-scratch build constraints, Colab workflow, documentation habits).
  Consult it before architecture/scope decisions, notebook edits, or
  writing a performance review. Only available in this project directory.
- `notebook_map.md` — cached index of the Colab notebook's cell IDs
  (`august_thesis_claude`), so edits can target a cell directly instead of
  fetching the whole notebook first. Kept in sync via
  `.claude/skills/notebook_map/SKILL.md` — consult and apply that skill
  any time a notebook cell is added/updated/deleted/moved, or the user
  reports a manual edit made in the browser.

## Working conventions
- The Colab MCP connection (`colab-proxy-mcp`) can silently disconnect
  between sessions — if its tools 404/timeout, ask the user to run `/mcp`
  to reconnect, then retry `open_colab_browser_connection`.
- Don't open a new Colab tab — connect to the one already open.
- Keep `handoff.md` and `QUESTIONS.md` at the project root; don't move them.
