---
name: thesis_check
description: Checklist of do's and don'ts for the IK_MDN_3R_3D thesis project, compiled from explicit user feedback and corrections given during this project. Consult before making an architecture/scope decision, before choosing between multiple technical options, before editing a notebook (local or Colab), before writing a performance review, or when the user invokes "/thesis_check" or asks to "check the rules"/"review guidelines" for this project.
---

# thesis_check — Project Do's and Don'ts

A running checklist of behavioral rules the user has explicitly given for this project. Re-read this before taking any action it covers. Append new entries here (don't duplicate into other files) whenever the user gives new feedback of this kind — keep each entry grounded in what was actually said, not inferred.

## Decision-making

- **Don't pick unilaterally between open technical options — ask.** When there are multiple valid next steps (e.g. hard-assignment loss vs. 2-head architecture), present the choice and wait for the user's answer instead of assuming one.
- **Don't assume the architecture is fixed in advance.** The user explicitly corrected an assumption that the solution "necessarily" had to be a mixture-density-based algorithm — stay open to alternatives when the question is about approach, not implementation of an already-agreed design.
- **Ask clarifying questions on ambiguous ML-design scope rather than guessing.** Before building something with several reasonable interpretations (e.g. "build the model from scratch" — which parts, what engine, what optimizer), ask; don't silently pick defaults for consequential choices.

## Answering / reviewing

- **Follow the exact framing requested, even if it overrides your default.** When the user says not to frame a review around the thesis pass/fail thresholds (mean angle < 5°, pose error < 5mm) and instead wants a substantive technical review, do that — don't default back to requirement-check framing.

## From-scratch model build (learning track)

These are the standing constraints for the pure-NumPy "build it from scratch" notebook work (as opposed to the main PyTorch MDN notebook):
- **Pure NumPy, manual backprop** — no autograd, no PyTorch, for this track.
- **No mixture density network** — single deterministic output head, until the user asks to add one.
- **No optimizer beyond plain gradient descent** — no Adam/momentum yet. Get a simple, correct baseline working first; optimize later only when asked.

## Colab workflow

- **Don't open a new Colab browser tab.** Connect to whichever notebook tab the user already has open; ask which one if unclear rather than opening a new one.
- **Any requested code change goes into the Google Colab tab that's already open** (currently `August_thesis.ipynb`, the from-scratch Model A), edited live via `colab-proxy-mcp`. Don't make the change locally/in a different notebook unless the user says otherwise.

## Documentation habits

- **Save `handoff.md` to the project root**, not the OS temp directory the `handoff` skill defaults to.
- **Log conceptual/technical Q&A in `QUESTIONS.md`** at the project root, organized into dated sections, with short answers.
- **Keep `CLAUDE.md`** (project root) up to date as the durable context file loaded at the start of future sessions — architecture facts, known limitations, gotchas already hit. Don't duplicate content across `CLAUDE.md`, `handoff.md`, and `QUESTIONS.md`; each has a distinct job (see `CLAUDE.md`'s "Where to look for more" section).
