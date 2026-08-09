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
- ~~**No optimizer beyond plain gradient descent** — no Adam/momentum yet. Get a simple, correct baseline working first; optimize later only when asked.~~ **Lifted 08-08-26**, when the user asked to plan Adam. The intent was satisfied as written: the plain baseline was verified correct first (the `dAL` bug was found and fixed under it), and only then was Adam added. Note how it was added, since the *spirit* of this rule still applies to future additions — plain GD stayed the default (`optimizer="gd"`), `update_parameters` was left untouched, and a `v5-repro` regression run confirmed the earlier path still reproduces exactly.
- ~~**Learning-rate decay** — deliberately excluded from v6 to keep the Adam comparison single-variable; required the user to ask first.~~ **Lifted 09-08-26**, when the user asked for it. Added the same way as Adam: `decay=None` stayed the default (v0–v6 stay bit-identical), and a `v6-repro` regression run confirmed the earlier path still reproduces exactly before any decay row was trusted. Logged as v7 via a single-variable sweep (α₀ pinned to v6's winner) — result was a genuine negative: every decay rate tested made pose error worse, not better, because v6's α₀ was already too small for decay to help at (`thesis_doc.md` §4.3). Worth remembering for any future optimizer-adjacent addition: a negative result from a properly isolated sweep is still a complete, reportable outcome — it does not mean the sweep needs to be re-run wider or the config searched further, unless the user asks for that specifically.
- **Keep each version single-variable where possible.** The user chose "Adam only, everything else pinned to v5" over bundling learning-rate decay in, and chose a logged LR sweep over a single run, specifically so the results table wouldn't gain another "two things changed at once" caveat like the ones already on v1→v2 and v3→v4. Default to isolating one change per logged version.
- **A negative sweep result can motivate the next sweep — follow that thread when the user asks.** v7's negative decay result (§4.3) predicted decay would help at a *larger* α₀; the user then explicitly asked to test that (09-08-26), escalating the search scope themselves across three passes ("explore further", "even larger learning rate") until a genuine, bracketed optimum was found (v8, §4.4: α₀=2e-3→~3e-4, 22.61mm, beating v6/v7 by 29%). Don't propose widening a search unprompted, but do follow it eagerly once asked, and keep tracking the running "best result to beat" explicitly across passes so nothing gets lost mid-search (the user said this explicitly: "don't lose the current result you found").
- **After promoting a sweep winner, ask before deleting the exploratory cells — don't assume either way.** When the user said "revoke the notebook changes to this best" after v8's promotion, the phrasing was genuinely ambiguous (delete vs. something else), so it was checked via AskUserQuestion rather than guessed. The user's answer: delete all exploratory sweep cells from the notebook, keep only the winning canonical config — the full sweep history lives in `thesis_doc.md` only, not duplicated in the notebook. This is now the established pattern for future promote-and-clean-up requests, but confirm scope again if the phrasing is ambiguous next time too.

## Colab workflow

- **Don't open a new Colab browser tab.** Connect to whichever notebook tab the user already has open; ask which one if unclear rather than opening a new one.
- **Any requested code change goes into the Google Colab tab that's already open** (currently `August_thesis.ipynb`, the from-scratch Model A), edited live via `colab-proxy-mcp`. Don't make the change locally/in a different notebook unless the user says otherwise.

## Documentation habits

- **Save `handoff.md` to the project root**, not the OS temp directory the `handoff` skill defaults to.
- **Log conceptual/technical Q&A in `QUESTIONS.md`** at the project root, organized into dated sections, with short answers.
- **Keep `CLAUDE.md`** (project root) up to date as the durable context file loaded at the start of future sessions — architecture facts, known limitations, gotchas already hit. Don't duplicate content across `CLAUDE.md`, `handoff.md`, and `QUESTIONS.md`; each has a distinct job (see `CLAUDE.md`'s "Where to look for more" section).
