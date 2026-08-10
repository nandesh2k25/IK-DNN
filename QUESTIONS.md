# Questions Log

A running log of conceptual/technical questions asked during this project, with short answers, organized by date. Append new sections at the bottom as new sessions happen.

---

## 06-08-26: Thursday

**Q1.** Is this model handling the periodic nature of angles (like 45° and -315° being in the same direction)?

**A:** No. The from-scratch NumPy model predicts raw `theta` values (mean/std normalized only, no sin/cos encoding). `theta2`/`theta3` are restricted to (-90°, 90°) so periodicity doesn't apply to them, but `theta1` spans the full (-180°, 180°) circle. Near the ±180° wrap boundary, physically close target positions get labeled with numerically far-apart `theta1` values, forcing the network to fit a discontinuity — expect elevated error there. Sin/cos encoding (already used in the PyTorch MDN notebook) fixes this; it is not yet in the NumPy build.

**Q2.** Is it taking into consideration the singularity (elbow/shoulder & base singularities) issues?

**A:** No. Training data is sampled uniformly at random over joint limits with no extra density near singular configurations, and the loss has no special weighting there. Two singularities are relevant to this arm: the elbow singularity near `theta3 ≈ 0` (full extension, Jacobian loses rank in the r-z plane) and a base/shoulder-type singularity near `r ≈ 0` (arm folded onto the z-axis, where `theta1` becomes ill-defined). Expect larger errors near both, consistent with the -0.644 error-vs-`|sin(theta3)|` correlation found earlier in the MDN model.

**Q3.** Is it trying to average multiple solutions (same end-effector pose)?

**A:** Yes — this is the main structural limitation. It's a single deterministic output head trained with plain MSE. Where two training examples land at nearly the same `(x,y,z)` but come from different elbow-up/elbow-down branches, MSE's optimal prediction is the *average* of the two `theta2`/`theta3` solutions, which is generally not a valid IK solution at all. This is exactly why the thesis model uses a mixture density network instead of a point-estimate regressor, and is likely the dominant cause of the large `theta3` error measured on this baseline.

**Q4.** For the `elbow_config` input, would it make a difference to encode it as `-1`/`+1` vs `0`/`1`?

**A:** Not much — the network learns either way. The only real difference is minor training smoothness: `-1`/`+1` is zero-centered (symmetric), matching how the other inputs are already set up, which tends to make training slightly more even; `0`/`1` isn't centered, so one of the two configs looks like "nothing extra" to that input's weight, making it marginally harder to weigh evenly. It doesn't change whether the approach works, just training quality at the margins. Went with `-1`/`+1` for consistency with the rest of the notebook's conventions.

---

## 08-08-26: Saturday

**Q5.** This notebook is very slow currently, how do I make it fast?

**A:** The slowness was the training cell itself — full-batch gradient descent over all 24,000 training examples through a `[4,256,128,64,4]` network, repeated 3000 times, meant every single iteration was one large matmul over the whole dataset. Fix (v5): switch to mini-batch gradient descent (batch size 256, via a new `random_mini_batches` helper mirroring the Andrew Ng Course 2 convention) — parameters now update ~94 times per epoch instead of once. Important nuance: this doesn't reduce the total arithmetic per epoch (an epoch of mini-batches still touches all 24,000 examples), the speedup comes from converging in far fewer epochs than 3000. In practice: 100 epochs finished in 65.6s and reached a lower cost (0.0116) and better test pose error (48.89mm) than the old full-batch run achieved in 3000 iterations (cost 0.040, pose error 74.08mm) — both faster and more accurate.

**Q6.** What do you mean by adding learning-rate decay?

**A:** Shrinking the learning rate `α` as training progresses instead of holding it fixed for all 100 epochs. The Course 2 form is `α_t = α_0 / (1 + decay_rate · t)` with `t` the epoch number, so α might start at 0.05 and be ~0.005 by epoch 100. It helps mini-batch training specifically: each mini-batch gives a *noisy* gradient estimate, and early on that noise is irrelevant next to how far the parameters still have to travel, so large steps are fine. Late in training, near the minimum, a fixed large α makes the parameters bounce around inside a region of noise rather than settling; decaying α shrinks that final wandering radius. **It is not redundant with Adam**, which is the natural assumption — Adam adapts the step *per parameter* (rescaling each weight's step by its own recent gradient history) but does not shrink the *global* step size over time; the α passed in still sets the overall scale throughout. The two are orthogonal and routinely combined (this is what PyTorch's `Adam` + an `LRScheduler` does). Decided to keep it out of v6 so Adam's effect stays single-variable; it's now logged as `thesis_doc.md` §6.8 as the remaining Course 2 optimizer topic.

**Q7.** What do you mean by implementing Adam only — does it mean you will implement momentum and RMSprop together (which is described in Course 2)?

**A:** Yes — Adam *is* momentum and RMSprop fused into one update rule, so implementing it necessarily means writing both mechanisms. Momentum supplies the numerator (`v`, an EW average of the gradient); RMSprop supplies the denominator (`s`, an EW average of the squared gradient); bias correction fixes the fact that both start at zero. Drop `β₂` and you have plain momentum; drop `β₁` and you have RMSprop. The real question was whether to expose them as *separately runnable* optimizers — Course 2 builds momentum as its own graded exercise before Adam, which here would have meant `optimizer="gd"|"momentum"|"adam"` and a three-row ablation. Chose **Adam only**: both mechanisms are implemented, but only as Adam's internal `v`/`s` accumulators, giving a single GD→Adam row in the results table (`thesis_doc.md` §4). See §3.11 for the full derivation, including the non-obvious detail that the two bias corrections do *not* cancel in the ratio `v̂/√ŝ` — omitting them makes the first step ~3.16x too **large**, not too small.

---

## 10-08-26: Monday

**Q8.** Elbow-selection accuracy (58.8%) and pose error are both still not great — is this fixable, and would converting Model A into an MDN help?

**A:** The two symptoms have different causes and neither is fixed by an MDN. $\theta_1$ MAE (0.48°, branch-independent) is ~24–76x better than $\theta_2$/$\theta_3$ MAE (11.70°/36.43°, branch-dependent) — that gap is branch-selection noise, confirmed directly by the §4.5 oracle-conditioned diagnostic: conditioning on the true `elbow_config` drops mean angle MAE to 1.19°, with $\theta_2$/$\theta_3$ landing near $\theta_1$'s level. So the regression itself already meets the thesis's 5° target; the reported angle MAE is a metric artifact of best-of-2 selection missing the sampled label ~41% of the time (largely a coin flip — see the elbow-accuracy ceiling estimate below). Pose error is a *separate*, genuine regression-precision gap (v8: 22.61mm mean, ≈0.86° of effective angle error, consistent with $\theta_1$'s precision) — best-of-2 already takes the min over both candidates, so it isn't a branch problem.

Model B (the PyTorch MDN, K=8) already ran this exact comparison: its angle MAEs ($\theta_1$ 0.46–0.58°, $\theta_2$ 11.3–12.3°, $\theta_3$ 35.3–38.4°) and branch-confusion rate (43.7%) are statistically indistinguishable from Model A's — consistent with both being dominated by the same unlearnable-label effect, not model class. Model B's pose RMSE (~11.8mm) is meaningfully better than Model A's, but that's confounded with Model B's FK-consistency loss term, which Model A doesn't have — the more likely explanation than the mixture itself.

**Follow-up — now answered.** Ran the analytic ceiling diagnostic (`thesis_doc.md` §4.6): 84.1% of test targets have both elbow branches geometrically valid (in-limits), giving a predicted ceiling of 57.9% assuming a coin flip on those. v8's observed 58.8% accuracy matches that almost exactly (0.8pp above, likely because reprojection distance carries a small amount of real signal even on "ambiguous" targets). **58.8% is essentially the ceiling for this dataset, not a modelling gap** — no further training, capacity, or architecture work on the current selection approach can move it materially.

**Q9.** Does taking the true elbow branch (oracle) at inference also fix pose error, the way it fixes angle MAE?

**A:** No — surprisingly, oracle conditioning makes pose error *worse* (22.61mm → 30.16mm), the opposite of the angle-MAE direction. `predict_ik`'s best-of-2 selection criterion *is* "pick whichever candidate reprojects closer to the target," so it already optimizes pose error directly, acting as an implicit two-candidate ensemble — regardless of which branch matches the sampled label. Oracle conditioning discards that ensembling benefit whenever the network's prediction on the "correct" branch happens to be the noisier of its two candidates for that sample. This was not predicted going in; see `thesis_doc.md` §4.5 for the full write-up and numbers.

