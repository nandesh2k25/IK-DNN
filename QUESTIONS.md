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
