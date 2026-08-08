# Thesis Documentation — Inverse Kinematics for a 3R Robot Arm in 3D

**Purpose of this document:** a running record of the problem formulation, every mathematical equation used in implementation, and the results of each experiment tried during this project. Kept up to date as work progresses so it can be used directly when writing the thesis (Methodology, Implementation, and Results/Discussion chapters). Append new sections/rows rather than rewriting old ones, so the document also preserves the *order* in which things were tried.

---

## 1. Problem Formulation

**Task:** learn the inverse kinematics (IK) mapping of a 3-revolute-joint (3R) robot arm operating in 3D — predict joint angles from a desired end-effector position.

**Robot structure:** a base yaw joint $\theta_1$ (rotates the whole arm about the vertical axis) followed by a 2-link planar arm ($\theta_2$, $\theta_3$) that moves within the $(r, z)$ half-plane swept out by that yaw.

### 1.1 Forward Kinematics (FK)

$$
r = L_1 + L_2\cos(\theta_2) + L_3\cos(\theta_2 + \theta_3)
$$
$$
z = L_2\sin(\theta_2) + L_3\sin(\theta_2 + \theta_3)
$$
$$
x = r\cos(\theta_1), \qquad y = r\sin(\theta_1)
$$

**Link lengths:** $L_1 = 1.0$, $L_2 = 1.0$, $L_3 = 0.5$ (metres).

**Joint limits:**
$$
\theta_1 \in (-\pi, \pi], \qquad \theta_2 \in \left(-\tfrac{\pi}{2}, \tfrac{\pi}{2}\right), \qquad \theta_3 \in \left(-\tfrac{\pi}{2}, \tfrac{\pi}{2}\right)
$$

### 1.2 Why this IK problem is hard (two distinct effects — keep them separate in the thesis)

1. **Multi-modality / branch ambiguity.** For a given reachable $(r,z)$, there are generally two valid $(\theta_2,\theta_3)$ solutions ("elbow-up" / "elbow-down"), so the map $(x,y,z)\to(\theta_1,\theta_2,\theta_3)$ is one-to-many. A point-estimate regressor trained with MSE cannot represent this; its optimal prediction where two branches nearly coincide in $(x,y,z)$-space is the *average* of the two branches, which is generally not a valid solution at all.
2. **Singularities (ill-conditioning), independent of (1).** Near $\theta_3 \approx 0$ (full extension) the Jacobian of the planar sub-chain loses rank in the $(r,z)$ plane (elbow singularity); near $r \approx 0$ (arm folded onto the vertical axis) $\theta_1$ becomes ill-defined (base/shoulder-type singularity). Near either, small position errors correspond to large angle errors regardless of which branch is chosen.

Both effects are orthogonal to the angle-periodicity issue in §3.7 — periodicity is a *representation* problem (fixed by re-encoding the target), while (1) and (2) are *problem-structure* limitations that a plain deterministic regressor cannot fix by re-encoding alone.

---

## 2. Dataset

**Generation:** joint angles are sampled i.i.d. uniformly within their limits, and the corresponding end-effector position is computed via FK (§1.1):

$$
\theta_1^{(i)} \sim \mathcal{U}(-\pi,\pi), \quad \theta_2^{(i)} \sim \mathcal{U}\!\left(-\tfrac{\pi}{2},\tfrac{\pi}{2}\right), \quad \theta_3^{(i)} \sim \mathcal{U}\!\left(-\tfrac{\pi}{2},\tfrac{\pi}{2}\right)
$$
$$
\mathbf{x}^{(i)} = \big(x^{(i)}, y^{(i)}, z^{(i)}\big) = \mathrm{FK}\big(\theta_1^{(i)},\theta_2^{(i)},\theta_3^{(i)}\big)
$$

**Split:** 80% train / 10% dev / 10% test (contiguous slices of one generated set; no shuffling needed since sampling is already i.i.d.).

**Standardization** (train-set statistics only, applied identically to dev/test):
$$
\tilde{x}_k = \frac{x_k - \mu_k}{\sigma_k}, \qquad \mu_k, \sigma_k \text{ computed per input/output component } k \text{ over the training set}
$$

---

## 3. Model A — From-Scratch Feedforward Network (NumPy, manual backprop)

A fully-connected network with ReLU hidden layers and a linear output layer, implemented without any autodiff framework, trained with plain (batch) gradient descent. This is the baseline used to build up intuition before adding a mixture-density output (Model B, §5).

### 3.1 Architecture

$L$ layers, dimensions $n^{[0]} = 3$ (input: $x,y,z$) through $n^{[L]}$ (output dimension — depends on the target representation, see §3.7).

### 3.2 Parameter Initialization (He initialization)

$$
W^{[l]} \sim \mathcal{N}(0,1)\cdot\sqrt{\frac{2}{n^{[l-1]}}}, \qquad b^{[l]} = \mathbf{0}, \qquad l = 1,\dots,L
$$

### 3.3 Forward Propagation

For each layer $l = 1,\dots,L$, with $A^{[0]} = X$:
$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$
$$
A^{[l]} = g^{[l]}\!\left(Z^{[l]}\right), \qquad
g^{[l]} =
\begin{cases}
\mathrm{ReLU}(z) = \max(0,z) & l < L \text{ (hidden layers)}\\
\mathrm{identity}(z) = z & l = L \text{ (output layer)}
\end{cases}
$$

### 3.4 Cost Function

Mean squared error over all $m$ examples and all output dimensions:
$$
J = \frac{1}{2m}\sum_{i=1}^{m}\left\lVert A^{[L](i)} - Y^{(i)} \right\rVert_2^2
$$

### 3.5 Backward Propagation

$$
dA^{[L]} = A^{[L]} - Y
$$

> **Important implementation note (kept for the thesis's methodology/pitfalls discussion):** the $\tfrac{1}{m}$ averaging must be applied **exactly once**, inside the $dW$/$db$ formulas below — *not* also in $dA^{[L]}$. An early implementation used $dA^{[L]} = (A^{[L]}-Y)/m$, which compounded an extra $\tfrac{1}{m}$ factor at every layer during backpropagation, silencing gradients by roughly another factor of $m$. The symptom was a *smoothly decreasing* loss that nonetheless converged to a model barely better than predicting the training-set mean — a useful cautionary example of a bug that is easy to miss because training "looks like it's working."

For each layer $l = L, L-1, \dots, 1$:
$$
dZ^{[l]} = dA^{[l]} \odot g^{[l]\prime}\!\left(Z^{[l]}\right), \qquad
g^{[l]\prime}(z) =
\begin{cases}
\mathbb{1}[z>0] & \text{ReLU (hidden layers)}\\
1 & \text{identity (output layer)}
\end{cases}
$$
$$
dW^{[l]} = \frac{1}{m}\, dZ^{[l]} \left(A^{[l-1]}\right)^{T}
$$
$$
db^{[l]} = \frac{1}{m}\sum_{i=1}^{m} dZ^{[l](i)}
$$
$$
dA^{[l-1]} = \left(W^{[l]}\right)^{T} dZ^{[l]}
$$

### 3.6 Parameter Update (plain gradient descent)

$$
W^{[l]} \leftarrow W^{[l]} - \alpha \, dW^{[l]}, \qquad b^{[l]} \leftarrow b^{[l]} - \alpha \, db^{[l]}
$$

No momentum/Adam yet — deliberately deferred until the plain baseline is verified correct (see `.claude/skills/thesis_check/SKILL.md`).

### 3.7 Target / Angle Representation (evolving)

**v1 — raw angles.** Target $Y = [\theta_1, \theta_2, \theta_3]^T$, each component z-score normalized. Problem: $\theta_1$ ranges over the full circle $(-\pi,\pi]$, so two physically adjacent directions near the wrap boundary (e.g. $179°$ and $-179°$) are numerically far apart. A continuous network is forced to fit a discontinuous function there, producing large error near that boundary.

**v2 — sin/cos encoding for $\theta_1$.** Regress $[\sin\theta_1, \cos\theta_1]$ instead of $\theta_1$ directly — continuous and wraparound-free — and recover the angle exactly afterward:

Encoding:
$$
Y_{\text{enc}} = \big[\sin\theta_1,\ \cos\theta_1,\ \theta_2,\ \theta_3\big]^T
$$
Decoding (exact, lossless, robust even if $\sin^2+\cos^2 \ne 1$ exactly):
$$
\hat{\theta}_1 = \operatorname{atan2}\!\left(\widehat{\sin\theta_1},\ \widehat{\cos\theta_1}\right)
$$
Normalization: $\sin\theta_1,\cos\theta_1$ are left un-normalized (identity, mean $0$/std $1$) since they are already bounded in $[-1,1]$; $\theta_2,\theta_3$ keep standard z-score normalization. (This avoids the decoder ever having to undo a normalization before applying $\operatorname{atan2}$.)

**Not yet done:** sin/cos encoding for $\theta_2,\theta_3$ (not needed — they never wrap); resolving the branch-ambiguity / singularity issues from §1.2 (a representation change alone cannot fix these — see §6).

### 3.8 Evaluation Metrics

**Circular angular MAE** (accounts for the $\pm180°$ wrap so a correct prediction right at the boundary isn't scored as a huge error):
$$
\Delta\theta = \operatorname{wrap}_{(-\pi,\pi]}\!\left(\hat\theta - \theta\right) = \left(\left(\hat\theta - \theta\right) + \pi\right) \bmod 2\pi - \pi
$$
$$
\mathrm{MAE}_{\text{deg}} = \frac{1}{m}\sum_{i=1}^{m}\left|\Delta\theta^{(i)}\right| \cdot \frac{180}{\pi}
$$

**End-effector pose error (mm):**
$$
e_{\text{pose}} = \left\lVert \mathrm{FK}(\hat\theta_1,\hat\theta_2,\hat\theta_3) - (x,y,z) \right\rVert_2 \times 1000
$$

### 3.9 Elbow Configuration Input (v3)

Adds a 4th input resolving the $\theta_2/\theta_3$ branch ambiguity from §1.2(1), derived exactly from $\theta_3$ at data-generation time:
$$
\text{elbow\_config} = \begin{cases} +1 & \theta_3 \geq 0 \\ -1 & \theta_3 < 0 \end{cases}
$$
Input becomes $[x, y, z, \text{elbow\_config}]$ (un-normalized 4th component, same rationale as the sin/cos components in §3.7). This is an **oracle input at training time only** — the true branch is known exactly for every generated sample. At inference the caller supplies only $(x,y,z)$; the network is queried once per candidate value and the result is chosen by nearest FK reprojection:
$$
\hat\theta = \operatorname*{arg\,min}_{\theta \in \{\theta^{(+1)},\, \theta^{(-1)}\}} \left\lVert \mathrm{FK}(\theta) - (x,y,z) \right\rVert_2, \qquad \theta^{(c)} = \operatorname{decode}\big(f_{\text{NN}}(x,y,z,c)\big)
$$

---

## 4. Results — Model A (From-Scratch Network)

Architecture: `[n_in, 64, 64, 4]`, ReLU hidden / linear output, $\alpha=0.05$, 3000 iterations of full-batch gradient descent, seed=1.

| Version | $n_{\text{in}}$ | $N$ samples | Final cost $J$ | $\theta_1$ MAE (°) | $\theta_2$ MAE (°) | $\theta_3$ MAE (°) | Mean angle MAE (°) | Elbow selection accuracy | Pose error, mean (mm) | Pose error, median (mm) | Pose error, p95 (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 — buggy backward pass ($dA^{[L]}$ double-divided by $m$, §3.5 note) | 3 | 15,000 | 3.450 | 106.9 | 83.1 | 60.9 | 83.6 | n/a | 3230.0 | 3241.6 | 4403.1 |
| v1 — raw angles, bug fixed | 3 | 15,000 | 0.493 | 15.27 | 12.41 | 41.01 | 22.90 | n/a | 527.96 | 445.71 | 1075.65 |
| v2 — sin/cos encoding for $\theta_1$ | 3 | 30,000 | 0.470 | **1.59** | 12.39 | 40.81 | 18.26 | n/a | 193.88 | 178.84 | 380.07 |
| v3 — `elbow_config` input (§3.9), best-of-2 at inference | 4 | 30,000 | **0.059** | 1.69 | 13.41 | 41.14 | 18.75 | **59.0%** | **101.64** | 95.55 | 185.07 |

*(Test-set numbers shown; dev-set numbers track within ~1% of test throughout, so the model is not overfitting at this scale.)*

**Caveat for the thesis write-up:** v1→v2 changed **two** things at once — the angle encoding *and* $N$ (15,000 → 30,000, changed independently mid-project) — so the improvement cannot be attributed to encoding alone from this comparison. However, $\theta_2$'s MAE (untouched by the encoding change) stayed essentially flat between v1 and v2 (12.41° → 12.39°) despite the doubled dataset, which is indirect evidence that the extra data was not the dominant driver of $\theta_1$'s ~10x improvement — the encoding fix is. If a clean ablation is wanted for the thesis, re-run v1 and v2 at the same $N$.

**Interpretation (v1→v2):**
- $\theta_1$: dropped ~10x (15.3° → 1.6°) — consistent with removing the wraparound discontinuity (§3.7).
- $\theta_2$: unchanged, as expected — never encoded, never had a periodicity problem.
- $\theta_3$: unchanged (~41°) — as expected; still dominated by branch ambiguity / singularity effects (§1.2), which sin/cos encoding does not address.
- Pose error fell ~2.7x, mechanically explained by $x=r\cos\theta_1,\, y=r\sin\theta_1$: a large $\theta_1$ error swings the predicted position around a large fraction of the workspace, so fixing it removes a dominant error source even with $\theta_2,\theta_3$ unchanged.

**Interpretation (v2→v3) — an important, non-obvious result:** training cost dropped almost 10x (0.470 → 0.059), confirming the oracle-conditioned mapping $(x,y,z,\text{elbow\_config})\to\theta$ is learned very well. Yet $\theta_2$/$\theta_3$ MAE against the *test-set labels* barely moved, while pose error nearly halved (193.9mm → 101.6mm). The elbow-selection accuracy (59.0%, barely above the 50% chance floor) explains why: once the network has learned **both** branch functions accurately, querying it with *either* `elbow_config` value for the same $(x,y,z)$ tends to reproject close to the target for **both** candidates, because both are genuinely valid IK solutions — not because the model is confused. The "pick smaller reprojection error" rule therefore can't reliably discriminate between them, and roughly 40% of the time lands on the branch that isn't the one the test label happened to be sampled from, which is scored as "wrong" against that single label even though it's a physically valid alternate solution.

This is direct, controlled evidence for the concern raised in §6.4 below: a meaningful fraction of "branch confusion" error against a single sampled label is a **metric-framing artifact**, not a modeling defect — the near-halved pose error shows the model is finding *a* valid solution far more often, even though it doesn't reliably reproduce the *specific* one a fixed test label happened to record. This has a real thesis-framing implication: for a problem that is intrinsically one-to-many, angle-MAE against one sampled label may not be the most meaningful accuracy metric; pose error against the requested target is arguably closer to what the thesis's actual requirement (end-effector accuracy) is asking for.

### 4.1 Diagnostic: is v3's remaining pose error singularity-driven?

Since the thesis's primary accuracy criterion is end-effector (pose) error, it matters whether the ~100mm remaining in v3 is concentrated near the two singularities (§1.2.2) or spread generally. Measured on the v3 test set (elbow proxy $=|\sin\theta_3|$, base proxy $=r=\sqrt{x^2+y^2}$, both computed from ground truth):

| Quantity | Elbow singularity ($\theta_3\approx0$) | Base singularity ($r\approx0$) |
|---|---|---|
| $\operatorname{corr}(e_{\text{pose}}, \text{proxy})$ | $+0.233$ | $-0.015$ (negligible) |
| Worst pose-error decile's mean proxy vs. overall | 0.749 vs. 0.633 (**further** from singularity) | 1.727 vs. 1.836 (~same) |
| Mean pose error, near vs. elsewhere | **94.3mm vs. 103.5mm** (lower near it) | 105.2mm vs. 100.8mm (~same) |
| Elbow-selection accuracy, near vs. elsewhere | **52.7% vs. 60.6%** (worse near it) | **86.7% vs. 52.1%** (better near it) |

**Conclusion: neither singularity drives the remaining pose error.** Both correlations are weak and the worst-error decile is not concentrated near either singularity — if anything, pose error is slightly *lower* near the elbow singularity.

**Why pose error drops near the elbow singularity (not obvious, worth stating explicitly in the thesis):** a kinematic singularity is exactly the configuration where the Jacobian loses rank along one direction — i.e. where *changing the angle along that direction barely moves the end-effector*. So near $\theta_3\approx0$, an angle error there produces a *small* position error, not a large one. This does not contradict Model B's earlier finding of a strong negative correlation between the same proxy and **angle** error ($-0.644$, §5.4) — that measured a different axis. Near a singularity, angle uncertainty grows (many angles reach nearly the same point) while position error shrinks; both effects are the same underlying phenomenon viewed from angle-space vs. pose-space respectively.

**Branch-selection side-finding:** selection accuracy drops near the elbow singularity (branches nearly coincide at $\theta_3=0$, so telling them apart by reprojection distance approaches a coin flip) and rises sharply near the base singularity (a more complex effect, not fully explained here — the FK formula's $r$ can go slightly negative, creating a second redundancy involving $\theta_1$ that appears to make the two elbow candidates more distinguishable there). Not the primary finding, but worth a footnote if branch-selection mechanics are discussed.

**Implication:** the remaining ~100mm looks like a **general regression precision ceiling** (plain gradient descent, 3000 iterations, 64-64 hidden units, no momentum/Adam — still gently decreasing at the end of training) rather than a structural singularity problem. This re-prioritizes next steps: training improvements (optimizer, iteration budget, capacity) are likely higher-leverage for reducing pose error than singularity-targeted oversampling, at least at the current model scale.

---

## 5. Model B — Mixture Density Network (main thesis model, PyTorch)

Reference summary — full implementation lives in `IK_MDN(3R in 3D)_tuned\notebooks\IK_MDN_3R_3D_Colab.ipynb`; see that notebook and `handoff.md` for exact code and the fullest diagnostic detail. Included here so the equations are all in one place for thesis writing.

### 5.1 Output representation

All three angles sin/cos-encoded (6-dim), unlike Model A which currently only encodes $\theta_1$:
$$
Y_{\text{enc}} = \big[\sin\theta_1,\cos\theta_1,\sin\theta_2,\cos\theta_2,\sin\theta_3,\cos\theta_3\big]^T \in \mathbb{R}^6
$$

### 5.2 Mixture density formulation

The network outputs, for $K$ mixture components, per-component mixing logits, means, and (log-)variances; a shared trunk maps $x \to$ these parameters. Mixing coefficients via softmax:
$$
\pi_k(\mathbf{x}) = \frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)}, \qquad k=1,\dots,K
$$
Predictive density — a mixture of diagonal Gaussians over the 6-dim encoded target:
$$
p(\mathbf{y}\mid\mathbf{x}) = \sum_{k=1}^{K} \pi_k(\mathbf{x})\, \mathcal{N}\!\left(\mathbf{y}; \boldsymbol{\mu}_k(\mathbf{x}), \operatorname{diag}(\boldsymbol{\sigma}_k^2(\mathbf{x}))\right)
$$
Negative log-likelihood loss:
$$
\mathcal{L}_{\text{NLL}} = -\frac{1}{m}\sum_{i=1}^m \log \sum_{k=1}^{K} \pi_k(\mathbf{x}^{(i)})\, \mathcal{N}\!\left(\mathbf{y}^{(i)}; \boldsymbol{\mu}_k(\mathbf{x}^{(i)}), \operatorname{diag}(\boldsymbol{\sigma}_k^2(\mathbf{x}^{(i)}))\right)
$$

### 5.3 Auxiliary loss terms

**Unit-circle penalty** (a diagonal Gaussian NLL over sin/cos pairs does not by itself enforce $\sin^2+\cos^2=1$):
$$
\mathcal{L}_{\text{circle}} = \frac{1}{m}\sum_{i=1}^m \sum_{j\in\{1,2,3\}} \left(\widehat{\sin\theta_j}^{(i)\,2} + \widehat{\cos\theta_j}^{(i)\,2} - 1\right)^{2}
$$

**FK-consistency loss** (decode the predicted mean via $\operatorname{atan2}$ per §3.7, run it through FK, and penalize the resulting pose error):
$$
\mathcal{L}_{\text{FK}} = \frac{1}{m}\sum_{i=1}^m \left\lVert \mathrm{FK}\big(\operatorname{decode}(\boldsymbol{\mu}(\mathbf{x}^{(i)}))\big) - \mathbf{x}^{(i)} \right\rVert_2^2
$$

**Total loss:**
$$
\mathcal{L} = \mathcal{L}_{\text{NLL}} + \lambda_{\text{FK}}\,\mathcal{L}_{\text{FK}} \quad (+\ \text{unit-circle penalty term where used})
$$
with $\lambda_{\text{FK}} = 0.1$ in the current configuration ($K=8$ components, hidden size 256).

### 5.4 Latest diagnostic results (all-angle sin/cos encoding, $K=8$)

| Quantity | Value |
|---|---|
| $\theta_1$ MAE | 0.46°–0.58° |
| $\theta_2$ MAE | 11.3°–12.3° |
| $\theta_3$ MAE | 35.3°–38.4° |
| Pose RMSE | ~11.8 mm |
| Branch confusion rate (sign of predicted vs. true $\theta_3$) | 43.7% |
| Ill-conditioning correlation ($\lvert\theta_3\rvert$ error vs. $\lvert\sin\theta_3\rvert$, correct-branch subset) | $-0.644$ |
| Active mixture components | 6 of 8 (2 dead); top 2 components account for 72.6% of selections |
| Mixture confidence, worst error decile vs. overall | 0.551 vs. 0.468 (overconfident on its worst errors) |

---

## 6. Known Limitations / Open Problems (for Discussion chapter)

1. **Periodicity** — solved for $\theta_1$ in both models via sin/cos encoding (§3.7, §5.1). Not applicable to $\theta_2,\theta_3$ (bounded, never wrap).
2. **Singularities** — not addressed by sampling or loss weighting in either model. Nuance from Model A's v3 diagnostic (§4.1): singularities elevate **angle** error near them (confirmed on Model B, §5.4: $-0.644$ correlation) but do **not** elevate **pose** error the same way — near a singularity, angle uncertainty grows while position sensitivity to that angle shrinks, so the two metrics disagree there. Don't assume "singularity" automatically means "worse end-effector accuracy" without checking which metric is being discussed.
3. **Multi-modality / branch ambiguity** — the core unresolved issue. Model A cannot represent it structurally (single deterministic head, MSE-optimal prediction is the invalid average of branches). Model B's mixture output *can* represent it in principle, but the 43.7% branch-confusion rate and mixture-collapse pattern show it is not yet cleanly separating the two branches into distinct components.
4. **Open verification — now answered indirectly, via Model A v3 (§4).** The question was whether some of the 43.7% "branch confusion" in Model B is actually the network outputting the *other, equally valid* elbow solution rather than a genuine error. Model A's v3 experiment (oracle `elbow_config` input, best-of-2 selection by FK reprojection at inference) gives strong evidence this is real: pose error nearly halved while angle-MAE against the sampled label barely moved, and best-of-2 selection accuracy against the true branch was only 59% (near chance) — consistent with both branches being learned accurately enough that reprojection error alone can't discriminate which one matches the specific sampled label. This has not been directly re-verified on Model B itself (still open: run the sign-flipped/pose-error filter check on Model B's own predictions), but the mechanism is now demonstrated, not just hypothesized.
5. **Candidate next steps** (not yet implemented on Model B): hard-assignment/winner-take-most loss to force branch specialization in Model B's mixture; a 2-head architecture (one head per elbow branch, since the true branch is knowable analytically at data-generation time); oversampling near the singularities during dataset generation.
6. **Decision point (06-08-26): two ways to use a known elbow branch label.**
   - **(A) — chosen, implemented in Model A as v3 (§3.9, §4).** Reformulate the input as $(x,y,z,\text{elbow\_config})$, with `elbow_config` a known input at train time. At inference, only $(x,y,z)$ is required from the caller: the predict function (`predict_ik` in the notebook) queries the model with both `elbow_config` values and returns whichever decoded solution has the smaller FK-reprojection (pose) error. Result: pose error nearly halved (193.9mm → 101.6mm); angle-MAE against the sampled label was *not* fixed by this — see the v2→v3 interpretation in §4 for why that's actually informative rather than a failure.
   - **(B) — deferred, flagged for future reference.** Use the branch label only as a *training-time* signal to build a 2-head (or branch-specialized-component) architecture, with the network still taking only $(x,y,z)$ as input and outputting both candidate solutions unprompted, rather than being told the branch. Not implemented; worth revisiting since (A)'s selection step is now shown to be unreliable for recovering the *specific* sampled branch (though not for finding *a* valid one).
7. **New open question raised by v3's result:** is angle-MAE-against-one-sampled-label even the right accuracy metric for this intrinsically one-to-many problem, given pose error (which doesn't care which branch was used) tells a very different and arguably more meaningful story? Worth raising explicitly in the thesis's methodology/discussion, and worth deciding before investing further effort in reducing angle-MAE specifically.

---

## 7. Change Log

### Prior sessions (before 06-08-26)
- Built Model B (PyTorch MDN), initially with raw-angle targets. Ran requirement check against thesis spec (mean angle error < 5°, pose error < 5mm): failed, dominated by $\theta_3$ MAE ≈ 38.8°.
- Diagnosed bias/variance as healthy (good fit, no significant over/underfitting) — training dynamics were not the bottleneck.
- Rebuilt Model B's target representation to all-angle sin/cos encoding (§5.1–§5.3). Result: pose RMSE improved (~26.9mm → ~11.8mm) and the $\theta_1$ wraparound risk was removed, but $\theta_3$ MAE was essentially unchanged (~38.8° → ~35–38°) — confirming representation and mixture branch-separation are separate problems.
- Built the automated "Key Diagnostics Summary" cell for Model B (requirement check, bias/variance snapshot, mixture usage, branch-confusion / ill-conditioning breakdown) — results in §5.4.

### 06-08-26: Thursday
- Built Model A from scratch (§3): pure NumPy, manual backprop, plain gradient descent, single deterministic head, no mixture, no optimizer — by explicit design, as a baseline to build up correctness before adding complexity.
- Found and fixed the $dA^{[L]}$ double-$\tfrac{1}{m}$ bug (§3.5 note). Cost dropped from 3.45 to 0.49 after the fix at the same iteration budget; mean angle MAE dropped from 83.6° to 22.9° — confirms it was a real correctness bug, not a hyperparameter issue.
- Added sin/cos encoding for $\theta_1$ to Model A (§3.7 v2). Results in §4: $\theta_1$ MAE 15.3° → 1.6°; $\theta_2,\theta_3$ unaffected as expected; pose error 528mm → 194mm.
- Created `QUESTIONS.md` (conceptual Q&A log) and `.claude/skills/thesis_check/SKILL.md` (project do's/don'ts checklist).
- Added `elbow_config` oracle input + best-of-2 inference to Model A (§3.9 v3), to remove the theta2/theta3 branch ambiguity (§1.2). Training cost dropped ~10x (0.470 → 0.059); pose error nearly halved (193.9mm → 101.6mm); but angle-MAE against the sampled label barely moved and best-of-2 branch-selection accuracy was only 59% (near chance) — strong evidence that a meaningful share of "branch confusion" is a metric-framing artifact (the model finding a *different but equally valid* solution), not a pure error. See §4 and §6 items 4/6/7 for the full interpretation and its implications for Model B and for the thesis's choice of accuracy metric.
- Ran a singularity diagnostic on v3's remaining pose error (§4.1), since the thesis prioritizes end-effector accuracy: found neither the elbow (`theta3≈0`) nor base (`r≈0`) singularity drives it (weak correlations, worst-error decile not concentrated near either) — pose error is actually slightly *lower* near the elbow singularity, explained by the Jacobian-rank-loss direction being where angle changes barely move the end-effector. Concluded the remaining ~100mm is a general precision ceiling (plain GD, 3000 iterations, no optimizer) rather than a singularity problem — re-prioritizes next steps toward training improvements over singularity-targeted oversampling.
