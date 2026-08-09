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

This remains the **default** update rule and is used unchanged by every version through v5. Adam (§3.11, added in v6) is selected explicitly via an `optimizer` argument rather than replacing this, so all earlier results stay exactly reproducible — confirmed empirically, see the `v5-repro` row in §4.2.

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

### 3.10 Mini-Batch Gradient Descent (v5)

Full-batch gradient descent (§3.6) takes one parameter update per pass over the *entire* training set — every step is a large matmul over all $m$ examples. Mini-batch gradient descent instead splits each pass ("epoch") into batches of size $B$ and takes one update per batch, so parameters update $\lceil m/B \rceil$ times per epoch instead of once.

**Batch construction** (`random_mini_batches`, mirrors the Andrew Ng Course 2 convention): shuffle the $m$ training columns with a random permutation $\pi$, then partition into batches of size $B$ (the last batch may be smaller than $B$ if $B \nmid m$):
$$
n_b = \left\lceil \frac{m}{B} \right\rceil, \qquad \text{batch}_b = \{X_{:,\pi(i)}, Y_{:,\pi(i)}\}_{i \in [(b-1)B,\, bB)}
$$

**Per-epoch cost** (size-weighted average across batches, so it stays comparable in scale to the full-batch cost of §3.4):
$$
J_{\text{epoch}} = \sum_{b=1}^{n_b} \frac{m_b}{m}\, J_b, \qquad J_b = \frac{1}{2 m_b}\sum_{i \in \text{batch}_b} \left\lVert A^{[L](i)} - Y^{(i)} \right\rVert_2^2
$$

Forward propagation (§3.3), backward propagation (§3.5), and the update rule (§3.6) are all unchanged — each is simply applied once per batch instead of once per epoch. Data is reshuffled every epoch. This is still plain gradient descent (no momentum/Adam) — only the batching changed.

### 3.11 Adam Optimizer (v6)

Plain gradient descent (§3.6) applies the same step size to every parameter, in exactly the direction of the current gradient, with no memory of previous steps. Adam replaces that rule with two accumulators — it is the fusion of two separately-developed methods, so implementing Adam necessarily means implementing both.

**Momentum** — an exponentially-weighted moving average of the gradient. Mini-batch gradients are noisy estimates of the true gradient; averaging cancels the component that flips sign from batch to batch while reinforcing the consistent direction:
$$
v^{[l]} \leftarrow \beta_1 v^{[l]} + (1-\beta_1)\, dW^{[l]}
$$

**RMSprop** — an exponentially-weighted moving average of the *elementwise squared* gradient, used to normalize the step per parameter, so parameters with persistently small gradients still move meaningfully:
$$
s^{[l]} \leftarrow \beta_2 s^{[l]} + (1-\beta_2)\, \left(dW^{[l]}\right)^{\odot 2}
$$

**Bias correction** — both accumulators are initialized to zero, so early estimates are biased toward zero. Dividing by $1-\beta^t$ removes this bias exactly:
$$
\hat{v}^{[l]} = \frac{v^{[l]}}{1-\beta_1^{\,t}}, \qquad \hat{s}^{[l]} = \frac{s^{[l]}}{1-\beta_2^{\,t}}
$$

**Update:**
$$
W^{[l]} \leftarrow W^{[l]} - \alpha \frac{\hat{v}^{[l]}}{\sqrt{\hat{s}^{[l]}} + \epsilon}
$$

Identical formulas apply to $b^{[l]}$. Hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Two implementation notes worth carrying into the thesis's methodology/pitfalls discussion:**

1. **$t$ is the global parameter-update counter, not the epoch index.** Bias correction depends on $\beta^t$, so under mini-batching (§3.10) $t$ must advance once per *batch*. With $B=256$ over $m=24{,}000$ that is $\lceil m/B \rceil = 94$ increments per epoch — passing the epoch index instead would leave the correction wrong by that factor for the entire run. Like the $dA^{[L]}$ bug in §3.5, this is a defect that still produces a smoothly decreasing loss, so it was asserted explicitly rather than inferred from the loss curve (the implementation prints its final update count; a 100-epoch run must report 9,400).

2. **The two biases do not cancel in the ratio $\hat{v}/\sqrt{\hat{s}}$.** Because $s$ is far more biased than $v$ at small $t$, omitting bias correction entirely makes the *first* step too **large** by a factor of
$$
\frac{1-\beta_1}{\sqrt{1-\beta_2}} = \frac{0.1}{\sqrt{0.001}} \approx 3.16
$$
— the opposite direction from what "both accumulators start too small" intuitively suggests. This is verified numerically in the notebook's sanity check.

**Consequence for the learning rate (important for reading the results table):** because the step is divided by $\sqrt{\hat{s}}$, its magnitude is approximately $\pm\alpha$ per parameter *regardless of gradient scale*. $\alpha$ therefore does **not** carry the same meaning across optimizers, and v5's $\alpha=0.05$ is far too large for Adam. This is why v6 is reported alongside a learning-rate sweep (§4.2) rather than at a single inherited $\alpha$.

### 3.12 Learning-Rate Decay (v7)

Adam (§3.11) adapts the step **per parameter**, rescaling each weight's update by its own recent gradient history — but the base learning rate $\alpha$ passed in still sets the overall scale of every step for the entire run. Learning-rate decay addresses the orthogonal axis: shrinking that **global** scale over time. The two are routinely combined (this is what PyTorch's `Adam` + an `LRScheduler` does).

Motivation, specific to mini-batch training (§3.10): each mini-batch gradient is a *noisy* estimate of the true gradient. Early in training that noise is irrelevant next to how far the parameters still have to travel, so large steps are fine. Late in training, near a minimum, a fixed large $\alpha$ makes the parameters bounce around inside a region set by that noise rather than settling — decaying $\alpha$ shrinks that final wandering radius.

**Smooth inverse-time decay:**
$$\alpha_t = \frac{\alpha_0}{1 + \text{decay\_rate} \cdot t}$$

**Staircase decay** (constant for `time_interval` epochs at a stretch, then drops):
$$\alpha_t = \frac{\alpha_0}{1 + \text{decay\_rate} \cdot \lfloor t / \text{time\_interval} \rfloor}$$

Setting `time_interval=1` makes staircase decay reduce exactly to inverse-time decay — verified in the notebook's sanity checks, along with epoch-0 being undecayed and the closed form at a specific epoch.

**A third gotcha in the same family as the $dA^{[L]}$ bug (§3.5) and Adam's $t$ bug (§3.11):** decay is indexed by the **epoch**; Adam's bias correction is indexed by $t$, the **global parameter-update counter**. Both counters are live in the same training loop and are trivially swappable. Passing $t$ into a decay schedule instead of the epoch index would decay $\alpha$ roughly 94x too fast at batch size 256 over 24,000 examples ($t$ reaches ~9,400 over a 100-epoch run, vs. 99 for the epoch index), collapsing the effective step to near zero almost immediately. Like both prior bugs, this would still produce a smooth, plausible-looking loss curve — it does not fail loudly. `nn_model`'s final print line reports the schedule's actual $\alpha_0 \to \alpha_{\text{final}}$, specifically so this is checkable at a glance. Verified directly in the v7 sweep (§4.3): every printed final $\alpha$ matched its closed-form value exactly (e.g. staircase, decay\_rate=0.1 → $\alpha_0/1.9$; inverse, decay\_rate=0.1 → $\alpha_0/10.9$).

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

Architecture (v0–v3): `[n_in, 64, 64, 4]`, ReLU hidden / linear output, $\alpha=0.05$, 3000 iterations of full-batch gradient descent, seed=1. **Architecture (v4–v5): `[4, 256, 128, 64, 4]`** — see the v3→v4 caveat below.

| Version | $n_{\text{in}}$ | $N$ samples | Final cost $J$ | $\theta_1$ MAE (°) | $\theta_2$ MAE (°) | $\theta_3$ MAE (°) | Mean angle MAE (°) | Elbow selection accuracy | Pose error, mean (mm) | Pose error, median (mm) | Pose error, p95 (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 — buggy backward pass ($dA^{[L]}$ double-divided by $m$, §3.5 note) | 3 | 15,000 | 3.450 | 106.9 | 83.1 | 60.9 | 83.6 | n/a | 3230.0 | 3241.6 | 4403.1 |
| v1 — raw angles, bug fixed | 3 | 15,000 | 0.493 | 15.27 | 12.41 | 41.01 | 22.90 | n/a | 527.96 | 445.71 | 1075.65 |
| v2 — sin/cos encoding for $\theta_1$ | 3 | 30,000 | 0.470 | 1.59 | 12.39 | 40.81 | 18.26 | n/a | 193.88 | 178.84 | 380.07 |
| v3 — `elbow_config` input (§3.9), best-of-2 at inference | 4 | 30,000 | 0.059 | 1.69 | 13.41 | 41.14 | 18.75 | 59.0% | 101.64 | 95.55 | 185.07 |
| v4 — increased capacity (`[4,256,128,64,4]`), still full-batch GD, 3000 iterations | 4 | 30,000 | 0.040 | 1.29 | 14.78 | 45.73 | 20.60 | 56.2% | 74.08 | 68.65 | 140.34 |
| v5 — mini-batch GD (§3.10), batch size 256, 100 epochs, same architecture as v4 | 4 | 30,000 | 0.012 | 0.97 | 12.33 | 38.24 | 17.18 | 57.0% | 48.89 | 46.29 | 88.95 |
| v6 — Adam (§3.11), $\alpha = 3\times10^{-4}$, all else identical to v5 | 4 | 30,000 | **0.0032** | **0.64** | **11.86** | **36.95** | **16.49** | 57.0% | **31.61** | **28.84** | **60.09** |
| v7 — LR decay (§3.12) evaluated via sweep, $\alpha_0=3\times10^{-4}$; no decay rate beat "no decay," so v7 = v6 | 4 | 30,000 | 0.0032 | 0.64 | 11.86 | 36.95 | 16.49 | 57.0% | 31.61 | 28.84 | 60.09 |
| **v8 — LR decay paired with a larger $\alpha_0$ (§4.4): $\alpha_0=2\times10^{-3}$, inverse decay $\to\approx3\times10^{-4}$ by epoch 99** | 4 | 30,000 | **0.0015** | 0.48 | **11.70** | 36.43 | **16.20** | 58.8% | **22.61** | **20.91** | **43.32** |

*(Test-set numbers shown; dev-set numbers track within ~1% of test throughout, so the model is not overfitting at this scale.)*

**Caveat for the thesis write-up:** v1→v2 changed **two** things at once — the angle encoding *and* $N$ (15,000 → 30,000, changed independently mid-project) — so the improvement cannot be attributed to encoding alone from this comparison. However, $\theta_2$'s MAE (untouched by the encoding change) stayed essentially flat between v1 and v2 (12.41° → 12.39°) despite the doubled dataset, which is indirect evidence that the extra data was not the dominant driver of $\theta_1$'s ~10x improvement — the encoding fix is. If a clean ablation is wanted for the thesis, re-run v1 and v2 at the same $N$.

**Interpretation (v1→v2):**
- $\theta_1$: dropped ~10x (15.3° → 1.6°) — consistent with removing the wraparound discontinuity (§3.7).
- $\theta_2$: unchanged, as expected — never encoded, never had a periodicity problem.
- $\theta_3$: unchanged (~41°) — as expected; still dominated by branch ambiguity / singularity effects (§1.2), which sin/cos encoding does not address.
- Pose error fell ~2.7x, mechanically explained by $x=r\cos\theta_1,\, y=r\sin\theta_1$: a large $\theta_1$ error swings the predicted position around a large fraction of the workspace, so fixing it removes a dominant error source even with $\theta_2,\theta_3$ unchanged.

**Interpretation (v2→v3) — an important, non-obvious result:** training cost dropped almost 10x (0.470 → 0.059), confirming the oracle-conditioned mapping $(x,y,z,\text{elbow\_config})\to\theta$ is learned very well. Yet $\theta_2$/$\theta_3$ MAE against the *test-set labels* barely moved, while pose error nearly halved (193.9mm → 101.6mm). The elbow-selection accuracy (59.0%, barely above the 50% chance floor) explains why: once the network has learned **both** branch functions accurately, querying it with *either* `elbow_config` value for the same $(x,y,z)$ tends to reproject close to the target for **both** candidates, because both are genuinely valid IK solutions — not because the model is confused. The "pick smaller reprojection error" rule therefore can't reliably discriminate between them, and roughly 40% of the time lands on the branch that isn't the one the test label happened to be sampled from, which is scored as "wrong" against that single label even though it's a physically valid alternate solution.

This is direct, controlled evidence for the concern raised in §6.4 below: a meaningful fraction of "branch confusion" error against a single sampled label is a **metric-framing artifact**, not a modeling defect — the near-halved pose error shows the model is finding *a* valid solution far more often, even though it doesn't reliably reproduce the *specific* one a fixed test label happened to record. This has a real thesis-framing implication: for a problem that is intrinsically one-to-many, angle-MAE against one sampled label may not be the most meaningful accuracy metric; pose error against the requested target is arguably closer to what the thesis's actual requirement (end-effector accuracy) is asking for.

**Caveat (v3→v4):** the capacity increase from `[4,64,64,4]` to `[4,256,128,64,4]` was made outside a tracked planning discussion (found already applied when this project was revisited); it is logged here for completeness rather than as a deliberate ablation. Its effect is nonetheless clear and directionally sensible: cost fell further (0.059 → 0.040) and pose error dropped substantially (101.64mm → 74.08mm), while angle-MAE and elbow-selection accuracy stayed roughly flat — consistent with the extra capacity mainly buying a lower general precision ceiling (§4.1), not resolving the branch-selection near-chance behavior, which is a metric-framing effect (see above) rather than a capacity limitation.

**Interpretation (v4→v5) — mini-batch GD, isolated from the capacity change:** with architecture held fixed at `[4,256,128,64,4]`, switching only the training loop from full-batch (3000 iterations) to mini-batch (§3.10, batch size 256, 100 epochs) improved every accuracy metric simultaneously: cost 0.040 → 0.012, mean angle MAE 20.60° → 17.18°, pose error mean 74.08mm → 48.89mm — while also finishing in **65.6 seconds**, versus a full-batch run that (while not directly timed) took long enough in this session to repeatedly require polling for completion. This is a cleaner comparison than v1→v2 or v3→v4 above (only one thing changed), and confirms the mechanism predicted in §3.10: mini-batching doesn't reduce the arithmetic per epoch, but the ~94x more frequent parameter updates per epoch let it reach a lower cost in far fewer epochs (100) than full-batch needed iterations (3000), which is why it is both faster *and* more accurate here rather than trading one for the other. Elbow-selection accuracy stayed flat (56.2% → 57.0%, still near the 50% chance floor) — expected, since that's the same metric-framing effect from v2→v3 above, not something training speed or convergence quality addresses.

**Interpretation (v5→v6) — Adam, isolated from every other change.** Architecture, batch size, epoch count, and seed were all held at v5's values; only the update rule and $\alpha$ changed. Test pose error fell **48.89mm → 31.61mm (−35%)** and final cost fell 0.012 → 0.0032, while mean angle MAE moved only slightly (17.18° → 16.49°) and elbow-selection accuracy was **unchanged to three significant figures (57.0% → 57.0%)**. That split is exactly the pattern §4 has established for this model: improvements to *convergence* buy pose accuracy, while $\theta_3$ MAE (~37°) and branch selection (~57%) are floored by the problem's one-to-many structure (§1.2.1) and are untouchable by any optimizer. Adam cost ~15% more wall-clock per epoch than plain GD (72.7s vs 63.5s for 100 epochs) — the price of maintaining two extra accumulators per parameter — which is negligible against a 35% pose-error reduction at an identical epoch budget.

Two further points from the sweep in §4.2 that matter for how this result should be stated in the thesis:

- **Adam beat plain GD at every learning rate tested**, including its own worst (41.95mm at $\alpha=10^{-2}$, still well below GD's 48.89mm). The improvement therefore cannot be attributed to having found a lucky $\alpha$ — it survives a full order-of-magnitude of variation in the one hyperparameter that changed alongside the optimizer. This is the cleanest way to report v6 given that $\alpha$ *had* to change (§3.11).
- **Training cost and pose error ranked the runs differently**, discussed below.

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

**Re-run on v5 (mini-batch GD, §3.10):** the same diagnostic was re-run after v5's training-speed/quality improvements (§4). Conclusion unchanged — correlations stayed weak ($\text{corr}(e_{\text{pose}}, |\sin\theta_3|) = 0.018$, $\text{corr}(e_{\text{pose}}, r) = 0.026$), and pose error near vs. elsewhere for both singularities stayed within a few mm of each other. This confirms v4→v5's ~26mm pose-error improvement (§4) came from better general convergence, not from incidentally fixing a singularity-related weakness.

**Re-run on v6 (Adam, §3.11):** conclusion unchanged again — $\text{corr}(e_{\text{pose}}, |\sin\theta_3|) = 0.068$, $\text{corr}(e_{\text{pose}}, r) = 0.091$, both still weak; the worst-error decile's mean $|\sin\theta_3|$ was 0.630 vs. 0.633 overall (i.e. not concentrated near the elbow singularity at all); and mean pose error near vs. elsewhere was 31.80mm vs. 31.57mm (elbow) and 29.60mm vs. 32.12mm (base). The branch-selection side-finding also persisted: 48.8% near the elbow singularity vs. 59.0% elsewhere, and 86.0% near the base singularity vs. 49.7% elsewhere.

**Standing conclusion across v3, v5, v6, and v8:** the §4.1 implication has now been confirmed four times and has been *acted on* successfully. It predicted that training improvements would be higher-leverage for pose error than singularity-targeted oversampling; three successive training changes (mini-batching, then Adam, then decay paired with a larger $\alpha_0$ — §4.4) have since taken pose error from 101.6mm → 48.9mm → 31.6mm → 22.6mm without any singularity-specific intervention, while the singularity correlations stayed weak throughout (v8's re-check: 0.054 elbow, 0.059 base). The prediction and the outcome are worth presenting together in the thesis — it is one of the few places in this project where a diagnostic drove a decision and the decision paid off as predicted, repeatedly.

**Interpretation (v6→v7) — learning-rate decay, an honest negative result.** The prediction going into this experiment (recorded before running it, not after) was that the gain might be small or absent: $\alpha=3\times10^{-4}$ was already the *smallest* $\alpha$ tested in the §4.2 sweep and had already won, which is not obviously the "bouncing at too large a step near the minimum" regime decay is meant to fix. The §4.3 sweep confirms this more strongly than expected: **every decay configuration tested performed worse than no decay**, monotonically so as decay\_rate increased — from 33.4mm (mildest rate) up to 59.9mm (most aggressive), against 31.61mm for no decay. See §4.3 for the full table and mechanism. v7 is therefore reported as v6's numbers with `decay=None` confirmed as the genuine winner of a real search, not a placeholder — the result belongs in the thesis as a demonstration that decay is not automatically beneficial, and that its benefit depends on where in the loss landscape $\alpha_0$ already sits at the epoch budget being used.

**Interpretation (v7→v8) — decay redeemed at a larger $\alpha_0$.** v7's negative result was not the end of the decay story: its own mechanical explanation (§4.3) predicted decay would help if paired with a rate that genuinely needed shrinking over the run, rather than one already at its optimum. Testing that directly (§4.4) confirmed it — $\alpha_0=2\times10^{-3}$, decaying to $\approx3\times10^{-4}$ by epoch 99, drove test pose error to **22.61mm**, a further **−28% from v6/v7's 31.61mm** and **−79mm/−78% from v3's 101.6mm oracle baseline**. Mean angle MAE and elbow-selection accuracy moved only slightly (16.49°→16.20°, 57.0%→58.8%), the same convergence-improves-pose-not-branch-selection split seen at every step since v5. This makes v8 the best Model A configuration found in this project, and its story — a negative result at one setting motivating and correctly predicting a positive result at another — is a clean illustration of how a properly isolated single-variable experiment stays useful even when it "fails."

### 4.2 Learning-rate sweep for v6 (and a caution about using cost as the selection criterion)

Since $\alpha$ is not comparable across optimizers (§3.11), v6 was selected from a sweep with everything else pinned to v5's configuration (`[4,256,128,64,4]`, batch 256, 100 epochs, seed 1). All runs logged automatically to `model_a_training_log.csv`.

| Run | Optimizer | $\alpha$ | Final cost $J$ | Mean angle MAE (°) | Pose error, mean (mm) | Elbow selection | Time (s) |
|---|---|---|---|---|---|---|---|
| `v5-repro` | plain GD | $5\times10^{-2}$ | 0.011637 | 17.178 | 48.89 | 57.0% | 63.5 |
| **`v6-lr3e-04`** | **Adam** | $3\times10^{-4}$ | 0.003168 | **16.485** | **31.61** | 57.0% | 72.7 |
| `v6-lr1e-03` | Adam | $1\times10^{-3}$ | 0.002724 | 16.708 | 33.64 | 57.8% | 72.6 |
| `v6-lr3e-03` | Adam | $3\times10^{-3}$ | **0.002673** | 18.250 | 34.53 | 53.6% | 76.6 |
| `v6-lr1e-02` | Adam | $1\times10^{-2}$ | 0.004571 | 16.914 | 41.95 | 58.3% | 74.7 |

**`v5-repro` is a regression test, not a new experiment.** Adding the optimizer switch required refactoring the parameter update inside `nn_model` into a shared helper; re-running plain GD at v5's settings afterward reproduced v5's numbers *exactly* (cost 0.011637, pose 48.89mm, angle MAE 17.178°, selection 57.0%), confirming the refactor left the v0–v5 path untouched. It doubles as the sweep's baseline row.

**The important finding here is that final cost and pose error disagree.** The lowest-cost run ($\alpha=3\times10^{-3}$, $J=0.002673$) has a *worse* pose error (34.53mm) than the selected run ($J=0.003168$, 31.61mm), and also the worst angle MAE and worst branch-selection accuracy in the sweep. Selecting a model by training cost would therefore have picked a measurably worse model by the thesis's own stated criterion (end-effector accuracy). This is a concrete, controlled instance of the metric question already raised in §6.7 — the normalized-MSE training objective is a proxy, and the gap between it and the criterion actually being optimized for is large enough to change which model gets selected. Model selection in this project should be done on pose error; reported cost values should be read as diagnostics of convergence, not as a ranking.

**A prediction that did not hold:** $\alpha=10^{-2}$ was expected to be unstable or divergent. It was neither — it converged, just to a worse solution (0.004571, 41.95mm). Adam's per-parameter normalization evidently makes it more tolerant of an oversized $\alpha$ than plain GD would be at a comparable relative overshoot. Recorded because a stability claim was made in advance and the data contradicted it.

### 4.3 Learning-rate decay sweep for v7

$\alpha_0$ is pinned to $3\times10^{-4}$ (v6's winner from §4.2) throughout, so this sweep changes exactly one thing at a time versus v6: the decay schedule and its decay\_rate. Architecture, batch size, epoch count, and seed are all held at v6's values (`[4,256,128,64,4]`, batch 256, 100 epochs, seed 1). `v6-repro` doubles as the sweep's baseline row and a regression test on the `_step`/decay refactor.

| Run | Decay | decay\_rate | Final $\alpha$ | Final cost $J$ | Mean angle MAE (°) | Pose error, mean (mm) | Elbow selection | Time (s) |
|---|---|---|---|---|---|---|---|---|
| **`v6-repro`** | none | — | $3\times10^{-4}$ | 0.003168 | **16.485** | **31.61** | 57.0% | 59.8 |
| `v7-stair-0.1` | staircase (10) | 0.1 | $\alpha_0/1.9$ | 0.003722 | 16.712 | 33.44 | 56.7% | 62.2 |
| `v7-inv-0.01` | inverse | 0.01 | $\alpha_0/2.0$ | 0.003870 | 16.664 | 33.67 | 57.8% | 62.5 |
| `v7-stair-0.3` | staircase (10) | 0.3 | $\alpha_0/3.7$ | 0.005301 | 17.604 | 36.81 | 55.4% | 60.0 |
| `v7-inv-0.03` | inverse | 0.03 | $\alpha_0/4.0$ | 0.005443 | 17.560 | 36.86 | 56.2% | 59.2 |
| `v7-stair-1.0` | staircase (10) | 1.0 | $\alpha_0/10.0$ | 0.007864 | 17.336 | 43.05 | 56.9% | 61.7 |
| `v7-inv-0.1` | inverse | 0.1 | $\alpha_0/10.9$ | 0.008686 | 17.774 | 46.08 | 56.8% | 63.1 |
| `v7-inv-0.3` | inverse | 0.3 | $\alpha_0/30.7$ | 0.012632 | 17.551 | 59.86 | 57.7% | 60.1 |

**No decay rate at this $\alpha_0$ beat no decay — every row is strictly worse than `v6-repro`, monotonically so as decay\_rate increases.** This is the cleanest possible outcome for isolating cause: cost, angle MAE, and pose error all move together, worsening in lockstep as $\alpha$ is driven down more aggressively (for context: staircase decay\_rate=0.1 finishes at $\alpha_0/1.9 \approx 1.6\times10^{-4}$; inverse decay\_rate=0.3 finishes near $\alpha_0/31 \approx 10^{-5}$, close to two orders of magnitude below $\alpha_0$).

**Why decay hurt here rather than helping (consistent with the prediction in §3.12/§4's interpretation):** decay's benefit comes from letting a *larger* $\alpha_0$ move fast early and then settle late, avoiding the late-training noise-driven wandering a fixed large $\alpha$ would cause. But $\alpha_0=3\times10^{-4}$ here is already the smallest value in the §4.2 sweep and was already the winner — evidence it was not in a "too large, bouncing near the minimum" regime to begin with. Shrinking an already-small step further, on a 100-epoch budget, simply means less total displacement over the run: fewer effective updates reach their full size, so the network is under-trained relative to the no-decay run at the same epoch count. This is a fully mechanical explanation, not a mysterious one, and it predicts (though this was not tested) that decay would look more favorable paired with a *larger* $\alpha_0$ and/or a longer epoch budget than used here — a natural follow-up if this is revisited.

Elbow-selection accuracy stayed flat across every row (54.6%–58.3%, no trend with decay\_rate) — consistent with the established pattern (§4 interpretations, §6 item 3) that branch selection is floored by the problem's one-to-many structure and is untouched by anything optimizer- or schedule-related.

### 4.4 Decay paired with a larger $\alpha_0$ (v8) — a three-pass sweep

§4.3's mechanical explanation for its negative result made a specific, testable prediction: decay would look more favorable paired with a *larger* $\alpha_0$, because $3\times10^{-4}$ was already the smallest/best-tuned fixed rate and not in the "step too large, bouncing near the minimum" regime decay is meant to fix. This section tests that prediction directly, across three widening passes (all Adam, `[4,256,128,64,4]`, batch 256, 100 epochs, seed 1 — only $\alpha_0$ and the decay target vary). **The sweep cells themselves were run interactively in Colab and then deleted from the notebook once the winner was promoted to the canonical training cell**, per the user's request to keep the notebook itself showing only the clean path to the final model; this section is therefore the complete record of the search.

**Pass 1 — coarse grid.** $\alpha_0\in\{10^{-3},3\times10^{-3},10^{-2}\}$ (the three larger rates already characterized in §4.2), each with three decay conditions: no decay (re-run for a same-log comparison), "mild" (decay\_rate=0.01, $\alpha$ halves by epoch 99), and "targeted" (decay\_rate chosen so $\alpha\to\approx3\times10^{-4}$ by epoch 99 — the value that won §4.2).

| $\alpha_0$ | No decay | Targeted decay (→~3e-4) | Decay won? |
|---|---|---|---|
| $10^{-3}$ | 33.64mm | 27.72mm | yes |
| $3\times10^{-3}$ | 34.53mm | **24.35mm** | yes |
| $10^{-2}$ | 41.95mm | 27.89mm | yes |

Decay won at every $\alpha_0$ tested — the opposite of §4.3's result, and exactly as predicted. The mild condition also did well at $\alpha_0=10^{-3}$ (24.40mm, nearly tying the coarse-grid's best), suggesting the region around $\alpha_0\approx10^{-3}$–$3\times10^{-3}$ was worth resolving more finely rather than treating $\alpha_0=3\times10^{-3}$ as a lucky single point.

**Pass 2 — fine grid.** Two sub-sweeps, still inverse decay only (holding schedule type fixed keeps this a single-variable extension of the $\alpha_0$ question):

*Grid A* — finer $\alpha_0$ spacing (`1e-3` to `5e-3`, 8 steps), all decaying to the same $\approx3\times10^{-4}$ target that won Pass 1:

| $\alpha_0$ | Pose error (mm) |
|---|---|
| $1\times10^{-3}$ | 27.72 |
| $1.5\times10^{-3}$ | 25.03 |
| **$2\times10^{-3}$** | **22.54** |
| $2.5\times10^{-3}$ | 23.93 |
| $3\times10^{-3}$ | 24.35 |
| $3.5\times10^{-3}$ | 22.85 |
| $4\times10^{-3}$ | 23.38 |
| $5\times10^{-3}$ | 24.38 |

This is a shallow, well-behaved optimum spanning $\alpha_0\in[2\times10^{-3},3.5\times10^{-3}]$ (all within 22.5–23.9mm), degrading gently on both sides — not a fragile, noise-driven single point. **New best: $\alpha_0=2\times10^{-3}\to\approx3\times10^{-4}$, 22.54mm.**

*Grid B* — at $\alpha_0\in\{2\times10^{-3},3\times10^{-3},4\times10^{-3}\}$, decaying further to $\approx10^{-4}$ instead of $\approx3\times10^{-4}$: uniformly worse than Grid A at the same $\alpha_0$ (e.g. $\alpha_0=3\times10^{-3}$: 31.13mm at target $10^{-4}$ vs. 24.35mm at target $3\times10^{-4}$). This confirms $\approx3\times10^{-4}$ is specifically the right value to *settle near* — not an arbitrary choice, and decaying past it re-triggers the same under-training mechanism as §4.3's original null result, just from a different starting point.

**Pass 3 — even larger $\alpha_0$.** Pushes further out ($\alpha_0\in\{7\times10^{-3},10^{-2},1.5\times10^{-2},2\times10^{-2},3\times10^{-2},5\times10^{-2}\}$, all still targeting $\approx3\times10^{-4}$) to check whether the degradation past $3.5\times10^{-3}$ continues monotonically or a second good regime exists:

| $\alpha_0$ | Pose error (mm) | Initial cost (epoch 0) |
|---|---|---|
| $7\times10^{-3}$ | 25.22 | 0.37 |
| $10^{-2}$ | 27.89 | 0.50 |
| $1.5\times10^{-2}$ | 39.24 | 0.87 |
| $2\times10^{-2}$ | 54.03 | 1.54 |
| $3\times10^{-2}$ | 55.67 | 4.95 |
| $5\times10^{-2}$ | **112.00** | 33.56 |

No second regime — degradation continues and becomes catastrophic. At $\alpha_0=5\times10^{-2}$ the initial cost (33.6, vs. 0.3–0.9 everywhere else) shows the step is too large even for Adam's per-parameter normalization to absorb gracefully at epoch 0; decay (this row's is the fastest of the whole sweep) cannot fully recover from a bad start within 100 epochs. This is a *different* failure mode from §4.3's — that one failed by under-training an already-good rate; this one fails from early instability decay arrives too late to fix. The two mechanisms bracket a genuine sweet spot between them, rather than the sweet spot being one arbitrary lucky value.

**Overall picture, mechanistically complete:**
- $\alpha_0$ too small with decay (§4.3, $\alpha_0=3\times10^{-4}$; or Pass 2's $\alpha_0=10^{-3}$ decayed to $10^{-4}$) → under-trains → worse than the relevant fixed-$\alpha$ baseline.
- $\alpha_0\in[2\times10^{-3},3.5\times10^{-3}]$ decaying to $\approx3\times10^{-4}$ → **best regime found in this project: 22.54–22.85mm.**
- $\alpha_0\gtrsim1.5\times10^{-2}$ → early instability decay can't undo → degrades quickly, catastrophically by $5\times10^{-2}$.

**v8 selection and promotion.** $\alpha_0=2\times10^{-3}$, inverse decay to $\approx3\times10^{-4}$ by epoch 99 (decay\_rate=0.0572) was promoted as the canonical model, re-run with a fresh seed-1 pass to generate the numbers in §4's results table (22.61mm test pose error — matching the sweep's 22.54mm to within run-to-run noise) and re-verified against the §4.1-style singularity diagnostic (correlations stayed weak: $0.054$ elbow, $0.059$ base — consistent with every prior version, the remaining error is still a general precision ceiling, not a singularity effect). This is a **29% improvement over v6/v7's tuned-fixed-$\alpha$ baseline (31.61mm)** and the best result across the entire from-scratch track to date.

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
   - **Strengthened by v6 (§4.2).** The Adam learning-rate sweep produced a case where the *training cost* and the *pose error* rank models differently — the lowest-cost run was not the most accurate one, and selecting on cost would have chosen a model 2.9mm worse in pose error and 1.8° worse in angle MAE. So the metric question is not only about angle-MAE vs. pose error at evaluation time; it also affects **model selection**, and it has now been observed rather than merely hypothesized.
8. **Optimizer progression is now complete through learning-rate decay (§3.12, v7/v8) — the full Course 2 optimizer topic list (momentum/RMSprop/Adam, mini-batching, LR decay) has been implemented, and the best-performing configuration found and promoted.** Decay was first added as a single-variable follow-up on top of v6 (§3.11) at v6's own $\alpha_0$ (§4.3): every configuration tested there was worse than no decay, a genuine negative result caused by $\alpha_0$ already being the smallest/best-tuned rate. That negative result made a specific mechanical prediction — decay should help paired with a *larger* $\alpha_0$ — which was then tested directly across a three-pass sweep (§4.4) and confirmed: $\alpha_0=2\times10^{-3}$ decaying to $\approx3\times10^{-4}$ beat every prior version, **22.61mm test pose error vs. v6/v7's 31.61mm (a 29% improvement)**, now the project's best result. Both the negative and positive decay results are reportable, and together they demonstrate that a well-isolated null finding can directly motivate the experiment that supersedes it.

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

### 08-08-26: Saturday
- Logged v4 (§4): a capacity increase to `[4,256,128,64,4]` found already applied to the notebook (made outside a tracked session) — cost 0.059→0.040, pose error 101.64mm→74.08mm, angle-MAE/selection accuracy roughly flat.
- Added mini-batch gradient descent to Model A (§3.10, `random_mini_batches` + epoch loop, mirroring the Andrew Ng Course 2 convention), keeping the full-batch path available via `batch_size=None` for comparison. Logged as **v5** (§4): with architecture held fixed at v4's, switching to mini-batch GD (batch size 256, 100 epochs) improved cost (0.040→0.012), mean angle MAE (20.60°→17.18°), and pose error (74.08mm→48.89mm) simultaneously, finishing in 65.6 seconds — the cleanest single-variable comparison in the table so far (§4 interpretation). Re-ran the singularity diagnostic (§4.1) on v5: conclusion unchanged, the improvement is general convergence quality, not a singularity effect. Answers the session's motivating question ("this notebook is very slow, how do I make it fast") — logged as QUESTIONS.md Q5.
- Added the **Adam optimizer** to Model A (§3.11), hand-derived like the rest of the track: momentum + RMSprop accumulators with bias correction, selected via a new `optimizer` argument so plain GD stays the default and v0–v5 stay reproducible. Logged as **v6** (§4): with architecture, batch size, epochs, and seed all held at v5's values, pose error fell 48.89mm → **31.61mm (−35%)** and cost 0.012 → 0.0032, while angle MAE moved only 17.18° → 16.49° and elbow-selection accuracy was unchanged at 57.0% — the by-now-familiar split between what convergence can fix and what the problem's one-to-many structure floors.
- Reported v6 alongside a **learning-rate sweep** (§4.2) rather than a single α, because Adam's magnitude-normalized step means α is not comparable across optimizers. Adam beat plain GD at *every* α tested (worst Adam 41.95mm vs. GD's 48.89mm), so the gain is not attributable to a lucky α. The sweep also included a `v5-repro` regression row that reproduced v5's numbers exactly, verifying the `nn_model` refactor.
- **Two findings from the sweep worth carrying into the thesis:** (1) final cost and pose error *ranked the runs differently* — selecting on cost would have picked a worse model by the thesis's own criterion, which turns the §6.7 metric question from a hypothesis into an observed model-selection problem; (2) α = 1e-2 was predicted to diverge and did not, it merely converged worse — recorded as a corrected prediction.
- Re-ran the singularity diagnostic (§4.1) on v6: conclusion unchanged for the third consecutive version. Added a standing note that §4.1's original implication (prefer training improvements over singularity-targeted sampling) has now been acted on twice and paid off as predicted, 101.6mm → 48.9mm → 31.6mm.
- Answers this session's conceptual questions — QUESTIONS.md Q6 (what learning-rate decay is, and why it is orthogonal to rather than redundant with Adam) and Q7 (whether implementing Adam means implementing momentum and RMSprop together).

### 09-08-26: Sunday
- Added **learning-rate decay** to Model A (§3.12): both Course 2 forms (`update_lr`, smooth inverse-time; `schedule_lr_decay`, staircase), selected via a new `decay` argument on `nn_model`, defaulting to `None` so v0–v6 stay bit-identical. Verified via a `v6-repro` regression run (exact match: cost 0.003168, pose 31.61mm, angle MAE 16.485°, 9,400 parameter updates) before trusting any decay row.
- Logged as **v7** (§4) via a decay-rate sweep (§4.3, $\alpha_0=3\times10^{-4}$ pinned to v6's winner, single-variable vs. v6): **every decay configuration tested was worse than no decay**, monotonically so as decay\_rate increased (33.4mm → 59.9mm, vs. 31.61mm for no decay). This was flagged as a real possibility going in (§4's interpretation), not discovered after the fact — $\alpha_0$ was already the smallest value in the §4.2 sweep and had already won, which is not the regime decay is meant to help. v7's reported numbers equal v6's, with `decay=None` as the confirmed winner of a genuine search.
- Re-ran the §4.1 singularity diagnostic and the §5 scatter-plot diagnostic on the v7 run for consistency with prior versions; both reproduce v6's numbers exactly, as expected since v7's final configuration is identical to v6's.
- Lifted the `.claude/skills/thesis_check/SKILL.md` hold on learning-rate decay (see that file's changelog note) now that it's implemented.
- **Tested §4.3's untested prediction directly: paired decay with larger $\alpha_0$ values across a three-pass sweep (§4.4, coarse grid → fine grid → even-larger-$\alpha_0$ grid), 25 runs total.** Result: decay won at every $\alpha_0$ tested up to a point, with a well-characterized optimum at $\alpha_0=2\times10^{-3}$ decaying to $\approx3\times10^{-4}$ (22.54mm in the sweep), degrading on both sides — under-training when $\alpha_0$ too small or decayed too far, catastrophic early instability when $\alpha_0\gtrsim1.5\times10^{-2}$.
- **Promoted this as v8, the new best result (22.61mm test pose error, −29% vs. v6/v7's 31.61mm).** Re-ran the training/evaluate/diagnostic cells with the winning config; the singularity diagnostic re-confirmed the standing conclusion (§4.1) for a fourth version. Per the user's explicit request, deleted the six exploratory sweep cells from the notebook after promotion — the full grid data lives only in `thesis_doc.md` §4.4 now, not in the notebook itself.
