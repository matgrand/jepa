# JEPA: A Deep Technical Guide
### LeJEPA & LeWorldModel — From First Principles

> **Papers covered**
> - **P1** · arXiv:2511.08544 · *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics* — Balestriero & LeCun, 2025
> - **P2** · arXiv:2603.19312 · *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels* — Maes et al., 2026

---

## Table of Contents

1. [What Is JEPA and Why Does It Matter?](#1-what-is-jepa-and-why-does-it-matter)
2. [Formalizing the Problem](#2-formalizing-the-problem)
3. [Why Must Embeddings Be Isotropic Gaussian?](#3-why-must-embeddings-be-isotropic-gaussian)
4. [SIGReg: Sketched Isotropic Gaussian Regularization](#4-sigreg-sketched-isotropic-gaussian-regularization)
5. [LeJEPA: The Full Model (P1)](#5-lejepa-the-full-model-p1)
6. [LeWorldModel: JEPA as a World Model (P2)](#6-leworldmodel-jepa-as-a-world-model-p2)
7. [Synthesis and Comparison](#7-synthesis-and-comparison)
8. [Deep Dives: Statistics and Methods](#8-deep-dives-statistics-and-methods)
   - 8.1 [Integrated Squared Bias (ISB)](#81-integrated-squared-bias-isb)
   - 8.2 [Fisher Information Functional](#82-fisher-information-functional)
   - 8.3 [Testing for Distribution Matching](#83-testing-for-distribution-matching)
   - 8.4 [Spearman Correlation](#84-spearman-correlation)
   - 8.5 [BatchNorm vs. LayerNorm for SIGReg](#85-batchnorm-vs-layernorm-for-sigreg)
9. [Adaptive Layer Normalization and Action Conditioning](#9-adaptive-layer-normalization-and-action-conditioning)

---

## 1. What Is JEPA and Why Does It Matter?

The fundamental challenge of modern AI is learning **useful representations of the world from raw observations** — without labels. Once you have a good representation, every downstream task (classification, planning, control) becomes much easier.

Two classical approaches dominate:

1. **Supervised learning** — requires costly human labels. The labels themselves prevent collapse by anchoring representations to a fixed set of classes.
2. **Reconstruction-based learning** (autoencoders, MAE, etc.) — no labels needed, but the model must reconstruct every pixel, including irrelevant detail. The objective is agnostic to semantics.

**JEPA** (Joint-Embedding Predictive Architecture), proposed formally by Yann LeCun in 2022, is a third path. Instead of reconstructing raw input, it operates entirely in an abstract latent space:

> Given two "views" of the same scene (e.g., two video frames, two crops of an image, past and future), encode both with a shared encoder, and train the model to **predict the embedding of one view from the embedding of the other**. The prediction happens in latent space — no pixel reconstruction required.

```
x  (view 1)  →  Enc(x)  →  z₁  →  Pred  →  ẑ₂  ┐
                                                   ├─  minimize ‖ẑ₂ − z₂‖²
x' (view 2)  →  Enc(x') →  z₂  ─────────────────-┘
```

The encoder can ignore irrelevant details (pixel noise, lighting changes) and focus purely on structure that is *predictable* across views.

### The Collapse Crisis

Many successful SSL methods are de-facto JEPAs: SimCLR, BYOL, DINO, I-JEPA, V-JEPA. They all share the same fundamental failure mode:

**Representation collapse**: if you simply minimize `‖Enc(x) − Enc(x')‖²`, the network learns the trivial solution — map *everything* to the same constant vector. The loss is zero, the representation is useless. A subtler failure is **dimensional collapse**, where embeddings live in a low-dimensional subspace, wasting most representational capacity.

To prevent this, existing methods use an ever-growing list of tricks:

| Method | Anti-Collapse Trick | Limitation |
|---|---|---|
| SimCLR | Contrastive loss with negative pairs | O(N²) memory, needs large batch |
| BYOL | Stop-gradient + EMA teacher | No formal objective; unexplained |
| VICReg | Variance + covariance regularization | Under-specified; moments only |
| I-JEPA | EMA + stop-gradient + masked prediction | EMA schedule is a sensitive hyperparameter |
| PLDM | VICReg with 7 loss terms | Unstable training, O(n⁶) hyperparameter search |

The insight of P1: instead of asking *"what trick prevents collapse?"*, ask *"what distribution should the embeddings have, and why?"* This transforms an ad-hoc engineering problem into a well-posed optimization problem.

---

## 2. Formalizing the Problem

### Data and Notation

The dataset has shape `(N, V, D)`:
- **N** — number of independent samples (images, videos)
- **V** — number of views per sample (augmented crops, or T time-steps)
- **D** — raw input dimension (pixels)

The encoder maps inputs to embeddings: `f_θ : ℝᴰ → ℝᴷ`, where K is the embedding dimension. The goal is to train θ so that `f_θ` becomes a *foundation model* — useful for many downstream tasks without changing its weights.

### The Formal JEPA Objective

```
JEPA(x) ⟺  Enc(x_{n,t+1}) is predictable from Enc(x_{n,t}),  ∀n,t
            AND  Enc(x) is not degenerate
```

Requirement (1) is easy: an MSE loss between predicted and actual embeddings. Requirement (2) — anti-collapse — is where every method differs, and where LeJEPA makes its contribution.

---

## 3. Why Must Embeddings Be Isotropic Gaussian?

This is the central theoretical contribution of P1. The argument is a minimax problem: *which embedding distribution minimizes the worst-case downstream prediction error, across all possible downstream tasks?*

### Intuition: The Agnostic Foundation Model

Imagine training a model with no idea what downstream tasks will be applied. Someone might use your encoder for object recognition, someone else for depth estimation, someone else for anomaly detection. Your embeddings must be useful for all of them.

The worst-case mindset requires a distribution that doesn't "favor" any particular direction in embedding space. If embeddings are anisotropic (spread unevenly), directions with small variance will be numerically weak for a downstream probe — the signal is there but hard to use. An **isotropic distribution** treats all directions equally: it is the maximally neutral, maximally flexible prior.

### Linear Probing: Two Lemmas

A linear probe (ridge regression on frozen embeddings) solves:

```
β̂ = argmin_{β ∈ ℝᴷ}  ‖y − Zβ‖²₂ + λ‖β‖²₂
```

where **Z** ∈ ℝ^{N×K} is the embedding matrix and **y** are unknown labels. Compare two embeddings with identical column spans (same information) but different covariance structures:

- `Z_aniso`: covariance eigenvalues {λ₁, ..., λ_K} — at least two distinct values
- `Z_iso`: all eigenvalues equal to (1/K)∑λ_k — same total energy, uniform geometry

**Lemma 1 — Anisotropy Amplifies Bias:**
For any λ > 0, there always exists a downstream task **y** for which `Z_aniso` produces a higher-bias estimator than `Z_iso`. Tikhonov regularization shrinks β toward zero; if one dimension has very small variance, its coefficient is over-shrunk, introducing bias for tasks relying on that dimension.

**Lemma 2 — Anisotropy Amplifies Variance:**
With λ = 0, `tr(Var(β̂_aniso)) > tr(Var(β̂_iso))`. High-variance dimensions amplify noise in the probe weights, increasing total estimation variance.

Together: anisotropy hurts both bias **and** variance of any linear downstream estimator. Isotropy is strictly better — regardless of the task.

### Nonlinear Probing: The Gaussian Emerges

For nonlinear probes (kNN, kernel regression), the analysis examines the **Integrated Squared Bias (ISB)** as a functional of the embedding distribution p. The key quantity that appears is:

```
J(p) = ∫ ‖∇ log p(z)‖² p(z) dz          # Fisher information functional
```

**Theorem 1 — Isotropic Gaussian Optimality:**
Among all distributions with fixed covariance energy, the isotropic Gaussian is the *unique minimizer* of J(p), and therefore the unique minimizer of integrated squared bias for both kNN and kernel regression.

This is because the Gaussian has minimum Fisher information among all distributions with fixed variance — a classical result in information theory. The isotropic Gaussian is the smoothest distribution consistent with a given variance: it spreads information uniformly across all directions and frequencies of the embedding space.

---

## 4. SIGReg: Sketched Isotropic Gaussian Regularization

We know the target distribution: `N(0, σ²I)`. The question is how to enforce it efficiently.

### The Core Challenge

Directly testing K-dimensional Gaussianity is intractable:
- Classical tests (KS, Anderson-Darling) are designed for 1D
- Multivariate tests (Mardia, Henze-Zirkler) scale as O(K²) or worse
- Moment matching (VICReg) only enforces first two moments — doesn't fully specify the Gaussian

### The Cramér-Wold Theorem: The Key Insight

> A K-dimensional random vector Z has distribution P **if and only if** every one-dimensional projection `u^T Z` has the corresponding one-dimensional marginal distribution.

Applied to Gaussians: if `Z ~ N(0, I_K)`, then for any unit vector **u**, the projection `h = u^T Z ~ N(0, 1)`. To enforce `Z ~ N(0, I_K)`, it suffices to enforce that **all 1D projections are standard normal**. This reduces K-dimensional distribution matching to a sequence of 1D normality tests — and 1D tests are well-understood, cheap, and differentiable.

### The SIGReg Formula

```
# Step 1: Sample M random unit-norm directions u⁽ᵐ⁾ ∈ S^{K-1}

# Step 2: Project embeddings to 1D
h⁽ᵐ⁾ = Z u⁽ᵐ⁾  ∈ ℝᴺ          # N scalar projections per direction

# Step 3-4: Compute and average normality test statistics
SIGReg(Z) = (1/M) ∑ᵐ T(h⁽ᵐ⁾)

# Minimizing SIGReg  →  each projection becomes N(0,1)  →  Z ~ N(0,I)
```

### The Epps-Pulley Test: What Is T(·)?

The paper recommends the **Epps-Pulley (EP) statistic**, a characteristic function-based normality test.

**Characteristic functions.** The characteristic function of a random variable X is:

```
φ_X(t) = E[e^{itX}]          # Fourier transform of the density
                               # For X ~ N(0,1): φ_X(t) = exp(−t²/2)
```

It uniquely characterizes the distribution (via the Lévy continuity theorem).

**The EP statistic.** Given N samples {h₁,...,hₙ}, the empirical CF is:

```
φ̂(t) = (1/N) ∑ⱼ e^{it·hⱼ}
```

The EP test measures the weighted L² distance between the empirical and Gaussian CFs:

```
T_EP(h) = ∫ |φ̂(t) − exp(−t²/2)|² w(t) dt

# w(t) is a Gaussian-shaped weight function
# T_EP → 0  iff  the samples are Gaussian
# Approximated at a discrete grid of evaluation points
```

**Why EP over alternatives:**

| Test | Differentiable | Bounded Gradients | Consistent | Complexity |
|---|---|---|---|---|
| Kolmogorov-Smirnov | No (sup operation) | — | Yes | O(N log N) |
| Anderson-Darling | No | — | Yes | O(N log N) |
| Jarque-Bera (moments) | Yes | No | No (moments only) | O(N) |
| **Epps-Pulley (CF)** | **Yes** | **Yes** (`|e^{itx}|=1`) | **Yes** | **O(N·knots)** |

The bounded gradient property is critical for training stability: since `|e^{itx}| = 1` always, gradient magnitudes are naturally controlled — no exploding gradients from the regularizer.

### Complexity Analysis

| Method | Time | Memory | Full Distribution? |
|---|---|---|---|
| Contrastive (SimCLR) | O(N²·K) | O(N²) | No |
| VICReg (moments) | O(N·K²) | O(K²) | No |
| Whitening | O(N·K²) | O(K²) | No |
| **SIGReg** | **O(N·K·M)** | **O(N·M)** | **Yes** |

With M fixed (M=1024 works well), SIGReg is linear in both N and K — a significant improvement over O(K²) methods for high-dimensional embeddings (e.g., ViT-H has K=1280).

---

## 5. LeJEPA: The Full Model (P1)

### Complete Objective

```
ℒ_LeJEPA = ℒ_pred + λ · SIGReg(Z)

ℒ_pred   = (1/NV) ∑_{n,v} ‖Enc(x_{n,v+1}) − Pred(Enc(x_{n,v}))‖²₂
SIGReg(Z) = (1/M) ∑ᵐ T_EP(Z u⁽ᵐ⁾)
```

`λ` is the **single hyperparameter** controlling the collapse/prediction tradeoff:
- `λ → 0`: only prediction loss → collapse
- `λ → ∞`: only Gaussian constraint → uninformative embeddings
- `λ ≈ 0.1`: Gaussian constraint prevents collapse, prediction loss forces semantic structure

### What LeJEPA Eliminates

- Stop-gradient / detach operations
- Teacher-student networks with EMA schedules
- Negative samples / contrastive pairs
- Whitening layers or explicit normalization
- Multiple loss terms with multiple hyperparameters
- Warmup schedules and LR schedulers for stability

### Training Loss as a Model Selection Criterion

The Spearman correlation between training loss and downstream ImageNet accuracy is **94.5%** across different λ values. This means model selection can be done purely by monitoring the training objective — no supervised evaluation needed. This is a major practical win that no previous SSL method could offer.

### Empirical Results

| Architecture | LeJEPA (frozen) | Setting |
|---|---|---|
| ViT-H/14 | 79.0% top-1 | ImageNet-1k linear probe |
| ConvNeXt-V2 Nano | 76.52% | Galaxy10 domain-specific |
| ResNet-34 | 78.17% | Galaxy10 domain-specific |

Domain-specific LeJEPA on small datasets **outperforms DINOv2/v3 transfer learning** — models pretrained on 100× more natural image data — validating in-domain SSL as a viable and principled strategy.

---

## 6. LeWorldModel: JEPA as a World Model (P2)

### What Is a World Model?

A world model learns the dynamics of an environment: given the current state and an action, predict the next state. With a good world model, you can plan in "imagination" — simulate many possible futures and pick the best action — without interacting with the real environment.

From a control systems perspective: a world model is the learned equivalent of a plant model (state-space equations), operating from raw pixels rather than engineered state vectors. LeWM learns a latent state-space model:

```
z_t     = Enc(o_t)            # encode observation to latent state
ẑ_{t+1} = Pred(z_t, a_t)     # predict next latent state given action
```

Planning then becomes a finite-horizon optimal control problem solved with Model Predictive Control (MPC).

### Architecture

**Encoder:** ViT-Tiny (~5M params). The [CLS] token is passed through a 1-layer MLP with **Batch Normalization** to produce `z_t ∈ ℝ¹⁹²`. BatchNorm here is essential — LayerNorm (used inside ViT) is incompatible with SIGReg (see §8.5).

**Predictor:** 6-layer causal transformer (~10M params). Actions are incorporated via **Adaptive Layer Normalization (AdaLN)** at each layer. AdaLN parameters are initialized to zero, ensuring action conditioning grows progressively from zero influence.

Total: **15M parameters**, trainable on a single GPU in a few hours.

### Training Objective

```
LLeWM = Lpred + λ · SIGReg(Z)

Lpred     = ‖ẑ_{t+1} − z_{t+1}‖²₂           # next-embedding prediction (teacher-forcing)
SIGReg(Z) = mean over time-steps of SIGReg(Z_t)  # step-wise, applied at each timestep
```

No stop-gradient. No EMA. No reconstruction. No reward signal. The entire model trains end-to-end from raw pixels.

```python
def LeWorldModel(obs, actions, lambd=0.1):
    # obs:     (B, T, C, H, W) raw pixel sequences
    # actions: (B, T, A) action sequences
    emb      = encoder(obs)                          # (B, T, D)
    next_emb = predictor(emb, actions)               # (B, T, D)

    pred_loss   = F.mse_loss(emb[:, 1:], next_emb[:, :-1])
    sigreg_loss = mean(SIGReg(emb.transpose(0, 1)))  # step-wise

    return pred_loss + lambd * sigreg_loss
```

### Latent Planning with MPC

At inference time, trajectory optimization is performed in latent space:

```
# Given: initial observation o₁, goal observation o_g
z₁  = Enc(o₁)
z_g = Enc(o_g)

# Optimize action sequence to minimize terminal cost:
a*₁:H = argmin_{a₁:H}  ‖ẑ_H − z_g‖²₂
        where  ẑ_{t+1} = Pred(ẑ_t, a_t)   (autoregressive rollout)
```

Solved with the **Cross-Entropy Method (CEM)**: sample K action sequences, keep the top-J elite sequences by cost, refit a Gaussian to the elite set, repeat. Only the first K planned actions are executed before replanning (MPC strategy).

**Why planning in latent space is fast:** DINO-WM uses ~200 patch tokens per image. LeWM uses a single CLS token (192 dims). Rolling out H steps requires 200× fewer tokens per step → **~48× planning speedup**.

### Physical Understanding in the Latent Space

**Linear/nonlinear probing:** Probes trained on frozen LeWM embeddings recover agent location, block location, and block angle with high accuracy on PushT, consistently outperforming PLDM.

**Temporal path straightening (emergent):** LeWM's latent trajectories become progressively straighter during training — measured by cosine similarity between consecutive latent velocity vectors. This is an emergent property with no explicit regularization, and LeWM achieves *higher* temporal straightness than PLDM, which has a dedicated temporal smoothness term.

**Violation-of-expectation:** When objects are teleported (violating physical continuity), prediction error `‖ẑ_{t+1} − z_{t+1}‖²` spikes sharply. The model reliably detects physically implausible events (p < 0.01, paired t-test) while being relatively insensitive to visual-only perturbations (color changes).

### Limitations

- **Low-complexity environments:** SIGReg forces a K-dimensional Gaussian prior. When the environment's intrinsic dimensionality is much lower than K, the prior is over-constraining. (Two-Room environment is the observed failure case.)
- **Short planning horizons:** Autoregressive rollout accumulates prediction error; MPC re-planning mitigates but doesn't eliminate this.
- **Offline dataset requirements:** Needs sufficient coverage of environment dynamics.
- **Action label dependence:** Requires action annotations during training.

---

## 7. Synthesis and Comparison

### The Logical Chain

Every design choice in both papers follows from a theorem:

```
Foundation model goal: minimize worst-case downstream risk
    ↓
    ask: what distribution minimizes J(p)?
    ↓
Theorem 1: Isotropic Gaussian is uniquely optimal
    ↓
    target: enforce Z ~ N(0, I)
    ↓
Cramér-Wold: full K-dim distribution = all 1D projections
    ↓
    match 1D marginals via differentiable normality tests
    ↓
SIGReg: O(NK), differentiable, bounded gradients
    ↓
LeJEPA = L_pred + λ·SIGReg    →    heuristic-free SSL (P1)
    ↓
LeWorldModel = LeJEPA + latent MPC    →    pixel-to-action world model (P2)
```

### P1 vs. P2 Comparison

| Aspect | P1: LeJEPA | P2: LeWorldModel |
|---|---|---|
| Primary goal | General-purpose representation learning | World model for visual control |
| Input views | Augmented image crops | Consecutive frames + actions |
| Predictor | Simple MLP | Causal transformer + AdaLN |
| Anti-collapse | SIGReg | SIGReg (step-wise) |
| Downstream use | Linear/kNN probing, fine-tuning | Latent MPC planning |
| Scale | Up to 1.8B parameters | 15M parameters, single GPU |
| Key result | 79% ImageNet (ViT-H), stable training | 48× faster planning than DINO-WM |
| Extra insight | Training loss predicts accuracy (94% Spearman) | Emergent temporal path straightening |

### Open Questions

- The Gaussian optimality proof assumes no knowledge of downstream tasks. With partial task knowledge, could you derive a better target distribution?
- Could an adaptive target distribution (matching the data's intrinsic dimensionality) fix the low-complexity environment failure case?
- The Cramér-Wold theorem is exact; in practice M is finite. What is the statistical rate of convergence as a function of M?
- LeWM drops spatial patch information (uses only CLS token). Does this hurt 3D tasks? (OGBench-Cube results suggest yes.)
- Can SIGReg replace heuristics like target networks in value-function learning for RL?

---

## 8. Deep Dives: Statistics and Methods

### 8.1 Integrated Squared Bias (ISB)

At evaluation, you fix the encoder and train a small probe (linear or kNN) on frozen embeddings. The probe sees N training examples {(z_n, y_n)} and must predict labels for new query points q.

**Bias at a single point:**

```
Bias(q) = E[ŷ(q)] − y*(q)
```

The expectation is over different training sets. Even with infinite data, a biased estimator is systematically wrong.

**Integrated Squared Bias:**

```
ISB = ∫ [Bias(q)]² p(q) dq
```

Why **squared**: bias can be positive or negative at different points; squaring prevents cancellation.
Why **integrated**: performance matters everywhere in the embedding space, not just at one point.

**Why ISB and not test error?** In the SSL setting, you're asking a theoretical question before seeing any labels: *given only the embedding distribution, which distribution minimizes expected downstream error?* ISB is a pure functional of p(z) — computable without labels. The key result in P1 is that ISB is proportional to J(p), the Fisher information functional.

### 8.2 Fisher Information Functional

**Classical Fisher information (parametric):** For a distribution `p(x; θ)`, Fisher information measures how much a sample tells you about the parameter θ:

```
I(θ) = E[(d/dθ log p(x; θ))²]
```

The term `d/dθ log p` is the **score function** — how steeply the log-density changes with θ.

**The Fisher information functional** has no parameter. It treats the distribution p(z) itself as the object and measures the "roughness" of its shape:

```
J(p) = ∫ ‖∇ log p(z)‖² p(z) dz
```

- `∇ log p(z)` is the gradient of the log-density at z — pointing in the direction where p changes most steeply
- `‖∇ log p(z)‖²` measures how fast p is changing at z
- Integration weighted by p(z) gives the expected squared log-gradient under the distribution itself

**Intuition:** J(p) measures how rough or spiky the distribution is. A flat distribution has `∇ log p = 0` everywhere, so J(p) = 0. A distribution with sharp peaks and valleys has large gradients, so J(p) is large.

**Why J(p) controls bias:** A kNN estimator predicts y at query q by averaging labels of neighbors within radius r₀. Bias comes from the density of points within that ball being uneven — if p(z) is more concentrated in one corner of the ball, your average is pulled toward that corner. Where p is rough (large `∇ log p`), neighbors are unevenly distributed, and the estimator is biased. Smooth distributions (small J(p)) give uniformly distributed neighbors and unbiased estimates.

**Why the Gaussian minimizes J(p):** This follows from the Cramér-Rao bound:

```
Var(x) · J(p) ≥ 1       (in 1D)
```

Equality holds if and only if p is Gaussian. The Gaussian wastes none of its variance budget on roughness — it spreads probability mass in the smoothest way consistent with a given spread. In K dimensions with fixed trace of covariance, the isotropic Gaussian uniquely minimizes J(p).

### 8.3 Testing for Distribution Matching

**The core idea:** you have N samples from unknown distribution p and want to test if p = q (e.g., q = N(0,I)). Every test computes a scalar **test statistic** T that is small when p ≈ q, large when p ≠ q, and has a known distribution under the null hypothesis p = q.

**Three families of tests:**

**Moment-based** (e.g., Jarque-Bera): compute skewness and kurtosis, compare to Gaussian values (0 and 3). Fast, but only checks 3rd and 4th moments. A distribution can have exactly the right first four moments and still be completely non-Gaussian.

**CDF-based** (e.g., KS, Anderson-Darling): compare the empirical CDF F̂(x) to the target CDF:

```
KS = sup_x |F̂(x) − F(x)|
```

Consistent (detect any fixed deviation with enough samples) but not differentiable — the supremum kills gradients.

**Characteristic function-based** (e.g., Epps-Pulley): compare empirical CF to target CF over a range of frequencies. Differentiable, consistent, and bounded gradients. (See §4 for full treatment.)

**How many samples do you need?**

The empirical CF satisfies:

```
E[|φ̂(t) − φ(t)|²] = (1/N) · Var[e^{itX}] ≤ 1/N
```

Estimation error shrinks as 1/√N. For a fixed difference δ between p and q, you need N ~ 1/δ² samples to detect it reliably.

**The curse of dimensionality:** In K dimensions, estimating the full density to accuracy ε requires N ~ ε^{-K} samples. For K = 512 this is catastrophic. This is why direct K-dimensional tests are impractical.

**How SIGReg escapes the curse:** By projecting onto 1D, you only ever test univariate normality. Sample complexity of a 1D test is independent of K — it depends only on the amount of non-Gaussianity in that direction. The Cramér-Wold theorem guarantees that matching all 1D projections gives the full K-dimensional distribution for free. In practice, M = 1024 projections with batch size 256–1024 gives reliable gradient signal; the paper shows performance is flat above a few hundred projections.

### 8.4 Spearman Correlation

**Pearson correlation** measures linear relationships:

```
r = Cov(X, Y) / (σ_X · σ_Y)
```

Sensitive to outliers, detects only linear trends.

**Spearman correlation**: apply Pearson correlation to the *ranks* of the data, not the data itself.

For N paired observations {(x₁,y₁), ..., (xₙ,yₙ)}:

1. Replace each xᵢ with its rank rᵢˣ (1 = smallest, N = largest)
2. Replace each yᵢ with its rank rᵢʸ
3. Compute Pearson on the ranks: `ρ_S = Corr(rank(X), rank(Y))`

Simplified formula (no ties):

```
ρ_S = 1 − 6·∑(dᵢ²) / (N(N²−1))

where dᵢ = rᵢˣ − rᵢʸ  (rank difference for each pair)
```

**What it measures:** Monotonic relationships — whether X and Y tend to increase together, but not necessarily at a constant rate. Example: if Y = X², both increase together for X > 0, so Spearman ≈ 1, but Pearson < 1 because the relationship isn't linear.

**Why ranks work:**
- **Robust to outliers:** a single huge value just gets rank N — same as a value slightly above the median
- **Invariant to monotonic transformations:** log, sqrt, exp don't change ranks
- **Order information only:** do large X values tend to pair with large Y values?

**In the LeJEPA context:** The 94.5% Spearman correlation between training loss and downstream ImageNet accuracy means you don't need supervised GPU evaluations to pick λ — just monitor the training loss. Spearman rather than Pearson is appropriate here because the loss-accuracy relationship is likely monotonic but nonlinear, and Spearman is more robust to outlier (λ, accuracy) pairs at extreme hyperparameter values.

### 8.5 BatchNorm vs. LayerNorm for SIGReg

**What both norms do:** normalize activations to zero mean and unit variance, then apply a learned affine transform (scale γ and shift β). The difference is *which dimension* you normalize over.

**Batch Normalization:**

```
# For feature dimension D and batch of N samples:
μ_d = (1/N) ∑ᵢ z_{i,d}          # mean over the BATCH, per feature
σ²_d = (1/N) ∑ᵢ (z_{i,d} − μ_d)²
ẑ_{i,d} = (z_{i,d} − μ_d) / σ_d
```

Each feature dimension is normalized **across the batch**. Statistics are computed over all N samples simultaneously.

**Layer Normalization:**

```
# For each sample i independently:
μᵢ = (1/D) ∑_d z_{i,d}          # mean over FEATURES, per sample
σ²ᵢ = (1/D) ∑_d (z_{i,d} − μᵢ)²
ẑ_{i,d} = (z_{i,d} − μᵢ) / σᵢ
```

Each sample is normalized **across its own feature dimensions**. Statistics are per-sample.

**Why LayerNorm kills SIGReg:**

SIGReg works by measuring the distribution of embeddings **across the batch** — it projects the matrix Z ∈ ℝ^{N×K} onto random directions and checks if the N resulting scalars are Gaussian. For this, the batch must exhibit diversity across samples.

LayerNorm normalizes each sample to have zero mean and unit variance *within itself*. Every sample's embedding vector has a constrained shape — its components sum to zero and have unit variance. Two very different inputs can produce embeddings with identical LayerNorm statistics. After LayerNorm, the projection `h = Z u` has its variance controlled by the within-sample normalization, not by between-sample diversity. The test statistic T(h) sees a distribution already constrained by LayerNorm — its gradients tell the encoder to adjust within-sample normalization rather than spread embeddings across the batch.

**Why BatchNorm works:**

BatchNorm normalizes each feature *across the batch*, which directly acts on the quantity SIGReg measures. After BatchNorm, feature d already has zero mean and unit variance across the batch. SIGReg can then push the *joint* distribution of features to be Gaussian — not just marginally normal feature-by-feature, but also decorrelated and jointly Gaussian. The gradient flow is clean and non-competing.

**The deeper reason:**

BatchNorm makes the embedding distribution a function of the *population statistics of the batch* — which is what you want to shape. LayerNorm makes each embedding a function of *its own content* — useful for sequence modeling where each token should be self-consistent, but destructive for distribution-level regularization where cross-sample comparisons must be meaningful.

This is why LeWM explicitly adds a BatchNorm MLP projector after the ViT's final LayerNorm — re-introducing the cross-sample normalization that LayerNorm discards, making the latent space tractable for SIGReg.

---

## 9. Adaptive Layer Normalization and Action Conditioning

### Why Conditioning Is Hard

The predictor in LeWM must solve: given the current latent state z_t and action a_t, predict the next latent state z_{t+1}. The action needs to *modulate how the state is processed*, not just be appended as extra input.

Naive concatenation `[z_t, a_t]` treats the action as another token. Early in training, when attention weights are nearly uniform, the action signal is diluted among state tokens. More fundamentally, you want the action to influence the *geometry* of how the state is transformed at every layer.

This is a **conditioning problem**: how do you make the processing of X depend on some external signal c?

### Standard Layer Normalization (Baseline)

Inside every transformer block, LayerNorm normalizes and then applies a learned affine transform:

```
LayerNorm(x) = γ · (x − μ) / σ + β

# γ, β ∈ ℝᴰ are fixed learned parameters — the same for all inputs
# They are the "personality" of the layer
```

If γ and β could be made *dynamic* — a function of some external signal — you could steer the entire layer's behavior based on that signal. That is exactly what AdaLN does.

### Adaptive Layer Normalization (AdaLN)

Instead of fixed γ and β, predict them from the conditioning signal c (here, c = a_t):

```
AdaLN(x, c):
    [γ, β] = MLP(c)           # predict scale and shift FROM the action
    return γ · (x − μ) / σ + β

# Standard LN:  γ_fixed · norm(x) + β_fixed
# AdaLN:        γ(a_t)  · norm(x) + β(a_t)
```

Every transformer block gets its own MLP mapping the action to (γ, β) for that block. The action can influence each layer differently — a natural curriculum emerges where the network allocates action influence where it is most needed.

### Why AdaLN Works: The Geometric Picture

After LayerNorm, the activation x lives on a normalized hypersphere (zero mean, unit variance). The γ and β perform an **affine remap** — stretching some dimensions, compressing others, shifting the center.

When γ and β come from the action, the action says: *"for this particular action, stretch the representation in these directions and compress it in those."* Different actions carve out different regions of activation space. This is more powerful than concatenation because:

- The action influences the **processing of every token simultaneously** at every layer
- It operates at the level of **representation geometry** (scaling/shifting the normalized space)
- It is **multiplicative** (via γ) as well as additive (via β) — multiplicative conditioning is far more expressive than purely additive injection

### The Zero Initialization Trick

LeWM initializes all AdaLN MLP weights to zero:

```
γ(a_t) = 0,  β(a_t) = 0   for all a_t   (at initialization)
→  AdaLN(x, c) = 0   (action has zero influence at the start)
```

The predictor starts as if actions don't exist; the encoder first learns to make embeddings predictable from state alone. Only as the AdaLN weights grow does action influence emerge. This is an implicit curriculum: first learn "what the world looks like," then learn "how actions change it." Without this, random action gradients fight with state representation gradients — a common source of instability.

### Other Methods for Action Conditioning

**Concatenation:**

```
input = concat([z_t, a_t])
ẑ_{t+1} = Transformer(input)
```

Simple, but the action competes for attention with state tokens and is processed identically to them. Works for weak action signals; not the most parameter-efficient for a global control signal.

**Addition (residual injection):**

```
h = z_t + W_a · a_t          # project action, add to state
# or at every layer:
h_l = Transformer_layer(h_{l-1}) + W_a^{(l)} · a_t
```

Purely a **shift** in activation space — can move the representation to a different region but cannot scale or rotate it. The difference from AdaLN is the difference between translating a point (additive) versus rescaling the axes (AdaLN's γ).

**Cross-attention:**

```
Q = W_Q · z_t
K = W_K · a_t,  V = W_V · a_t
action_context = softmax(QK^T / √d) · V
h = z_t + action_context
```

The state "queries" the action for relevant information. Most expressive option for complex, high-dimensional conditioning signals. For a low-dimensional action (e.g., a 2D velocity command), cross-attention is overkill — full attention to route information from 2 numbers. Pays off when the conditioning signal is long or variable-length (e.g., language instructions).

**FiLM — Feature-wise Linear Modulation:**

```
FiLM(x, c) = γ(c) ⊙ x + β(c)
```

Identical to AdaLN but **without the normalization step**. FiLM directly scales and shifts the unnormalized activation x. Without first normalizing, the magnitudes of γ(c) and β(c) must be calibrated against the typical magnitude of x, which varies across layers and during training. AdaLN's normalization step stabilizes this: the input to the affine transform is always O(1), so the action MLP learns relative adjustments rather than absolute magnitudes. This is why AdaLN is preferred in modern architectures (DiT, etc.) over raw FiLM.

**Hypernetworks:**

```
all_weights = HyperNet(a_t)
ẑ_{t+1} = MainNet(z_t; all_weights)
```

A separate network generates all predictor parameters conditioned on the action. Maximally expressive but computationally catastrophic for large networks. Practical hypernetworks generate only a subset of parameters. Used for multi-task settings where the condition is a task embedding; rarely practical for per-step action conditioning.

### Comparison

| Method | Expressivity | Compute Cost | Best For |
|---|---|---|---|
| Concatenation | Low | Minimal | Simple baselines |
| Addition | Low–medium | Minimal | Weak action signals |
| FiLM | Medium–high | Low | General conditioning |
| AdaLN | High | Low | Transformer blocks, stable training |
| Cross-attention | Very high | Medium–high | Long or complex conditioning signals |
| Hypernetworks | Maximal | Very high | Multi-task, rarely per-step |

AdaLN hits the sweet spot for LeWM: the action is low-dimensional, LayerNorm is already present at every transformer block (so you just make it dynamic), and the zero-init trick provides a clean training curriculum at no extra cost. You get multiplicative expressivity with minimal overhead and built-in training stability.

---

*End of guide. Ready for further questions on any section.*
