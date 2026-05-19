# LeJEPA / LeWorldModel — Implementation Coding Guide
### From Source Code to Custom Applications

> **Sources read:** `galilai-group/lejepa` (module.py, MINIMAL.md, train.py) and `lucas-maes/le-wm` (jepa.py, module.py, train.py)
> **Goal:** Give you everything needed to implement JEPA-based world models on *your own data* — physical sensor arrays, time-series, non-pixel inputs — at any scale.

---

## Table of Contents

1. [Architecture at a Glance](#1-architecture-at-a-glance)
2. [SIGReg: The Core Loss — Clean Implementation](#2-sigreg-the-core-loss--clean-implementation)
3. [The Encoder](#3-the-encoder)
4. [The Projector and Why BatchNorm Is Non-Negotiable](#4-the-projector-and-why-batchnorm-is-non-negotiable)
5. [The Predictor (World Model)](#5-the-predictor-world-model)
6. [Action Conditioning with AdaLN](#6-action-conditioning-with-adaln)
7. [The Full Training Loop](#7-the-full-training-loop)
8. [Latent Planning with CEM-MPC](#8-latent-planning-with-cem-mpc)
9. [Using Non-Pixel Inputs (Physical Sensors)](#9-using-non-pixel-inputs-physical-sensors)
10. [Scaling Down: Minimal Viable Configurations](#10-scaling-down-minimal-viable-configurations)
11. [Hyperparameter Guide](#11-hyperparameter-guide)
12. [Common Failure Modes and Fixes](#12-common-failure-modes-and-fixes)

---

## 1. Architecture at a Glance

Both repos implement the same two-component system:

```
TRAINING:

  o_t ──► Encoder ──► [Projector] ──► z_t ─────────────────────────────► SIGReg(Z)
                                        │                                      ▲
                                        ▼                                      │
                             Predictor(z_t, a_t) ──► ẑ_{t+1}                 │
                                                           │                  │
                                                    MSE(ẑ_{t+1}, z_{t+1})    │
                                                           │                  │
                                    Loss = L_pred + λ · SIGReg(Z) ◄──────────┘

INFERENCE (planning):

  o_1 ──► Encoder ──► z_1
  o_g ──► Encoder ──► z_g  (goal)

  CEM: sample actions a_{1:H}, rollout ẑ_1 → ẑ_H, minimize ‖ẑ_H - z_g‖²
```

**Key design decisions from the repos:**

- The encoder output goes through a **BatchNorm projector MLP** before SIGReg sees it. This is not optional — see §4.
- The predictor is **separate** from the encoder. Gradients flow through both.
- No stop-gradient anywhere. No EMA. Everything trains jointly.
- SIGReg is applied to the embedding matrix `Z ∈ (T, B, D)` — across the batch dimension at each time step.

---

## 2. SIGReg: The Core Loss — Clean Implementation

This is the heart of the entire framework. Here it is, fully annotated from the source.

### What it computes

For a batch of embeddings `Z ∈ ℝ^{N×D}`:
1. Sample `M` random unit vectors in `ℝᴰ`
2. Project: `h⁽ᵐ⁾ = Z u⁽ᵐ⁾ ∈ ℝᴺ`
3. Compute the Epps-Pulley statistic on each 1D projection
4. Average across projections

The Epps-Pulley statistic measures the L² distance between the empirical characteristic function (ECF) and the target N(0,1) CF, weighted by a Gaussian window:

```
ECF(t) = (1/N) Σ_j exp(i·t·h_j) = (1/N) Σ_j [cos(t·h_j) + i·sin(t·h_j)]
target_CF(t) = exp(-t²/2)           ← the N(0,1) characteristic function

T_EP = ∫ |ECF(t) - target_CF(t)|² · w(t) dt
     ≈ Σ_k [(ReECF(t_k) - exp(-t_k²/2))² + (ImECF(t_k))²] · w(t_k) · Δt
```

The **symmetry trick** used in both repos: since cos is even and sin is odd, integrating on [0, t_max] and doubling is equivalent to [-t_max, t_max]. This halves the number of integration knots for the same accuracy.

### Clean self-contained implementation

```python
import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization.

    Enforces Z ~ N(0, I) by matching 1D projections to N(0,1)
    via the Epps-Pulley characteristic function test.

    Args:
        num_proj:  number of random 1D projections (M). More = better coverage.
                   Ablations show ~256 is sufficient; 1024 is safe default.
        knots:     number of quadrature points for the CF integral.
                   17 is the default from both repos; robust to this choice.
        t_max:     integration range [0, t_max]. 3.0 covers the relevant
                   frequency range for N(0,1).
    """

    def __init__(self, num_proj: int = 1024, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.num_proj = num_proj

        # Quadrature grid on [0, t_max]
        t = torch.linspace(0, t_max, knots)          # (knots,)
        dt = t_max / (knots - 1)

        # Trapezoid rule weights: 2*dt everywhere, dt at endpoints
        weights = torch.full((knots,), 2.0 * dt)
        weights[[0, -1]] = dt

        # Target N(0,1) characteristic function evaluated at grid: exp(-t²/2)
        phi = torch.exp(-0.5 * t.square())            # (knots,)  real part of N(0,1) CF
        # Im part is 0 (symmetric distribution), so only real part matters

        # Pre-multiply weights by phi to save computation in forward
        self.register_buffer("t", t)                  # (knots,)
        self.register_buffer("phi", phi)              # (knots,) — target Re(CF)
        self.register_buffer("weights", weights * phi)# (knots,) — combined weights

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: embeddings of shape (N, D) or (T, B, D).
               N = batch size, D = embedding dim.
               If (T, B, D): applied step-wise; average over T.

        Returns:
            scalar loss value. Minimize this to push Z ~ N(0, I).
        """
        if Z.dim() == 3:
            # Step-wise application for temporal data: average over time steps
            return torch.stack([self._compute(Z[t]) for t in range(Z.size(0))]).mean()
        return self._compute(Z)

    def _compute(self, Z: torch.Tensor) -> torch.Tensor:
        """Core computation for Z of shape (N, D)."""
        N, D = Z.shape

        # 1. Sample M random unit-norm projection directions
        #    Shape: (D, M)
        A = torch.randn(D, self.num_proj, device=Z.device, dtype=Z.dtype)
        A = A / A.norm(dim=0, keepdim=True)           # unit-normalize each column

        # 2. Project: h[n, m] = Z[n, :] @ A[:, m]
        #    Shape: (N, M)
        h = Z @ A

        # 3. Compute ECF at each quadrature point t_k
        #    x_t[n, m, k] = h[n, m] * t[k]
        #    Shape: (N, M, knots)
        x_t = h.unsqueeze(-1) * self.t               # broadcasting: (N, M, knots)

        # 4. ECF: mean over N samples of exp(i * x_t)
        #    Re(ECF)[m, k] = mean_n cos(x_t[n,m,k])
        #    Im(ECF)[m, k] = mean_n sin(x_t[n,m,k])
        #    Shape: (M, knots)
        re_ecf = x_t.cos().mean(dim=0)               # (M, knots)
        im_ecf = x_t.sin().mean(dim=0)               # (M, knots)

        # 5. Squared error vs target CF
        #    Re error: (Re(ECF) - phi)²,  Im error: Im(ECF)²  (target Im = 0)
        #    Shape: (M, knots)
        err = (re_ecf - self.phi).square() + im_ecf.square()

        # 6. Integrate: dot with quadrature weights, scale by N (test statistic)
        #    Shape: (M,)  then mean over projections
        statistic = (err @ self.weights) * N          # N scaling = test statistic
        return statistic.mean()


# ─── Usage examples ───────────────────────────────────────────────────────────

# Case 1: SSL representation learning (LeJEPA style)
# Z has shape (N, D) where N = batch_size
sigreg = SIGReg(num_proj=256, knots=17)
Z = encoder(batch)                   # (N, D)
loss = sigreg(Z)
loss.backward()

# Case 2: World model (LeWM style) — temporal data
# Z has shape (T, B, D) where T = sequence length, B = batch_size
sigreg = SIGReg(num_proj=256, knots=17)
Z = encoder(observations)            # (B, T, D)
loss = sigreg(Z.transpose(0, 1))     # pass (T, B, D)
loss.backward()

# Case 3: Minimal (small dataset, few dimensions)
sigreg = SIGReg(num_proj=64, knots=9)   # much smaller, still works
```

### The N scaling: why it matters

Both repos multiply the quadrature integral by `N = batch_size`. This makes the statistic scale consistently with the formal EP test statistic, which is defined as `N · ∫|ECF - CF|² w(t) dt`. Without this scaling, the loss magnitude changes with batch size, making λ non-transferable across batch sizes. With the N scaling, the same λ value works across different batch sizes.

### Projection resampling

In both repos, `A` is resampled every forward pass (not cached). This is intentional: fresh random directions every step means that over training, all directions of embedding space are eventually tested. If you cache A, you're only testing the same M directions forever, which can leave blind spots.

---

## 3. The Encoder

### Role

The encoder maps a raw observation `o_t` to a latent vector `z_t ∈ ℝᴷ`. For pixels, this is a ViT or ResNet. For physical sensor arrays, this is an MLP or small 1D-Conv network. The architecture is completely flexible — LeJEPA is architecture-agnostic by design.

### From the LeJEPA repo (vision, for reference)

```python
import timm
import torch.nn as nn
from torchvision.ops import MLP

class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,      # returns 512-dim features
            drop_path_rate=0.1,
        )
        # Critical: projector uses BatchNorm1d, not LayerNorm
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        # x: (N, V, C, H, W) — N samples, V views
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))          # (N*V, 512)
        proj = self.proj(emb).reshape(N, V, -1)       # (N, V, proj_dim)
        return emb, proj
```

### For physical sensor inputs (your use case)

```python
class SensorEncoder(nn.Module):
    """
    Encoder for physical sensor measurements.

    obs_dim:   dimensionality of a single observation vector
               (e.g., 4 for [θ₁, θ₂, ω₁, ω₂] of a double pendulum)
    latent_dim: output embedding dimension. Rule of thumb: 4-16x obs_dim.
                Must be large enough for SIGReg to work (see §10).
    hidden_dim: width of hidden layers.
    """
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # The projector with BatchNorm is the interface to SIGReg
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),   # <── critical, see §4
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        o: (..., obs_dim)  — any leading batch/time dimensions
        returns: (..., latent_dim)
        """
        leading = o.shape[:-1]
        flat = o.reshape(-1, o.shape[-1])
        h = self.net(flat)
        z = self.projector(h)
        return z.reshape(*leading, -1)
```

---

## 4. The Projector and Why BatchNorm Is Non-Negotiable

Both repos share the same critical design: the encoder output goes through an MLP with **BatchNorm1d** before being passed to SIGReg.

```python
# From le-wm/train.py — this is the projector applied after the ViT
projector = MLP(
    input_dim=hidden_dim,
    output_dim=embed_dim,
    hidden_dim=2048,
    norm_fn=torch.nn.BatchNorm1d,   # <── NOT LayerNorm
)
```

**Why BatchNorm and not LayerNorm:**

LayerNorm normalizes each sample *within itself* (across the feature dimension). After LayerNorm, all samples have identical within-sample statistics — the cross-sample diversity that SIGReg measures is destroyed.

BatchNorm normalizes each *feature* across the *batch*. This means feature `d` has zero mean and unit variance across all N samples in the batch. This is exactly the cross-sample structure SIGReg needs to measure and then push toward Gaussianity.

```
After LayerNorm:  each z_i has mean=0, std=1 within itself  ← SIGReg blind
After BatchNorm:  each feature dimension is centered/scaled across batch  ← SIGReg works

Concretely:
  Z = encoder_output + projector_with_BN   →   SIGReg(Z) gives meaningful gradients
  Z = encoder_output + projector_with_LN   →   SIGReg(Z) gradients fight with LN
```

**Practical consequence:** if your encoder already ends with LayerNorm (as ViTs do), you *must* add a BatchNorm projection head on top. The head's only job is to re-introduce cross-sample structure that LayerNorm strips out.

**For small datasets or batch sizes:** if your batch size is very small (< 32), BatchNorm becomes unstable because the batch statistics are noisy. In this case, use **Group Normalization** as a substitute:

```python
# For small batch sizes: replace BatchNorm1d with a 1-group GroupNorm
# which normalizes each sample's features (functionally like LayerNorm but
# applied after the linear layer rather than before, preserving cross-sample diversity)
nn.GroupNorm(1, hidden_dim)   # 1 group = LayerNorm semantics but different position

# OR: use running statistics with track_running_stats=False to avoid stale BN stats
nn.BatchNorm1d(hidden_dim, track_running_stats=False)
```

---

## 5. The Predictor (World Model)

### Role

The predictor maps `(z_t, a_t) → ẑ_{t+1}`. It models the environment dynamics in latent space. The predictor is only needed for the world model use case (LeWM). For pure representation learning (LeJEPA), no predictor is needed — just the prediction loss between two views.

### From le-wm/module.py — the autoregressive predictor

```python
class ARPredictor(nn.Module):
    """
    Autoregressive predictor: maps a history of (state, action) pairs
    to the next state embedding.

    Key design choices:
    - Transformer with causal masking (can't look at future states)
    - Action injected via AdaLN (Adaptive Layer Normalization), NOT concatenation
    - Positional embeddings are learned parameters
    - Input/output projections handle dimension mismatches
    """
    def __init__(self, *, num_frames, depth, heads, mlp_dim,
                 input_dim, hidden_dim, output_dim=None, dim_head=64, dropout=0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(
            input_dim, hidden_dim, output_dim or input_dim,
            depth, heads, dim_head, mlp_dim, dropout,
            block_class=ConditionalBlock,  # AdaLN blocks
        )

    def forward(self, x, c):
        """
        x: (B, T, D)   — latent state history
        c: (B, T, A)   — action embedding history (same length)
        returns: (B, T, D)  — predicted next states (shifted by 1)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]    # add positional embeddings
        x = self.dropout(x)
        x = self.transformer(x, c)           # c conditions each block via AdaLN
        return x
```

### Simpler predictor for small-scale / non-pixel use

For physical sensor data with small observation dimension, a simple MLP predictor with additive action injection works well:

```python
class MLPPredictor(nn.Module):
    """
    Simple MLP predictor for low-dimensional state spaces.
    Action is injected additively — sufficient when action dim is small
    and action influence is simple.
    """
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D) or (B, T, D)
        a: (B, A) or (B, T, A)
        """
        return self.net(z + self.action_proj(a))


class GRUPredictor(nn.Module):
    """
    GRU-based predictor — better than MLP for longer sequences
    because it has explicit hidden state memory.
    Good middle ground between MLP and full Transformer.
    """
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor,
                h: torch.Tensor = None) -> tuple:
        """
        z: (B, T, D)
        a: (B, T, A)
        returns: predictions (B, T, D), hidden state (1, B, H)
        """
        inp = torch.cat([z, a], dim=-1)      # (B, T, D+A)
        out, h = self.gru(inp, h)             # (B, T, H)
        return self.out_proj(out), h
```

---

## 6. Action Conditioning with AdaLN

From `le-wm/module.py`, here is the complete ConditionalBlock with AdaLN-zero:

```python
def modulate(x, shift, scale):
    """AdaLN modulation: scale and shift normalized activations."""
    return x * (1 + scale) + shift   # note: (1 + scale) not just scale


class ConditionalBlock(nn.Module):
    """
    Transformer block with AdaLN-zero conditioning.

    AdaLN-zero means: initialize the conditioning MLP to output zeros,
    so at the start of training the action has zero influence.
    The model first learns to predict without actions, then gradually
    incorporates action information as the MLP weights grow.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)

        # elementwise_affine=False: no learned γ/β in LayerNorm itself
        # because AdaLN will provide them externally
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Projects conditioning signal c → 6*dim:
        # [shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        # CRITICAL: zero initialization → zero action influence at init
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """
        x: (B, T, D) — state tokens
        c: (B, T, D) — action conditioning (same shape after projection)
        """
        # 6 modulation signals from the action
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Attention with AdaLN modulation + gating
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # FFN with AdaLN modulation + gating
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x
```

**The gate terms** (`gate_msa`, `gate_mlp`) are an additional feature: they scale the residual connection contribution by an action-dependent factor. At init (zero weights), gates = 0, so the block is effectively an identity — another stability mechanism on top of the zero-shift/scale.

**For simple MLP predictors**, AdaLN is overkill. Use additive injection:

```python
# Simple action injection: project action to same dim as state, add before MLP
h = z + self.action_proj(a)
pred = self.mlp(h)
```

---

## 7. The Full Training Loop

### From le-wm/train.py — the forward function (simplified and annotated)

```python
def training_step(encoder, predictor, sigreg, batch, cfg):
    """
    Core LeWM training step.

    batch: dict with keys:
        'obs':    (B, T, obs_dim)   — sequence of observations
        'action': (B, T, act_dim)   — sequence of actions (action[t] leads to obs[t+1])

    Returns dict of losses.
    """
    obs    = batch['obs']                # (B, T, obs_dim)
    action = batch['action']             # (B, T, act_dim)

    # Handle NaN at sequence boundaries (from dataset padding)
    action = torch.nan_to_num(action, 0.0)

    # 1. Encode all observations to get full embedding sequence
    B, T, _ = obs.shape
    emb = encoder(obs.reshape(B*T, -1)).reshape(B, T, -1)   # (B, T, D)

    # 2. Predict next embeddings (teacher forcing)
    #    Predictor sees obs[0:T-1] and actions[0:T-1], predicts obs[1:T]
    pred_emb = predictor(emb[:, :-1], action[:, :-1])        # (B, T-1, D)
    target   = emb[:, 1:]                                     # (B, T-1, D)

    # 3. Prediction loss: MSE between predicted and actual next embedding
    pred_loss = (pred_emb - target).pow(2).mean()

    # 4. SIGReg: applied to all embeddings, step-wise across time
    #    emb.transpose(0,1) gives shape (T, B, D)
    sigreg_loss = sigreg(emb.transpose(0, 1))

    # 5. Combined loss — λ is the single hyperparameter
    loss = pred_loss + cfg.lambda_sigreg * sigreg_loss

    return {
        'loss': loss,
        'pred_loss': pred_loss.detach(),
        'sigreg_loss': sigreg_loss.detach(),
    }
```

### From galilai-group/lejepa — the SSL version (no actions)

```python
def lejepa_step(encoder, sigreg, batch, lambda_sigreg):
    """
    LeJEPA step for SSL (no world model, no actions).

    batch: (N, V, ...) — N samples, V augmented views each.
    The prediction loss is simply that all V views of the same sample
    should map to the same embedding (invariance loss).
    """
    images = batch                               # (N, V, C, H, W) for pixels
                                                 # (N, V, obs_dim) for sensors
    N, V = images.shape[:2]

    # Encode all views
    emb, proj = encoder(images)                  # proj: (N, V, D) after projector

    # Invariance loss: all views of the same sample should agree
    # mean(0) = average embedding per sample (across V views)
    # then MSE of each view against the mean
    mean_proj = proj.mean(dim=1, keepdim=True)   # (N, 1, D)
    inv_loss = (proj - mean_proj).pow(2).mean()

    # SIGReg: applied to proj reshaped to (N*V, D)
    # Or equivalently to proj.reshape(-1, D)
    sigreg_loss = sigreg(proj.reshape(-1, proj.shape[-1]))

    # Combined: note lejepa uses (1-λ) for inv_loss, λ for sigreg
    # This keeps total loss scale roughly constant regardless of λ
    loss = lambda_sigreg * sigreg_loss + (1 - lambda_sigreg) * inv_loss

    return loss, inv_loss.detach(), sigreg_loss.detach()
```

### Training configuration (from lejepa README)

```python
# Optimizer: AdamW with separate weight decay for backbone vs. probe
optimizer = torch.optim.AdamW([
    {'params': encoder.parameters(), 'lr': 5e-4, 'weight_decay': 5e-2},
])

# LR schedule: linear warmup then cosine decay
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-3)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

# Mixed precision (highly recommended even on small models)
from torch.amp import GradScaler, autocast
scaler = GradScaler()

# Training step
with autocast('cuda', dtype=torch.bfloat16):
    losses = training_step(...)

optimizer.zero_grad()
scaler.scale(losses['loss']).backward()
scaler.step(optimizer)
scaler.update()
scheduler.step()
```

---

## 8. Latent Planning with CEM-MPC

From `le-wm/jepa.py`, the rollout and planning logic:

```python
def plan_cem(encoder, predictor, obs_init, obs_goal, action_dim,
             horizon=10, n_samples=512, n_elite=64, n_iter=5,
             history_size=3, device='cuda'):
    """
    Cross-Entropy Method planning in latent space.

    Given initial and goal observations, find the action sequence that
    drives the predicted latent state to the goal latent state.

    Args:
        horizon:    planning horizon H (number of steps to optimize)
        n_samples:  number of action sequences sampled per CEM iteration
        n_elite:    top-k sequences kept for distribution update
        n_iter:     number of CEM refinement iterations
        history_size: how many past states the predictor uses
    """
    with torch.no_grad():
        z_init = encoder(obs_init.to(device))     # (1, D)
        z_goal = encoder(obs_goal.to(device))     # (1, D)

    # CEM: initialize action distribution as N(0, 1)
    mu  = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device)

    for iteration in range(n_iter):
        # Sample n_samples action sequences
        eps = torch.randn(n_samples, horizon, action_dim, device=device)
        actions = mu.unsqueeze(0) + std.unsqueeze(0) * eps  # (S, H, A)

        # Rollout: simulate all S sequences from z_init
        costs = _rollout_cost(encoder, predictor, z_init, z_goal,
                              actions, history_size)        # (S,)

        # Select elite sequences (lowest cost)
        elite_idx = costs.topk(n_elite, largest=False).indices
        elite_actions = actions[elite_idx]                  # (n_elite, H, A)

        # Refit Gaussian to elite set
        mu  = elite_actions.mean(dim=0)                     # (H, A)
        std = elite_actions.std(dim=0).clamp(min=1e-3)     # (H, A)

    # Return best action sequence
    best_idx = costs.argmin()
    return actions[best_idx]                                # (H, A)


def _rollout_cost(encoder, predictor, z_init, z_goal, actions, history_size):
    """
    Roll out the predictor for all action sequences and compute terminal cost.

    z_init:  (1, D)
    z_goal:  (1, D)
    actions: (S, H, A)
    returns: (S,) terminal costs
    """
    S, H, A = actions.shape
    D = z_init.shape[-1]

    # Initialize: expand z_init for all S sequences
    # history: (S, 1, D)
    z_hist = z_init.expand(S, 1, D).clone()

    for t in range(H):
        a_t = actions[:, t, :]                              # (S, A)

        # Use last history_size states
        z_ctx = z_hist[:, -history_size:]                   # (S, hs, D)
        a_ctx = a_t.unsqueeze(1).expand(-1, z_ctx.size(1), -1)  # (S, hs, A)

        z_next = predictor(z_ctx, a_ctx)[:, -1:]           # (S, 1, D)
        z_hist = torch.cat([z_hist, z_next], dim=1)        # (S, t+2, D)

    # Terminal cost: MSE between last predicted state and goal
    z_final = z_hist[:, -1, :]                             # (S, D)
    cost = (z_final - z_goal.expand(S, -1)).pow(2).sum(dim=-1)  # (S,)
    return cost
```

**MPC wrapper:** in practice, you don't execute all H actions. You execute only the first K (receding horizon), then replan:

```python
def mpc_controller(encoder, predictor, env, n_steps, horizon=10, replan_every=3):
    obs = env.reset()
    for step in range(n_steps):
        if step % replan_every == 0:
            action_sequence = plan_cem(encoder, predictor, obs, goal_obs, ...)
            planned_actions = action_sequence[:replan_every]
        action = planned_actions[step % replan_every]
        obs, reward, done, _ = env.step(action.cpu().numpy())
```

---

## 9. Using Non-Pixel Inputs (Physical Sensors)

**Short answer: yes, the architecture works excellently for physical sensor arrays.** In fact, it may work *better* than for pixels in several ways:

1. **Lower input dimensionality** means the encoder can be much smaller and still extract meaningful representations
2. **Physical measurements are already information-dense** — there's no pixel-level redundancy to filter out
3. **SIGReg is data-type agnostic** — it only ever sees the latent embeddings, not the raw inputs

### What changes vs. pixel inputs

| Component | Pixel version | Sensor version |
|---|---|---|
| Encoder | ViT or ResNet | MLP or 1D-Conv |
| Input normalization | ImageNet mean/std | Per-feature z-score |
| View generation (SSL) | Random crop + color jitter | Noise + subsample + time shift |
| Sequence structure | T frames × (H×W×3) | T timesteps × obs_dim |
| Projector | Always BatchNorm MLP | BatchNorm MLP (same) |

### Input normalization is critical

Physical measurements have very different scales (e.g., angles in [-π, π], velocities in [-10, 10], forces in [-100, 100]). Normalize each feature to zero mean and unit variance over the training set:

```python
class RunningNormalizer(nn.Module):
    """
    Tracks running mean and std for input normalization.
    Fit on training data before training, then freeze.
    """
    def __init__(self, obs_dim: int):
        super().__init__()
        self.register_buffer('mean', torch.zeros(obs_dim))
        self.register_buffer('std',  torch.ones(obs_dim))

    def fit(self, data: torch.Tensor):
        """data: (N, obs_dim)"""
        self.mean.copy_(data.mean(dim=0))
        self.std.copy_(data.std(dim=0).clamp(min=1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean
```

### View generation for sensor data

For SSL (no actions, LeJEPA style), you need a way to generate multiple "views" of the same measurement that share semantic content. Options:

```python
def generate_sensor_views(obs: torch.Tensor, n_views: int = 2) -> torch.Tensor:
    """
    obs: (N, obs_dim)
    returns: (N, n_views, obs_dim)
    """
    views = []
    for _ in range(n_views):
        v = obs.clone()
        # Option 1: Additive Gaussian noise
        v = v + 0.01 * torch.randn_like(v)

        # Option 2: Random feature dropout (mask some sensors)
        mask = torch.rand_like(v) > 0.1   # drop 10% of features
        v = v * mask

        # Option 3: For temporal data — small time shift
        # (return obs at t vs obs at t+1 as the two views)
        views.append(v)
    return torch.stack(views, dim=1)
```

For the **world model case** (LeWM style), views are simply consecutive timesteps in a trajectory — no augmentation needed.

### Minimum latent dimensionality for SIGReg

SIGReg needs a latent space large enough to "fill" with Gaussian structure. A rule of thumb from the papers: `latent_dim >= 4 * intrinsic_dim_of_data`. For a double pendulum (4-dimensional state space), `latent_dim >= 16` is typically sufficient. Using `latent_dim = 32` or `64` provides margin.

If your latent dimension is too small relative to the dataset's intrinsic dimensionality, SIGReg will fight against the prediction loss (not enough dimensions to spread information Gaussian-ly while also being predictive).

---

## 10. Scaling Down: Minimal Viable Configurations

The original LeWM uses ~15M parameters total. For physical experiments you may want much less. Here are working configurations at different scales:

### Tiny (< 10K parameters, small datasets, fast iteration)

```python
OBS_DIM      = 4       # e.g., double pendulum: [θ₁, θ₂, ω₁, ω₂]
ACTION_DIM   = 1       # single torque
LATENT_DIM   = 32      # must be >> obs_dim for SIGReg
HIDDEN_DIM   = 64

ENCODER_LAYERS = 2
PREDICTOR_LAYERS = 2

NUM_PROJ  = 64         # SIGReg projections — 64 is fine for D=32
KNOTS     = 9          # quadrature points — 9 is sufficient
LAMBDA    = 0.1        # start here, tune with bisection

BATCH_SIZE = 128       # SIGReg needs B >= 64 for stable statistics
SEQ_LEN    = 8         # history length for predictor

LR           = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 200
```

### Small (< 100K parameters, moderate datasets)

```python
OBS_DIM      = 4
ACTION_DIM   = 1
LATENT_DIM   = 64
HIDDEN_DIM   = 128

ENCODER_LAYERS = 3
PREDICTOR_LAYERS = 3

NUM_PROJ  = 256
KNOTS     = 17         # paper default
LAMBDA    = 0.1

BATCH_SIZE = 256
SEQ_LEN    = 16
```

### Minimum batch size warning

SIGReg computes sample statistics over the batch. With very small batches (< 32), the empirical CF is noisy and the test has low power. **B = 128 minimum is recommended.** If your dataset is tiny, use the full dataset or a large fraction per batch.

---

## 11. Hyperparameter Guide

### λ (lambda_sigreg) — the only real hyperparameter

This controls the tradeoff between prediction quality and anti-collapse enforcement.

**Bisection search (from LeWM paper):** binary search over λ in log-scale:

```python
# Start with λ=0.1 and check for collapse
# If collapse (SIGReg loss stays high after warmup): increase λ
# If predictions poor (pred_loss barely decreases): decrease λ

# Suggested search grid:
lambdas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

# Diagnosis:
# sigreg_loss ≫ 0 after 20% training → λ too small (not enough anti-collapse pressure)
# pred_loss ≫ 0 and not decreasing → λ too large (over-regularized)
# Both losses decrease smoothly → good λ
```

**LeJEPA repo observation:** λ = 0.1 is robust across architectures and datasets.

### num_proj (M) — SIGReg projections

From ablations in LeWM: performance is largely flat above ~64–128 projections. 256 is a safe choice for small D. 1024 is used in the paper for large D (e.g., D=192 for ViT-Tiny CLS token).

Rule of thumb: `num_proj = max(64, 4 * latent_dim)`.

### knots — quadrature points

17 in both repos. Robust to this choice (tested 9–33 in LeWM ablations). Keep at 17 unless you have a reason to change.

### Sequence length / history size

LeWM uses `history_size = 3` (the predictor sees the last 3 frames). This is the context length of the causal transformer.

For physical systems, longer history helps if the system has long-range memory (e.g., a pendulum — the momentum is encoded in recent positions). Shorter history works for Markovian systems.

### Data normalization

Always normalize inputs to zero mean and unit variance before the encoder. This is not optional — unnormalized inputs lead to pathological gradient behavior with any deep network.

---

## 12. Common Failure Modes and Fixes

### Collapse (SIGReg loss stays high, pred_loss → 0 immediately)

The prediction loss collapsed to a trivial solution (constant embeddings) before SIGReg could prevent it.

**Fix:** increase λ. Or reduce learning rate. Or add a small amount of weight decay.

**Diagnosis:** plot the PCA of embeddings — if all embeddings cluster at a single point, it's collapse.

```python
# Collapse detection: measure embedding variance across batch
def check_collapse(Z: torch.Tensor) -> float:
    """Returns fraction of variance in top-1 principal component.
    >0.99 = collapse, <0.5 = healthy, ~1/D = perfectly isotropic"""
    cov = torch.cov(Z.T)
    eigvals = torch.linalg.eigvalsh(cov)
    return (eigvals[-1] / eigvals.sum()).item()
```

### SIGReg loss explodes early in training

The BatchNorm projector statistics are not yet initialized — early in training the embedding distribution can be very non-Gaussian, making the EP statistic large.

**Fix:** this is normal and expected. SIGReg loss drops sharply in the first few hundred steps (as shown in LeWM training curves). Don't panic or tune based on early training behavior.

### Predictions are poor despite loss decreasing

The predictor is overfitting to the training trajectories or the latent space is not structured enough for dynamics to be predictable.

**Fix:** increase `LATENT_DIM`, increase `PREDICTOR_LAYERS`, or check that your dataset has sufficient trajectory diversity.

### BatchNorm issues with variable-length sequences

If you flatten `(B, T, D) → (B*T, D)` for BatchNorm, the batch statistics mix across time steps. This is fine if your data is i.i.d. across time, but problematic if statistics drift across the sequence.

**Fix:** in LeWM, the BatchNorm projector is applied *before* the temporal structure enters — the encoder sees each frame independently. Keep this structure:

```python
# CORRECT: flatten to (B*T, D) for encoder + projector, then unflatten
emb_flat = encoder(obs.reshape(B*T, obs_dim))   # BatchNorm sees B*T samples
emb = emb_flat.reshape(B, T, latent_dim)         # then restore temporal structure
pred = predictor(emb, actions)                    # predictor operates on (B, T, D)

# WRONG: applying BatchNorm after the temporal predictor
```

### Small dataset: SIGReg behaves erratically

With very few trajectories, the embedding distribution has low intrinsic dimensionality — SIGReg is trying to fill a K-dimensional Gaussian but the data only spans a much smaller manifold.

**Fix:** either reduce `LATENT_DIM` to better match the data's intrinsic complexity, or increase dataset diversity. The LeWM paper notes this exact issue for Two-Room (very simple environment).

---

## Quick Reference: Repository Structure

### galilai-group/lejepa

```
lejepa/
├── lejepa/
│   ├── univariate.py    ← EP and other 1D normality test implementations
│   └── multivariate.py  ← SlicingUnivariateTest (SIGReg) implementation
├── scripts/             ← Training scripts for various architectures
├── eval/                ← Linear probe evaluation
├── MINIMAL.md           ← ← ← Start here: 130-line working example
└── README.md
```

**Install the package:** `pip install lejepa` — gives you clean Python API:

```python
import lejepa

ep_test = lejepa.univariate.EppsPulley(num_points=17)
sigreg  = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=ep_test,
    num_slices=256
)
loss = sigreg(embeddings)   # embeddings: (N, D)
```

### lucas-maes/le-wm

```
le-wm/
├── jepa.py    ← JEPA class: encode, predict, rollout, planning (get_cost)
├── module.py  ← SIGReg, ConditionalBlock (AdaLN), ARPredictor, Embedder, MLP
├── train.py   ← Full training loop with Hydra + Lightning + WandB
├── eval.py    ← MPC evaluation
└── config/    ← Hydra config files for environments
```

The core logic is in `jepa.py` + `module.py`. `train.py` is boilerplate around Lightning/Hydra.
