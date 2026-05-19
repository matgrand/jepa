import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization.

    Enforces Z ~ N(0, I) by matching random 1D projections to N(0,1)
    via the Epps-Pulley characteristic function test (Cramér-Wold reduction).

    Args:
        num_proj: number of random 1D projection directions (M).
        knots:    number of quadrature points for the CF integral.
        t_max:    integration range [0, t_max].
    """

    def __init__(self, num_proj: int = 1024, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)

        # Trapezoid rule weights: 2*dt interior, dt at endpoints
        weights = torch.full((knots,), 2.0 * dt)
        weights[[0, -1]] = dt

        phi = torch.exp(-0.5 * t.square())  # N(0,1) CF real part at grid

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: (N, D) or (T, B, D).
        Returns scalar loss. Minimize to push Z ~ N(0, I).
        """
        if Z.dim() == 3:
            return torch.stack([self._compute(Z[i]) for i in range(Z.size(0))]).mean()
        return self._compute(Z)

    def _compute(self, Z: torch.Tensor) -> torch.Tensor:
        N, D = Z.shape

        # Sample M random unit-norm projection directions: (D, M)
        A = torch.randn(D, self.num_proj, device=Z.device, dtype=Z.dtype)
        A = A / A.norm(dim=0, keepdim=True)

        # Project embeddings: h[n, m] = Z[n, :] @ A[:, m] → (N, M)
        h = Z @ A

        # ECF at each quadrature point: (N, M, knots)
        x_t = h.unsqueeze(-1) * self.t

        re_ecf = x_t.cos().mean(dim=0)  # (M, knots)
        im_ecf = x_t.sin().mean(dim=0)  # (M, knots)

        # Squared error vs N(0,1) CF (imaginary target = 0)
        err = (re_ecf - self.phi).square() + im_ecf.square()  # (M, knots)

        # Integrate with quadrature weights, scale by N (EP test statistic)
        return (err @ self.weights * N).mean()
