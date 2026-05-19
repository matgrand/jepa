import torch
import matplotlib.pyplot as plt
from utils import SIGReg


def test_sigreg():
    """
    Verifies SIGReg by gradient-descending a free embedding matrix toward N(0,I).
    Plots: loss curve, initial vs final distribution of one projection, and
    the embedding scatter before/after.
    """
    torch.manual_seed(0)
    N, D = 512, 32
    sigreg = SIGReg(num_proj=256, knots=17)

    # Start far from Gaussian: skewed, scaled, mean-shifted
    Z = torch.randn(N, D) * 3.0 + 2.0
    Z = Z.requires_grad_(True)
    optimizer = torch.optim.Adam([Z], lr=0.05)

    losses = []
    n_steps = 300
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = sigreg(Z)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    Z_final = Z.detach()
    Z_init  = (torch.randn(N, D) * 3.0 + 2.0)  # same distribution, for reference

    # --- fixed random projection direction for the 1D histogram comparison ---
    torch.manual_seed(42)
    u = torch.randn(D)
    u = u / u.norm()
    proj_init  = (Z_init  @ u).numpy()
    proj_final = (Z_final @ u).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("SIGReg test: gradient-descending a free embedding toward N(0,I)", fontsize=12)

    # Panel 1: loss curve
    ax = axes[0]
    ax.plot(losses)
    ax.set_xlabel("step")
    ax.set_ylabel("SIGReg loss")
    ax.set_title("Loss curve")
    ax.set_yscale("log")

    # Panel 2: 1D projection histogram before/after vs N(0,1)
    import numpy as np
    ax = axes[1]
    bins = 40
    ax.hist(proj_init,  bins=bins, alpha=0.5, density=True, label="initial")
    ax.hist(proj_final, bins=bins, alpha=0.5, density=True, label="final")
    x = np.linspace(-4, 4, 200)
    ax.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), "k--", label="N(0,1)")
    ax.set_xlabel("projection value")
    ax.set_title("1D projection histogram")
    ax.legend()

    # Panel 3: first two dimensions scatter
    ax = axes[2]
    ax.scatter(Z_init[:, 0].numpy(),  Z_init[:, 1].numpy(),  alpha=0.3, s=5, label="initial")
    ax.scatter(Z_final[:, 0].numpy(), Z_final[:, 1].numpy(), alpha=0.3, s=5, label="final")
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    ax.set_title("Embedding scatter (dims 0–1)")
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("test_sigreg.png", dpi=120)
    plt.show()

    loss_init  = sigreg(Z_init).item()
    loss_final = sigreg(Z_final).item()
    mean_final = Z_final.mean().item()
    std_final  = Z_final.std().item()
    print(f"SIGReg loss  — initial: {loss_init:.2f}  →  final: {loss_final:.4f}")
    print(f"Embedding stats after optimization — mean: {mean_final:.4f}, std: {std_final:.4f}")
    print("Plot saved to test_sigreg.png")


if __name__ == "__main__":
    test_sigreg()
