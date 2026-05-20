from utils import *
from double_pendulum import DoublePendulum


def test_sigreg():
    """
    Verifies SIGReg by gradient-descending a free embedding matrix toward N(0,I).
    Plots: loss curve, initial vs final distribution of one projection, and
    the embedding scatter before/after.
    """
    torch.manual_seed(42)
    N, D = 512, 32
    sigreg = SIGReg(num_proj=256, knots=17)

    # Start far from Gaussian: skewed, scaled, mean-shifted
    Z = torch.randn(N, D) * 3.0 + 2.0 + 0.5 * torch.randn(N, D)**3 - 1.0
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

    plt.figure(figsize=(14, 4))
    plt.suptitle("SIGReg test: gradient-descending a free embedding toward N(0,I)", fontsize=12)

    # Panel 1: loss curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("SIGReg loss")
    plt.title("Loss curve")
    plt.yscale("log")

    # Panel 2: 1D projection histogram before/after vs N(0,1)
    plt.subplot(1, 3, 2)
    bins = 40
    plt.hist(proj_init,  bins=bins, alpha=0.5, density=True, label="initial")
    plt.hist(proj_final, bins=bins, alpha=0.5, density=True, label="final")
    x = np.linspace(-4, 4, 200)
    plt.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), "k--", label="N(0,1)")
    plt.xlabel("projection value")
    plt.title("1D projection histogram")
    plt.legend()

    # Panel 3: first two dimensions scatter
    plt.subplot(1, 3, 3)
    plt.scatter(Z_init[:, 0].numpy(),  Z_init[:, 1].numpy(),  alpha=0.3, s=5, label="initial")
    plt.scatter(Z_final[:, 0].numpy(), Z_final[:, 1].numpy(), alpha=0.3, s=5, label="final")
    plt.xlabel("dim 0")
    plt.ylabel("dim 1")
    plt.title("Embedding scatter (dims 0-1)")
    plt.legend()
    plt.gca().set_aspect("equal")

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


def test_double_pendulum():
    """Simulate a double pendulum from a near-inverted position and animate it."""
    dp = DoublePendulum(m1=1.0, m2=0.5, l1=1.0, l2=0.8, b=0.05, max_torque=5.0, dt=0.01)
    x0 = np.array([np.pi - 0.2, np.pi + 0.1, 0.0, 0.0])
    traj = dp.sim(x0, u=0.0, t=10.0)
    dp.plot(traj)


if __name__ == "__main__":
    # test_sigreg()
    test_double_pendulum()
