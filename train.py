from torch.utils.data import TensorDataset, DataLoader
from utils import *
from double_pendulum import DoublePendulum


# create the dataset
NDS = 100  # number of trajectories
DT = 0.01  # time step for dataset (s)
T = 0.3    # trajectory length (s)

dp = DoublePendulum(dt=DT)

n_steps = int(round(T / DT))
x_curr_list, u_list, x_next_list = [], [], []

for _ in tqdm(range(NDS), desc="Generating dataset"):
    x0 = np.array([
        np.random.uniform(-np.pi, np.pi),
        np.random.uniform(-np.pi, np.pi),
        np.random.uniform(-2.0, 2.0),
        np.random.uniform(-2.0, 2.0),
    ])
    u_val = np.random.uniform(-dp.max_torque, dp.max_torque)
    traj = dp.sim(x0, u=u_val, t=T)          # (n_steps+1, 4)
    x_curr_list.append(traj[:-1])             # (n_steps, 4)
    u_list.append(np.full((n_steps, 1), u_val))
    x_next_list.append(traj[1:])

X_curr = torch.tensor(np.concatenate(x_curr_list), dtype=torch.float32)
U      = torch.tensor(np.concatenate(u_list),      dtype=torch.float32)
X_next = torch.tensor(np.concatenate(x_next_list), dtype=torch.float32)

ds = TensorDataset(X_curr, U, X_next)
dl = DataLoader(ds, batch_size=256, shuffle=True)


# net architecture
HID_END = 16
HID_DYN = 16
EMB = 2


class Enc(nn.Module):
    """MLP encoder: state (4) → embedding (EMB)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HID_END),    nn.SiLU(),
            nn.Linear(HID_END, HID_END), nn.SiLU(),
            nn.Linear(HID_END, EMB),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Dyn(nn.Module):
    """MLP dynamics predictor: (embedding, torque) → next embedding."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB + 1, HID_DYN), nn.SiLU(),
            nn.Linear(HID_DYN, HID_DYN), nn.SiLU(),
            nn.Linear(HID_DYN, EMB),
        )

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, u], dim=-1))


# training loop
enc = Enc()
dyn = Dyn()
sigreg = SIGReg(num_proj=256, knots=17)
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dyn.parameters()), lr=1e-3)

LAMBDA_SIG = 0.01
N_EPOCHS = 100

pred_losses, reg_losses = [], []

for epoch in range(N_EPOCHS):
    ep_pred, ep_reg = 0.0, 0.0
    for x_t, u_t, x_tp1 in dl:
        optimizer.zero_grad()

        z_t          = enc(x_t)
        z_tp1_pred   = dyn(z_t, u_t)
        z_tp1_target = enc(x_tp1).detach()  # stop-gradient on target (JEPA)

        pred_loss = nn.functional.mse_loss(z_tp1_pred, z_tp1_target)
        reg_loss  = sigreg(z_t)
        loss = pred_loss + LAMBDA_SIG * reg_loss

        loss.backward()
        optimizer.step()
        ep_pred += pred_loss.item()
        ep_reg  += reg_loss.item()

    pred_losses.append(ep_pred / len(dl))
    reg_losses.append(ep_reg  / len(dl))

    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1:3d}/{N_EPOCHS}  pred={pred_losses[-1]:.4f}  reg={reg_losses[-1]:.4f}")

epochs = range(1, N_EPOCHS + 1)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, pred_losses)
plt.title("Prediction loss")
plt.xlabel("epoch")
plt.yscale("log")
plt.subplot(1, 2, 2)
plt.plot(epochs, reg_losses, color="C1")
plt.title("SIGReg loss")
plt.xlabel("epoch")
plt.yscale("log")
plt.tight_layout()
plt.savefig("train_losses.png", dpi=120)
plt.show()
