import matplotlib.animation as animation
from utils import *


class DoublePendulum:
    """
    Double pendulum with torque input on the first joint.

    State x = [theta1, theta2, omega1, omega2]
      theta1, theta2: angles from vertical (rad)
      omega1, omega2: angular velocities (rad/s)
    """

    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        b: float = 0.1,
        max_torque: float = 5.0,
        g: float = 9.81,
        dt: float = 0.05,
    ):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.b = b
        self.max_torque = max_torque
        self.g = g
        self.dt = dt

    def _dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        """Continuous-time dynamics dx/dt from Euler-Lagrange equations."""
        th1, th2, w1, w2 = x
        m1, m2, l1, l2, b, g = self.m1, self.m2, self.l1, self.l2, self.b, self.g

        u = np.clip(u, -self.max_torque, self.max_torque)
        d = th1 - th2

        # Mass matrix (symmetric 2x2)
        M11 = (m1 + m2) * l1 ** 2
        M12 = m2 * l1 * l2 * np.cos(d)
        M22 = m2 * l2 ** 2

        # Generalised forces (Coriolis + gravity + friction + input)
        rhs1 = u - b * w1 - m2 * l1 * l2 * w2 ** 2 * np.sin(d) - (m1 + m2) * g * l1 * np.sin(th1)
        rhs2 =    - b * w2 + m2 * l1 * l2 * w1 ** 2 * np.sin(d) - m2 * g * l2 * np.sin(th2)

        # Solve M * [a1, a2]^T = rhs via Cramer's rule
        det = M11 * M22 - M12 ** 2
        a1 = (M22 * rhs1 - M12 * rhs2) / det
        a2 = (M11 * rhs2 - M12 * rhs1) / det

        return np.array([w1, w2, a1, a2])

    def step(self, x: np.ndarray, u: float) -> np.ndarray:
        """
        Advance the system by one time step using RK4 integration.

        Args:
            x: state [theta1, theta2, omega1, omega2]
            u: torque applied to joint 1 (clamped to ±max_torque)

        Returns:
            next state, same shape as x
        """
        dt = self.dt
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + 0.5 * dt * k1, u)
        k3 = self._dynamics(x + 0.5 * dt * k2, u)
        k4 = self._dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def sim(self, x0: np.ndarray, u, t: float) -> np.ndarray:
        """
        Simulate from initial state x0 for t seconds.

        Args:
            x0: initial state [theta1, theta2, omega1, omega2]
            u:  torque — scalar, or array/callable indexed/called by step index
            t:  duration in seconds

        Returns:
            trajectory of shape (n_steps + 1, 4)
        """
        n_steps = int(round(t / self.dt))
        traj = np.empty((n_steps + 1, 4))
        traj[0] = x0
        for i in range(n_steps):
            torque = u(i) if callable(u) else (u[i] if hasattr(u, '__len__') else u)
            traj[i + 1] = self.step(traj[i], torque)
        return traj

    def plot(self, traj: np.ndarray, save_path: str = None):
        """
        Animate the pendulum and plot the 4 states side by side.

        Args:
            traj:      trajectory array of shape (T, 4) from sim()
            save_path: if given, save the animation as a .gif or .mp4
        """
        import matplotlib.animation as animation
        T = len(traj)
        time = np.arange(T) * self.dt

        # Cartesian coordinates of both bobs
        x1 =  self.l1 * np.sin(traj[:, 0])
        y1 = -self.l1 * np.cos(traj[:, 0])
        x2 = x1 + self.l2 * np.sin(traj[:, 1])
        y2 = y1 - self.l2 * np.cos(traj[:, 1])

        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(2, 2, left=0.35, wspace=0.35, hspace=0.5)

        # --- animation panel (left half) ---
        ax_anim = fig.add_axes([0.02, 0.08, 0.28, 0.84])
        lim = (self.l1 + self.l2) * 1.1
        ax_anim.set_xlim(-lim, lim)
        ax_anim.set_ylim(-lim, lim)
        ax_anim.set_aspect("equal")
        ax_anim.set_title("Double pendulum")
        ax_anim.axhline(0, color="gray", lw=0.5, ls="--")
        ax_anim.axvline(0, color="gray", lw=0.5, ls="--")

        trail_len = min(50, T)
        trail, = ax_anim.plot([], [], "b-", lw=0.8, alpha=0.4)
        rod,   = ax_anim.plot([], [], "k-o", lw=2, ms=6)
        time_txt = ax_anim.text(0.02, 0.95, "", transform=ax_anim.transAxes, fontsize=9)

        # --- state panels (right half, 2x2 grid) ---
        labels = [r"$\theta_1$ (rad)", r"$\theta_2$ (rad)",
                  r"$\omega_1$ (rad/s)", r"$\omega_2$ (rad/s)"]
        axes_s = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
        vlines = []
        for ax, label, col in zip(axes_s, labels, traj.T):
            ax.plot(time, col, lw=1)
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("t (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            vl = ax.axvline(0, color="r", lw=0.8, ls="--")
            vlines.append(vl)

        def _init():
            trail.set_data([], [])
            rod.set_data([], [])
            time_txt.set_text("")
            for vl in vlines:
                vl.set_xdata([0])
            return [trail, rod, time_txt] + vlines

        def _update(i):
            trail.set_data(x2[max(0, i - trail_len):i + 1],
                           y2[max(0, i - trail_len):i + 1])
            rod.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
            time_txt.set_text(f"t = {time[i]:.2f} s")
            for vl in vlines:
                vl.set_xdata([time[i]])
            return [trail, rod, time_txt] + vlines

        interval_ms = self.dt * 1000
        ani = animation.FuncAnimation(
            fig, _update, frames=T, init_func=_init,
            interval=interval_ms, blit=True
        )

        if save_path is not None:
            ani.save(save_path, fps=int(1 / self.dt))

        plt.show()
        return ani
