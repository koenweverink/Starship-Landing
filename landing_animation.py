"""Simple Matplotlib animation of a simulated Starship landing."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import patches, transforms

from dynamics import LanderDynamics
from simulate_once import simulate_landing_once


def generate_nominal_trajectory():
    """Run a single landing simulation and return the trajectory dictionary."""
    r0 = np.array([0.0, 0.0, 3000.0])
    v0 = np.array([90.0, 0.0, -60.0])
    m0 = 150_000.0

    traj, _ = simulate_landing_once(r0, v0, m0)
    return traj


def build_animation(traj, save_path=None, interval_ms=40, max_frames=400):
    """Create a single side-view animation with body orientation shown.

    Parameters
    ----------
    traj : dict
        Trajectory dictionary returned by ``simulate_landing_once``.
    save_path : str or None
        If provided, the animation is saved to this path (GIF).
    interval_ms : int
        Delay between frames in milliseconds.
    """

    t = traj["t"]
    r = traj["r"]
    x = r[:, 0]
    alt = r[:, 2]

    # Extract body z-axes to compute tilt relative to vertical (x-z plane)
    z_axes = np.array([LanderDynamics.quat_to_dcm(q)[:, 2] for q in traj["q"]])
    tilt_deg = np.degrees(np.arctan2(z_axes[:, 0], z_axes[:, 2]))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(x, alt, color="#2b6cb0", linewidth=2, alpha=0.75)
    ship_marker, = ax.plot([], [], "o", color="#e53e3e", markersize=8)
    ax.set_xlabel("Downrange X [m]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Starship landing profile")
    ax.grid(True, linestyle="--", alpha=0.4)
    x_span = max(abs(x)) if np.any(x) else 1.0
    alt_margin = max(alt) * 0.08
    ax.set_xlim(-x_span * 1.05, x_span * 1.05)
    ax.set_ylim(0.0, max(alt) + alt_margin)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")

    # Rectangle representing the vehicle
    ship_height = max(80.0, 0.07 * (max(alt) + alt_margin))
    ship_width = ship_height * 0.18
    ship_patch = patches.Rectangle(
        (-ship_width / 2.0, -ship_height / 2.0),
        ship_width,
        ship_height,
        facecolor="#48bb78",
        edgecolor="#276749",
        alpha=0.8,
    )
    ax.add_patch(ship_patch)
    angle_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    def init():
        ship_marker.set_data([], [])
        ship_patch.set_transform(ax.transData)
        angle_text.set_text("")
        return ship_marker, ship_patch, angle_text

    frame_step = max(1, len(t) // max_frames)
    frame_indices = list(range(0, len(t), frame_step))
    if frame_indices[-1] != len(t) - 1:
        frame_indices.append(len(t) - 1)

    def update(frame_idx):
        xi, zi = x[frame_idx], alt[frame_idx]
        angle = tilt_deg[frame_idx]

        ship_marker.set_data([xi], [zi])

        transform = transforms.Affine2D().rotate_deg(angle).translate(xi, zi)
        ship_patch.set_transform(transform + ax.transData)

        angle_text.set_text(f"Angle: {angle:+.1f}°")
        return ship_marker, ship_patch, angle_text

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frame_indices,
        interval=interval_ms,
        blit=True,
    )

    if save_path:
        writer = animation.PillowWriter(fps=max(int(1000 / interval_ms), 1))
        anim.save(save_path, writer=writer)
        print(f"Saved landing animation to {save_path}")

    return anim, fig


def main():
    parser = argparse.ArgumentParser(description="Animate a simulated Starship landing.")
    parser.add_argument(
        "--save",
        default="landing.gif",
        help="File path to save the GIF (set to '' to skip saving)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=40,
        help="Frame delay in milliseconds (controls animation speed)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the animation window when not saving",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Maximum number of frames to render (subsamples if needed)",
    )
    args = parser.parse_args()

    save_path = args.save if args.save else None
    traj = generate_nominal_trajectory()
    anim, fig = build_animation(
        traj,
        save_path=save_path,
        interval_ms=args.interval,
        max_frames=args.max_frames,
    )

    if save_path is None and not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
