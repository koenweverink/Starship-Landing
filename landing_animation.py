"""Simple Matplotlib animation of a simulated Starship landing."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from simulate_once import simulate_landing_once


def generate_nominal_trajectory():
    """Run a single landing simulation and return the trajectory dictionary."""
    r0 = np.array([0.0, 0.0, 3000.0])
    v0 = np.array([90.0, 0.0, -60.0])
    m0 = 150_000.0

    traj, _ = simulate_landing_once(r0, v0, m0)
    return traj


def build_animation(traj, save_path=None, interval_ms=40, max_frames=400):
    """Create a Matplotlib animation showing altitude over time and a side view.

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

    fig, (ax_time, ax_side) = plt.subplots(1, 2, figsize=(12, 6))

    # Altitude vs time
    ax_time.plot(t, alt, color="#2b6cb0", linewidth=2)
    time_marker, = ax_time.plot([], [], "o", color="#e53e3e", markersize=8)
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel("Altitude [m]")
    ax_time.set_title("Altitude profile")
    ax_time.grid(True, linestyle="--", alpha=0.4)
    ax_time.set_xlim(t[0], t[-1])
    alt_margin = max(alt) * 0.05
    ax_time.set_ylim(0.0, max(alt) + alt_margin)

    # Side view: x vs altitude
    ax_side.plot(x, alt, color="#38a169", linewidth=2)
    ship_marker, = ax_side.plot([], [], marker="^", color="#2f855a", markersize=10)
    ax_side.set_xlabel("Downrange X [m]")
    ax_side.set_ylabel("Altitude [m]")
    ax_side.set_title("Side view of trajectory")
    ax_side.grid(True, linestyle="--", alpha=0.4)
    x_span = max(abs(x)) if np.any(x) else 1.0
    ax_side.set_xlim(-x_span * 1.05, x_span * 1.05)
    ax_side.set_ylim(0.0, max(alt) + alt_margin)

    def init():
        time_marker.set_data([], [])
        ship_marker.set_data([], [])
        return time_marker, ship_marker

    frame_step = max(1, len(t) // max_frames)
    frame_indices = list(range(0, len(t), frame_step))
    if frame_indices[-1] != len(t) - 1:
        frame_indices.append(len(t) - 1)

    def update(frame_idx):
        # Matplotlib expects sequences, so wrap scalars in small lists
        time_marker.set_data([t[frame_idx]], [alt[frame_idx]])
        ship_marker.set_data([x[frame_idx]], [alt[frame_idx]])
        return time_marker, ship_marker

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
