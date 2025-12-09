# debug_single_run.py
import numpy as np
import matplotlib.pyplot as plt
from simulate_once import simulate_landing_once

if __name__ == "__main__":
    r0 = np.array([0.0, 0.0, 3000.0])
    v0 = np.array([100.0, 0.0, -50.0])
    m0 = 1.5e5

    traj, ref = simulate_landing_once(r0, v0, m0, freeze_horizontal=True)

    t = traj["t"]
    r = traj["r"]
    v = traj["v"]

    alt = r[:, 2]
    v_vert = v[:, 2]

    v_horiz = np.linalg.norm(v[:, :2], axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_alt, ax_vvert, ax_vhoriz, ax_eng = axes.ravel()

    ax_alt.plot(t, alt)
    ax_alt.set_xlabel("Time [s]")
    ax_alt.set_ylabel("Altitude z [m]")
    ax_alt.set_title("Altitude vs Time")

    ax_vvert.plot(t, v_vert)
    ax_vvert.set_xlabel("Time [s]")
    ax_vvert.set_ylabel("Vertical velocity [m/s]")
    ax_vvert.set_title("Vertical velocity vs Time")

    ax_vhoriz.plot(t, v_horiz)
    ax_vhoriz.set_xlabel("Time [s]")
    ax_vhoriz.set_ylabel("Horizontal speed [m/s]")
    ax_vhoriz.set_title("Horizontal Speed vs Time")

    ax_eng.step(t, traj["engines"], where="post")
    ax_eng.set_xlabel("Time [s]")
    ax_eng.set_ylabel("Number of engines")
    ax_eng.set_title("Engine staging")

    fig.tight_layout()
    plt.show()


    print("Final altitude:", alt[-1])
    print("Final horizontal speed:", v_horiz[-1])
    print("Final vertical speed:", v[-1, 2])
    print("Final total speed:", np.linalg.norm(v[-1]))

