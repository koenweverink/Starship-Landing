import numpy as np
import matplotlib.pyplot as plt

# Import your simulator (make sure this matches your filename)
from simulate_once import simulate_landing_once


def run_monte_carlo(
    N_runs=200,
    base_alt=2000.0,
    base_m0=150_000.0,
    v_h_range=(70.0, 110.0),     # initial horizontal speed [m/s]
    v_z_range=(-80.0, -40.0),    # initial vertical speed [m/s] (negative = down)
    mass_frac_range=(0.9, 1.1),  # random mass factor
    dt_sim=0.05,
):
    rng = np.random.default_rng(42)

    # Storage
    v_vert_touch = []
    v_horiz_touch = []
    fuel_used = []
    peak_g = []
    max_thrust = []
    m0_list = []
    safe_flags = []
    min_altitudes = []

    for i in range(N_runs):
        # --- Sample random initial conditions ---
        # Random horizontal direction and magnitude
        v_h_mag = rng.uniform(*v_h_range)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        vx = v_h_mag * np.cos(angle)
        vy = v_h_mag * np.sin(angle)

        # Vertical speed
        vz = rng.uniform(*v_z_range)  # negative down

        # Initial state
        r0 = np.array([0.0, 0.0, base_alt])
        v0 = np.array([vx, vy, vz])

        # Random mass factor
        m_factor = rng.uniform(*mass_frac_range)
        m0 = base_m0 * m_factor

        # --- Run one landing ---
        try:
            traj, summary = simulate_landing_once(r0, v0, m0, dt_sim=dt_sim)
        except Exception as e:
            print(f"[RUN {i}] Simulation error: {e}")
            # Mark as unsafe and continue
            v_vert_touch.append(np.nan)
            v_horiz_touch.append(np.nan)
            fuel_used.append(np.nan)
            peak_g.append(np.nan)
            max_thrust.append(np.nan)
            m0_list.append(m0)
            safe_flags.append(False)
            min_altitudes.append(np.nan)
            continue

        v_vert = summary["v_touch_vertical"]
        v_h = summary["v_touch_horizontal"]
        fuel = summary["fuel_used_total"]
        gmax = summary["peak_g_load"]
        Tmax = summary["max_thrust"]
        min_alt = summary["min_altitude"]

        v_vert_touch.append(v_vert)
        v_horiz_touch.append(v_h)
        fuel_used.append(fuel)
        peak_g.append(gmax)
        max_thrust.append(Tmax)
        m0_list.append(summary["m0"])
        min_altitudes.append(min_alt)

        # --- Define "safe landing" criteria ---
        #  - actually reached the ground (min_alt ~ 0)
        #  - vertical speed small
        #  - horizontal speed small
        #  - g-load not crazy
        landed = min_alt < 0.1
        soft_vertical = abs(v_vert) < 2.0     # m/s
        soft_horizontal = v_h < 5.0          # m/s
        g_ok = gmax < 4.0                    # < 4 g

        safe = landed and soft_vertical and soft_horizontal and g_ok
        safe_flags.append(safe)

        print(
            f"[RUN {i:03d}] "
            f"m0={m0/1000:.1f} t, "
            f"v0=({vx:.1f},{vy:.1f},{vz:.1f}) m/s -> "
            f"touch: v_z={v_vert:.2f} m/s, v_h={v_h:.2f} m/s, "
            f"fuel={fuel/1000:.2f} t, g_max={gmax:.2f}, SAFE={safe}"
        )

    # Convert to arrays for easier stats
    v_vert_touch = np.array(v_vert_touch)
    v_horiz_touch = np.array(v_horiz_touch)
    fuel_used = np.array(fuel_used)
    peak_g = np.array(peak_g)
    max_thrust = np.array(max_thrust)
    m0_list = np.array(m0_list)
    safe_flags = np.array(safe_flags, dtype=bool)
    min_altitudes = np.array(min_altitudes)

    # --- Summary statistics ---
    n_safe = np.sum(safe_flags)
    n_total = len(safe_flags)
    print("\n========== MONTE CARLO SUMMARY ==========")
    print(f"Total runs          : {n_total}")
    print(f"Safe landings       : {n_safe} ({100.0*n_safe/n_total:.1f}%)")

    if n_safe > 0:
        print("\nAmong SAFE landings:")
        print(f"  <v_z>   [m/s]     : {np.mean(v_vert_touch[safe_flags]):.3f}")
        print(f"  <|v_h|> [m/s]     : {np.mean(v_horiz_touch[safe_flags]):.3f}")
        print(f"  Fuel used [t]     : mean={np.mean(fuel_used[safe_flags])/1000:.3f}, "
              f"min={np.min(fuel_used[safe_flags])/1000:.3f}, "
              f"max={np.max(fuel_used[safe_flags])/1000:.3f}")
        print(f"  Peak g-load       : mean={np.mean(peak_g[safe_flags]):.3f}, "
              f"max={np.max(peak_g[safe_flags]):.3f}")
    else:
        print("No safe landings – time to tune guidance 😅")

    # --- Basic plots ---
    # Touchdown vertical & horizontal speed
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].hist(v_vert_touch[~np.isnan(v_vert_touch)], bins=30)
    ax[0, 0].set_title("Touchdown Vertical Speed [m/s]")
    ax[0, 0].axvline(-2.0, color="r", linestyle="--", label="Soft limit")
    ax[0, 0].axvline(2.0, color="r", linestyle="--")
    ax[0, 0].legend()

    ax[0, 1].hist(v_horiz_touch[~np.isnan(v_horiz_touch)], bins=30)
    ax[0, 1].set_title("Touchdown Horizontal Speed [m/s]")
    ax[0, 1].axvline(5.0, color="r", linestyle="--", label="Soft limit")
    ax[0, 1].legend()

    ax[1, 0].hist((fuel_used[~np.isnan(fuel_used)] / 1000.0), bins=30)
    ax[1, 0].set_title("Fuel Used [t]")

    ax[1, 1].hist(peak_g[~np.isnan(peak_g)], bins=30)
    ax[1, 1].set_title("Peak g-load [g]")

    plt.tight_layout()
    plt.show()

    return {
        "v_vert_touch": v_vert_touch,
        "v_horiz_touch": v_horiz_touch,
        "fuel_used": fuel_used,
        "peak_g": peak_g,
        "max_thrust": max_thrust,
        "m0": m0_list,
        "safe": safe_flags,
        "min_alt": min_altitudes,
    }


if __name__ == "__main__":
    results = run_monte_carlo(
        N_runs=200,
        base_alt=2000.0,
        base_m0=150_000.0,
        v_h_range=(70.0, 110.0),
        v_z_range=(-80.0, -40.0),
        mass_frac_range=(0.9, 1.1),
        dt_sim=0.05,
    )
