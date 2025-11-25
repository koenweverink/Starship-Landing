import numpy as np
import matplotlib.pyplot as plt

from simulate_once import simulate_landing_once


def sample_initial_condition(rng,
                             base_r0,
                             base_v0,
                             base_m0,
                             sigma_r=(5.0, 5.0, 10.0),
                             sigma_v=(2.0, 2.0, 2.0),
                             mass_rel_sigma=0.02):
    """
    Draw one randomized initial condition around the nominal state.

    sigma_r: std dev [m] for (x, y, z)
    sigma_v: std dev [m/s] for (vx, vy, vz)
    mass_rel_sigma: relative std dev for mass (e.g. 0.05 = ±5%)
    """
    r0 = base_r0 + rng.normal(0.0, sigma_r)
    v0 = base_v0 + rng.normal(0.0, sigma_v)
    m0 = base_m0 * (1.0 + rng.normal(0.0, mass_rel_sigma))
    return r0, v0, m0


def run_monte_carlo(num_runs=50,
                    verbose=True,
                    seed=42):
    """
    Run many randomized landings and return a list of results dicts.
    Each result has: final_r, final_v, traj.
    """
    results = []
    rng = np.random.default_rng(seed)

    # Nominal initial conditions
    base_r0 = np.array([0.0, 0.0, 2000.0])   # 2 km above surface
    base_v0 = np.array([100.0, 0.0, -50.0])  # lateral + vertical speed
    base_m0 = 1.5e5                          # kg

    for i in range(num_runs):
        # --- Sample uncertainties around the nominal ---
        r0, v0, m0 = sample_initial_condition(
            rng,
            base_r0,
            base_v0,
            base_m0,
            sigma_r=(20.0, 20.0, 50.0),
            sigma_v=(5.0, 5.0, 5.0),
            mass_rel_sigma=0.05,
        )

        # --- Run one simulation ---
        traj, ref = simulate_landing_once(r0, v0, m0, freeze_horizontal=False)

        final_r = traj["r"][-1]
        final_v = traj["v"][-1]

        pos_err = np.linalg.norm(final_r)
        vel_err = np.linalg.norm(final_v)

        if verbose:
            print(
                f"Run {i+1:3d}: "
                f"pos error = {pos_err:8.2f} m, "
                f"vel error = {vel_err:6.2f} m/s, "
                f"final_alt = {final_r[2]:7.2f} m"
            )

        results.append(
            {
                "final_r": final_r,
                "final_v": final_v,
                "traj": traj,
            }
        )

    return results


def summarize_results(results,
                      pos_tol=10.0,
                      vel_tol=2.0,
                      make_plots=True):
    """
    Compute statistics on Monte Carlo results, print them,
    and optionally plot histograms of position & velocity error.
    """
    final_r = np.array([r["final_r"] for r in results])
    final_v = np.array([r["final_v"] for r in results])

    pos_err = np.linalg.norm(final_r, axis=1)
    vel_err = np.linalg.norm(final_v, axis=1)

    # --- Basic stats ---
    print("\n=== Monte Carlo Summary ===")
    print(f"Number of runs: {len(results)}")

    print("\nPosition error (m):")
    print(f"  min  = {pos_err.min():8.3f}")
    print(f"  mean = {pos_err.mean():8.3f}")
    print(f"  max  = {pos_err.max():8.3f}")
    print(f"  std  = {pos_err.std():8.3f}")

    print("\nVelocity error (m/s):")
    print(f"  min  = {vel_err.min():8.3f}")
    print(f"  mean = {vel_err.mean():8.3f}")
    print(f"  max  = {vel_err.max():8.3f}")
    print(f"  std  = {vel_err.std():8.3f}")

    # --- Success criteria ---
    success = (pos_err <= pos_tol) & (vel_err <= vel_tol)
    success_rate = 100.0 * success.sum() / len(success)

    print("\nSuccess criteria:")
    print(f"  pos_err <= {pos_tol} m AND vel_err <= {vel_tol} m/s")
    print(
        f"  successes: {success.sum()} / {len(success)} "
        f"({success_rate:5.1f} %)"
    )

    # --- Optional plots ---
    if make_plots:
        plt.figure()
        plt.hist(pos_err, bins=20)
        plt.xlabel("Landing position error [m]")
        plt.ylabel("Count")
        plt.title("Monte Carlo: Position error")

        plt.figure()
        plt.hist(vel_err, bins=20)
        plt.xlabel("Landing velocity error [m/s]")
        plt.ylabel("Count")
        plt.title("Monte Carlo: Velocity error")

        plt.show()


if __name__ == "__main__":
    # Run more sims if you want – 100+ is nice if it’s fast enough
    results = run_monte_carlo(num_runs=50, verbose=True, seed=42)
    summarize_results(results, pos_tol=10.0, vel_tol=2.0, make_plots=True)
