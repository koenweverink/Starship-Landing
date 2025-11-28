# simulate_once.py
import numpy as np
from mars_env import get_mars_params
from dynamics import LanderDynamics
from zem_zev_guidance import ZEMZEVGuidance  # noqa: F401  (not used yet)


def simulate_landing_once(
    r0,
    v0,
    m0,
    dt_sim=0.05,
    freeze_horizontal=True,
):
    """
    Simple powered-descent simulation with a 1D braking law:

    * Ignition decided from ballistic time-to-ground vs braking capability.
    * Vertical acceleration chosen so we can stop by the surface.
    * Below ~3 m we cut for a soft touch.
    * Horizontal velocity is killed with side-thrust while altitude is > ~20 m.
    """

    # ------------------------------------------------------------------
    # Environment and vehicle parameters
    # ------------------------------------------------------------------
    env = get_mars_params()
    g_vec = env["g_vec"]
    g_mag = -g_vec[2]

    r0 = np.array(r0, dtype=float)
    v0 = np.array(v0, dtype=float)
    m0 = float(m0)

    # Treat this as a single “effective engine” vehicle
    T_ENGINE_MAX = 3_000_000.0        # [N] effective max thrust
    THROTTLE_MIN = 0.0                # allow full shut-down
    THROTTLE_MAX = 1.0

    dyn = LanderDynamics(
        g_vec=g_vec,
        isp=380.0,
        thrust_min=THROTTLE_MIN * T_ENGINE_MAX,
        thrust_max=THROTTLE_MAX * T_ENGINE_MAX,
        dry_mass=85_000.0,
        cd_area=120.0,
        rho0=0.020,
        h_scale=11000.0,
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def ballistic_ttg(pos, vel):
        """Ballistic time-to-ground under gravity only (no drag)."""
        z, vz = pos[2], vel[2]
        if z <= 0.0:
            return 0.0
        # z(t) = z + vz t + 0.5 g t^2 = 0
        a = 0.5 * g_vec[2]
        b = vz
        c = z
        disc = b * b - 4 * a * c
        if disc < 0 or abs(a) < 1e-9:
            if vz >= 0:
                return 1e9
            return -z / vz
        t1 = (-b - np.sqrt(disc)) / (2 * a)
        t2 = (-b + np.sqrt(disc)) / (2 * a)
        ts = [t for t in (t1, t2) if t > 0]
        return min(ts) if ts else 1e9

    # ------------------------------------------------------------------
    # Simulation state
    # ------------------------------------------------------------------
    t = 0.0
    r = r0.copy()
    v = v0.copy()
    m = m0

    engines_on = False

    traj = {
        "t": [],
        "r": [],
        "v": [],
        "m": [],
        "engines": [],
    }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while r[2] > 0.05 and t < 300.0:
        alt = r[2]
        v_h = v[:2]
        v_h_mag = np.linalg.norm(v_h)

        # ----------------------------------------------------------
        # Ignition logic: compare ballistic TTG to braking time
        # ----------------------------------------------------------
        if not engines_on:
            t_ball = ballistic_ttg(r, v)

            # With max thrust we can get net upward accel:
            # a_net_max = T_max/m - g
            a_net_max = T_ENGINE_MAX / max(m, 1.0) - g_mag
            if a_net_max <= 0.0:
                a_net_max = 1.0

            # Time needed to cancel vertical speed at max accel
            if v[2] < 0.0:
                t_stop = -v[2] / a_net_max
            else:
                t_stop = 0.0

            # Ignite when ballistic TTG is only slightly larger than t_stop
            if t_ball <= 1.2 * t_stop + 2.0:
                engines_on = True
                print(f"[t={t:.1f}s] ENGINES IGNITED @ {alt:.0f}m")

        # ----------------------------------------------------------
        # Thrust vector computation
        # ----------------------------------------------------------
        if not engines_on:
            T_vec = np.zeros(3)
            n_eng = 0
        else:
            n_eng = 1  # we model it as one effective engine

            # Desired *net upward* vertical accel so we stop at z=0
            v_z = v[2]
            safe_alt = max(alt, 1.0)

            # Simple braking law: a ≈ v^2 / (2 h)
            if v_z < 0.0:  # descending
                a_z_net = (v_z * v_z) / (2.0 * safe_alt)
            else:
                # Already going up or very slow: just counter gravity gently
                a_z_net = 0.5

            # Limit net accel depending on altitude
            if alt > 80.0:
                a_z_net = np.clip(a_z_net, 0.5, 8.0)    # strong braking allowed
            else:
                a_z_net = np.clip(a_z_net, 0.5, 5.0)    # softer near ground

            # Convert desired net upward accel to thrust magnitude:
            # a_net = -g + T_z / m  =>  T_z = (a_net + g) * m
            T_z_required = (a_z_net + g_mag) * m
            T_z_required = np.clip(T_z_required, 0.0, T_ENGINE_MAX)

            # Start with purely vertical thrust
            T_vec = np.array([0.0, 0.0, T_z_required])

            # ---------- STRONGER HORIZONTAL BRAKING ----------
            # Use side thrust to kill horizontal velocity while high.
            if freeze_horizontal and alt > 20.0 and v_h_mag > 0.2:
                v_h_unit = v_h / v_h_mag
                # Stronger lateral decel (was 2.0, now 5.0 m/s^2)
                a_side = min(5.0, v_h_mag)   # don't over-brake in one step
                T_side = a_side * m
                T_vec[0] = -v_h_unit[0] * T_side
                T_vec[1] = -v_h_unit[1] * T_side
            # -------------------------------------------------

            # Clamp total thrust to engine limits
            T_mag = np.linalg.norm(T_vec)
            T_min = THROTTLE_MIN * T_ENGINE_MAX
            T_max = THROTTLE_MAX * T_ENGINE_MAX

            if T_mag > 1e-6:
                scale = 1.0
                if T_mag < T_min:
                    scale = T_min / T_mag
                if T_mag > T_max:
                    scale = T_max / T_mag
                T_vec *= scale

            # Shut down for a gentle touchdown
            if alt < 3.0 and abs(v_z) < 1.0 and v_h_mag < 0.8:
                print(
                    f"[t={t:.1f}s] TOUCHDOWN! "
                    f"v_vert = {v_z:.3f} m/s | v_h = {v_h_mag:.3f} m/s"
                )
                T_vec[:] = 0.0
                engines_on = False
                n_eng = 0

        # Update dynamic thrust limits (for internal clamping)
        dyn.thrust_min = THROTTLE_MIN * T_ENGINE_MAX
        dyn.thrust_max = THROTTLE_MAX * T_ENGINE_MAX

        # ----------------------------------------------------------
        # Integrate one step
        # ----------------------------------------------------------
        (r, v, m), _, _ = dyn.step((r, v, m), T_vec, dt_sim)

        traj["t"].append(t)
        traj["r"].append(r.copy())
        traj["v"].append(v.copy())
        traj["m"].append(m)
        traj["engines"].append(n_eng if engines_on else 0)

        t += dt_sim

    for k in traj:
        traj[k] = np.array(traj[k])

    return traj


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example initial conditions
    r0 = np.array([0.0, 0.0, 2000.0])      # 2 km altitude
    v0 = np.array([90.0, 0.0, -60.0])      # 90 m/s sideways, 60 m/s down
    m0 = 150_000.0

    traj = simulate_landing_once(r0, v0, m0)

    import matplotlib.pyplot as plt

    t = traj["t"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t, traj["r"][:, 2])
    ax[0, 0].set_title("Altitude [m]")
    ax[0, 1].plot(t, traj["v"][:, 2])
    ax[0, 1].set_title("Vertical Velocity [m/s]")
    ax[1, 0].plot(t, np.linalg.norm(traj["v"][:, :2], axis=1))
    ax[1, 0].set_title("Horizontal Speed [m/s]")
    ax[1, 1].step(t, traj["engines"], where="post")
    ax[1, 1].set_title("Engines On")
    plt.tight_layout()
    plt.show()

    print("TOUCHDOWN vertical   :", traj["v"][-1, 2])
    print("TOUCHDOWN horizontal :", np.linalg.norm(traj["v"][-1, :2]))
