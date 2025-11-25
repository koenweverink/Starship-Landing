# simulate_once.py

import numpy as np

from mars_env import get_mars_params
from dynamics import LanderDynamics
from convex_reference import solve_reference_trajectory
from zem_zev_guidance import ZEMZEVGuidance


def estimate_ballistic_time_to_ground(r0, v0, g_vec, tf_fallback=25.0):
    """
    Solve z(t) = z0 + vz0 t + 0.5 gz t^2 = 0 for t > 0.

    If no positive real root exists (e.g. already below ground or going up),
    fall back to tf_fallback [s].
    """
    z0 = r0[2]
    vz0 = v0[2]
    gz = g_vec[2]

    # If already at/below ground or gravity non-downward, fallback
    if z0 <= 0.0 or gz >= 0.0:
        return tf_fallback

    # Quadratic: 0.5*gz*t^2 + vz0*t + z0 = 0
    a = 0.5 * gz
    b = vz0
    c = z0

    # If a ~ 0, treat as linear
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return tf_fallback  # degenerate, no vertical motion
        t_lin = -c / b
        return t_lin if t_lin > 0 else tf_fallback

    roots = np.roots([a, b, c])

    positive_real_roots = [
        r.real for r in roots
        if abs(r.imag) < 1e-6 and r.real > 0
    ]

    if not positive_real_roots:
        return tf_fallback

    # Use the *later* impact time (most conservative)
    return max(positive_real_roots)


# ----------------------------------------------------------------------
# Engine allocation and thrust direction
# ----------------------------------------------------------------------


def allocate_engines_stage_limited(
    T_des_mag,
    T_engine_max,
    stage_max_engines,
    throttle_min=0.4,
    throttle_max=1.0,
):
    """
    Given desired total thrust magnitude T_des_mag, choose:
        - n_engines in {0,1,2,3}, but never exceeding stage_max_engines
        - T_cmd_mag (actual total thrust)

    Respects per-engine throttle bounds [throttle_min, throttle_max].
    """
    if T_des_mag <= 0.0 or stage_max_engines == 0:
        return 0, 0.0

    # Candidate ranges for each engine count up to stage_max_engines
    choices = []
    for n in (1, 2, 3):
        if n > stage_max_engines:
            continue
        T_min_n = n * throttle_min * T_engine_max
        T_max_n = n * throttle_max * T_engine_max
        choices.append((n, T_min_n, T_max_n))

    if not choices:
        return 0, 0.0

    # 1) Try to find n where T_des lies inside [T_min_n, T_max_n]
    feasible = []
    for n, T_min_n, T_max_n in choices:
        if T_min_n <= T_des_mag <= T_max_n:
            feasible.append(
                (abs(T_des_mag - 0.5 * (T_min_n + T_max_n)), n, T_min_n, T_max_n)
            )

    if feasible:
        feasible.sort()
        _, n, T_min_n, T_max_n = feasible[0]
        T_cmd = np.clip(T_des_mag, T_min_n, T_max_n)
        return n, T_cmd

    # 2) If T_des is too small for even 1 engine at min:
    n, T_min_1, _ = choices[0]
    if T_des_mag < T_min_1:
        # “Always burning once on”: one engine at minimum
        return 1, T_min_1

    # 3) If T_des exceeds max of stage_max_engines, saturate there
    n, _, T_max_n = choices[-1]
    return n, T_max_n


def clamp_thrust_direction(a_cmd, m, T_cmd_mag, max_gimbal_deg=20.0):
    """
    Convert desired control acceleration a_cmd into a thrust vector
    with magnitude T_cmd_mag and direction limited to a cone around +Z.

    T_cmd_mag is already chosen by the engine allocator (0 .. total max).
    """
    up = np.array([0.0, 0.0, 1.0])

    # If engines are off
    if T_cmd_mag <= 0.0 or m <= 0.0:
        return np.zeros(3)

    # Desired thrust direction from acceleration
    T_des = m * a_cmd
    T_norm = np.linalg.norm(T_des)
    if T_norm < 1e-9:
        # No clear direction from guidance; just point up
        direction = up
    else:
        dir_des = T_des / T_norm
        cos_angle = np.clip(np.dot(dir_des, up), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        max_gimbal_rad = np.radians(max_gimbal_deg)
        if angle <= max_gimbal_rad:
            direction = dir_des
        else:
            # Project onto cone boundary
            perp = T_des - np.dot(T_des, up) * up
            if np.linalg.norm(perp) < 1e-9:
                direction = up
            else:
                perp_unit = perp / np.linalg.norm(perp)
                direction = (
                    np.cos(max_gimbal_rad) * up +
                    np.sin(max_gimbal_rad) * perp_unit
                )
                direction /= np.linalg.norm(direction)

    return T_cmd_mag * direction


# ----------------------------------------------------------------------
# Main single-run simulation
# ----------------------------------------------------------------------


def simulate_landing_once(
    r0,
    v0,
    m0,
    tf=None,
    N_ref=60,
    dt_sim=0.05,
    freeze_horizontal=True,
):
    """
    Run one powered landing simulation with convex-guidance + ZEM/ZEV feedback on Mars.

    r0, v0 : initial position and velocity (3,)
    m0     : initial mass [kg]
    tf     : final time [s]; if None, chosen from ballistic TOF and
             refined by the convex solver.
    freeze_horizontal : if True, switch to vertical-only guidance once
                        horizontal speed is small.
    """
    env = get_mars_params()
    g_vec = env["g_vec"]

    r0 = np.array(r0, dtype=float)
    v0 = np.array(v0, dtype=float)
    m0 = float(m0)

    # --- Choose final time tf from ballistic TOF if not given ---
    if tf is None:
        t_ball = estimate_ballistic_time_to_ground(r0, v0, g_vec, tf_fallback=25.0)
        tf_factor = 1.2  # tweak for aggressiveness
        tf = tf_factor * t_ball
        print(f"[simulate] ballistic TOF ~ {t_ball:.2f} s, using tf = {tf:.2f} s")

    # --- Convex reference trajectory (thrust-feasible, monotonic descent, 3→2→1) ---
    ref = None
    try:
        ref = solve_reference_trajectory(
            r0=r0,
            v0=v0,
            g_vec=g_vec,
            tf=tf,
            N=N_ref,
        )
    except Exception as e:
        print(
            f"[simulate] Warning: convex reference solve failed ({e}). "
            f"Continuing with ZEM/ZEV only."
        )

    # --- Dynamics and guidance objects ---
    dyn = LanderDynamics(
        g_vec=g_vec,
        isp=330.0,
        thrust_min=0.0,      # overwritten each step by allocator
        thrust_max=1.0e9,
        dry_mass=1.2e5,
        cd_area=90.0,        # ≈ C_d * A for Starship-like shape
        rho0=0.02,           # surface density on Mars [kg/m^3]
        h_scale=11000.0,     # scale height [m]
    )

    guidance = ZEMZEVGuidance(g_vec=g_vec)

    # --- Engine configuration (Raptor-ish) ---
    T_ENGINE_MAX = 3.0e6      # N, per engine (~3 MN)
    THROTTLE_MIN = 0.40       # 40% of full thrust
    THROTTLE_MAX = 1.0

    # --- Horizontal vs vertical phase logic ---
    horizontal_done = False
    v_switch_h = 2.0          # m/s: below this, optionally freeze sideways accel

    # --- Simulation state ---
    t = 0.0
    r = r0.copy()
    v = v0.copy()
    m = m0

    traj = {"t": [], "r": [], "v": [], "m": [], "engines": []}

    engines_on = False  # hysteresis flag

    # --- Main integration loop ---
    while t < tf and r[2] > 0.0:
        t_go = max(tf - t, 0.1)

        # ZEM/ZEV control acceleration toward terminal state (0,0,0)
        a_cmd = guidance.compute_accel(r, v, t_go)

        # Horizontal speed
        v_h = np.linalg.norm(v[:2])

        # Optional: Phase 1 (3D) → Phase 2 (vertical only)
        if freeze_horizontal:
            if horizontal_done:
                a_cmd = a_cmd.copy()
                a_cmd[0] = 0.0
                a_cmd[1] = 0.0
            else:
                if v_h <= v_switch_h:
                    horizontal_done = True
                    a_cmd = a_cmd.copy()
                    a_cmd[0] = 0.0
                    a_cmd[1] = 0.0

        # --- Engine ON/OFF decision with hysteresis ---
        if not engines_on:
            # Ignite when we are low enough and descending fast enough
            if r[2] < 0.8 * r0[2] and v[2] < -10.0:
                engines_on = True

        # --- Monotonic descent enforcement (A) ---
        # Once engines are on, do NOT allow any upward vertical acceleration
        # that would make v_z positive.
        if engines_on and a_cmd[2] > 0.0 and v[2] > -1.0:
            a_cmd = a_cmd.copy()
            a_cmd[2] = 0.0

        # Desired thrust magnitude from guidance
        T_des = m * a_cmd
        T_des_mag = max(0.0, np.linalg.norm(T_des))

        # --- Stage-based engine cap (B: approx 3→2→1 like convex plan) ---
        # Roughly match the convex scheduling: early: up to 3, middle: 2, late: 1.
        if t_go > 0.6 * tf:
            stage_max_engines = 3
        elif t_go > 0.3 * tf:
            stage_max_engines = 2
        else:
            stage_max_engines = 1

        if engines_on:
            n_engines, T_cmd_mag = allocate_engines_stage_limited(
                T_des_mag,
                T_ENGINE_MAX,
                stage_max_engines=stage_max_engines,
                throttle_min=THROTTLE_MIN,
                throttle_max=THROTTLE_MAX,
            )
        else:
            n_engines, T_cmd_mag = 0, 0.0

        # Configure dynamics limits for this step
        if n_engines == 0:
            dyn.thrust_min = 0.0
            dyn.thrust_max = 0.0
        else:
            dyn.thrust_min = n_engines * THROTTLE_MIN * T_ENGINE_MAX
            dyn.thrust_max = n_engines * THROTTLE_MAX * T_ENGINE_MAX

        # Build thrust vector with proper direction and magnitude
        T_vec = clamp_thrust_direction(a_cmd, m, T_cmd_mag)

        # ----------------------------------------------------
        # HARD "NO-CLIMB" CLAMP:
        # If the next vertical velocity would become positive,
        # scale thrust down so that vz_next ≈ 0 instead.
        # This guarantees we never accelerate away upward.
        # ----------------------------------------------------
        if engines_on and m > 0.0:
            # approximate vertical accel from thrust + gravity (ignore drag)
            a_z = g_vec[2] + T_vec[2] / m
            v_z_next = v[2] + a_z * dt_sim

            if v_z_next > 0.0 and a_z > 0.0:
                # scale thrust so that v_z_next ≈ 0
                scale = -v[2] / (a_z * dt_sim)
                scale = np.clip(scale, 0.0, 1.0)
                T_vec *= scale
        # ----------------------------------------------------

        # Step dynamics
        (r, v, m), a, saturated = dyn.step((r, v, m), T_vec, dt_sim)


        # Store trajectory once per loop
        traj["t"].append(t)
        traj["r"].append(r.copy())
        traj["v"].append(v.copy())
        traj["m"].append(m)
        traj["engines"].append(n_engines)

        t += dt_sim

    # Convert lists to arrays
    for k in traj:
        traj[k] = np.array(traj[k])

    return traj, ref


if __name__ == "__main__":
    # quick manual test
    r0 = np.array([0.0, 0.0, 2000.0])
    v0 = np.array([100.0, 0.0, -50.0])
    m0 = 1.5e5

    traj, ref = simulate_landing_once(r0, v0, m0, freeze_horizontal=True)
    print("Sim finished. Final state:")
    print("r:", traj["r"][-1])
    print("v:", traj["v"][-1])
    print("m:", traj["m"][-1])
