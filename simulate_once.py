# simulate_once.py

import numpy as np

from mars_env import get_mars_params
from dynamics import LanderDynamics
from convex_reference import solve_reference_trajectory
from zem_zev_guidance import ZEMZEVGuidance


# ------------------------------------------------------------------
# 1. HELPER FUNCTIONS (Restored/Preserved)
# ------------------------------------------------------------------

def estimate_ballistic_time_to_ground(r0, v0, g_vec, tf_fallback=25.0):
    """
    Solve z(t) = z0 + vz0 t + 0.5 gz t^2 = 0 for t > 0.
    """
    z0 = float(r0[2])
    vz0 = float(v0[2])
    gz = float(g_vec[2])

    if z0 <= 0.0 or gz >= 0.0:
        return tf_fallback

    a = 0.5 * gz
    b = vz0
    c = z0

    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return tf_fallback
        t_lin = -c / b
        return t_lin if t_lin > 0 else tf_fallback

    roots = np.roots([a, b, c])
    positive_real_roots = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]

    if not positive_real_roots:
        return tf_fallback

    return max(positive_real_roots)


def allocate_engines(T_des_mag, T_engine_max, max_engines,
                     throttle_min=0.4, throttle_max=1.0):
    """
    Choose (n_engines, T_cmd_mag) based on required thrust and max stage.
    """
    if T_des_mag <= 0.0 or max_engines == 0:
        return 0, 0.0

    choices = []
    # Only check up to the current max_engines allowed by stage
    for n in range(1, max_engines + 1):
        T_min_n = n * throttle_min * T_engine_max
        T_max_n = n * throttle_max * T_engine_max
        choices.append((n, T_min_n, T_max_n))

    if not choices:
        return 0, 0.0

    # 1. Look for a configuration that can exactly match T_des_mag
    feasible = []
    for n, T_min_n, T_max_n in choices:
        if T_min_n <= T_des_mag <= T_max_n:
            feasible.append((abs(T_des_mag - 0.5 * (T_min_n + T_max_n)), n, T_min_n, T_max_n))

    if feasible:
        feasible.sort()
        _, n, T_min_n, T_max_n = feasible[0]
        T_cmd = np.clip(T_des_mag, T_min_n, T_max_n)
        return n, T_cmd

    # 2. If no exact match, clamp to nearest available
    n_small, T_min_small, _ = choices[0]
    if T_des_mag < T_min_small:
        return n_small, T_min_small # Request is too low, forced to minimum thrust

    n_large, _, T_max_last = choices[-1]
    return n_large, T_max_last # Request is too high, clamped to max thrust


def allocate_engines_stage_limited(T_des_mag, T_engine_max, stage_max_engines, 
                                   throttle_min, throttle_max):
    return allocate_engines(T_des_mag, T_engine_max, stage_max_engines, 
                            throttle_min, throttle_max)

def clamp_direction_to_cone(vec, axis, max_angle_deg):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    norm = np.linalg.norm(vec)
    if norm < 1e-9: return np.zeros(3)
    v = vec / norm
    cos_ang = np.clip(np.dot(v, axis), -1.0, 1.0)
    ang = np.arccos(cos_ang)
    max_ang = np.radians(max_angle_deg)
    if ang <= max_ang: return vec
    perp = v - np.dot(v, axis) * axis
    if np.linalg.norm(perp) < 1e-9: return norm * axis
    perp /= np.linalg.norm(perp)
    v_clamped = np.cos(max_ang) * axis + np.sin(max_ang) * perp
    v_clamped /= np.linalg.norm(v_clamped)
    return norm * v_clamped


def clamp_thrust_direction(a_cmd, m, T_cmd_mag, gimbal_limit=20.0):
    if np.linalg.norm(a_cmd) < 1e-6:
        return np.array([0.0, 0.0, T_cmd_mag]) 
    T_vec_ideal = (a_cmd / np.linalg.norm(a_cmd)) * T_cmd_mag
    T_vec_clamped = clamp_direction_to_cone(T_vec_ideal, np.array([0.0, 0.0, 1.0]), gimbal_limit)
    if np.linalg.norm(T_vec_clamped) > 1e-9:
        T_vec_clamped = (T_vec_clamped / np.linalg.norm(T_vec_clamped)) * T_cmd_mag
    return T_vec_clamped


# ------------------------------------------------------------------
# 2. MAIN SIMULATION (With Prediction-Based Staging)
# ------------------------------------------------------------------

def simulate_landing_once(
    r0,
    v0,
    m0,
    tf=None, 
    N_ref=60,
    dt_sim=0.05,
    freeze_horizontal=True,
):
    env = get_mars_params()
    g_vec = env["g_vec"]
    g_mag = np.linalg.norm(g_vec)

    r0 = np.array(r0, dtype=float)
    v0 = np.array(v0, dtype=float)
    m0 = float(m0)

    # --- 1. Convex Reference (unchanged) ---
    if tf is None:
        z0 = r0[2]
        vz0 = v0[2]
        if vz0 < 0:
            tf = 1.2 * abs(z0 / vz0) 
        else:
            tf = 30.0

    print(f"[simulate] Initializing guidance with rough tf = {tf:.2f} s")

    ref = None
    try:
        ref = solve_reference_trajectory(r0, v0, g_vec, tf, N=N_ref)
    except Exception as e:
        print(f"[simulate] Convex solver warning (non-critical): {e}")

    # --- 2. Setup Dynamics & Guidance (unchanged) ---
    dyn = LanderDynamics(
        g_vec=g_vec,
        isp=330.0,
        thrust_min=0.0, 
        thrust_max=1.0e9, 
        dry_mass=1.2e5,
        cd_area=90.0,
        rho0=0.02,
        h_scale=11000.0,
    )

    guidance = ZEMZEVGuidance(g_vec=g_vec)

    # Engine Config
    T_ENGINE_MAX = 3.0e6
    THROTTLE_MIN = 0.40
    THROTTLE_MAX = 1.0

    # Simulation State
    t = 0.0
    r = r0.copy()
    v = v0.copy()
    m = m0
    
    traj = {"t": [], "r": [], "v": [], "m": [], "engines": []}
    
    # CONTROL STATES
    engines_on = False
    horizontal_done = False
    # Explicit 0-3-2-1 staging (engine count is sticky within each stage)
    current_stage_max = 0  # 0 = free fall, then 3->2->1
    
    # --- 3. Main Adaptive Loop ---
    max_duration = 100.0
    t_last_stage_change = 0.0

    while r[2] > 0.1 and t < max_duration:
        
        # A. Altitude Definition (FIX: alt was undefined)
        alt = r[2]
        
        # B. Guidance Command (The Prediction)
        t_go = guidance.compute_tgo(r, v, min_tgo=2.0)
        a_cmd = guidance.compute_accel(r, v, t_go)

        # Keep an updated ballistic estimate every cycle (used by staging and braking)
        t_ballistic = estimate_ballistic_time_to_ground(r, v, g_vec)
        
        # C. Required Thrust (The Prediction in physical units)
        # T_thrust = m*(a_cmd + g)
        T_req_vec = m * a_cmd - m * g_vec 
        T_req_mag = np.linalg.norm(T_req_vec)
        
        # D. Define Thrust Thresholds (Dynamic based on mass)
        T_max_2_engines = 2 * THROTTLE_MAX * T_ENGINE_MAX
        T_max_1_engine = 1 * THROTTLE_MAX * T_ENGINE_MAX
        T_min_1_engine = 1 * THROTTLE_MIN * T_ENGINE_MAX
        
        # E. Engine Staging Logic (Explicit 0-3-2-1 profile)

        # Step 1: Initial Ignition (Free Fall -> Stage 3, or 2 if 3 is too much)
        if not engines_on:
            # Only consider ignition when we're falling and guidance calls for real thrust
            a_net_max3 = (3 * THROTTLE_MAX * T_ENGINE_MAX) / m + g_vec[2]
            burn_margin = 1.1
            t_brake_max3 = (-v[2] / a_net_max3) if a_net_max3 > 1e-6 else np.inf

            # Ignite when we're close enough that a three-engine burn could actually be needed
            if (
                T_req_mag > T_min_1_engine * 0.9
                and v[2] < -5.0
                and t_ballistic <= burn_margin * t_brake_max3
            ):
                preferred_stage = 3
                T_min_3_engine = 3 * THROTTLE_MIN * T_ENGINE_MAX

                # If three engines at minimum thrust exceed the required thrust, start with two
                if T_req_mag < 0.95 * T_min_3_engine:
                    preferred_stage = 2

                engines_on = True
                current_stage_max = preferred_stage
                t_last_stage_change = t
                print(
                    f"Ignition (Stage {preferred_stage}) at t={t:.2f}, Alt={alt:.1f}m, "
                    f"T_req={T_req_mag/1e3:.1f}kN"
                )

        # Step 2: Stage Down (3->2->1 only, never back up)
        if engines_on:
            # Stage-down decisions rely on whether the lower stage can still null velocity
            # before hitting the ground (with a safety margin) while avoiding thrashing.
            min_dwell = 1.5  # seconds to stay in a stage before considering another drop
            burn_window = 2.5  # ensure there's enough time to use the next stage

            def stage_safe_to_drop(n_engines_target):
                T_max_target = n_engines_target * THROTTLE_MAX * T_ENGINE_MAX
                T_min_target = n_engines_target * THROTTLE_MIN * T_ENGINE_MAX
                # Require that the desired thrust fits comfortably in the lower stage envelope
                if not (0.6 * T_min_target <= T_req_mag <= 0.9 * T_max_target):
                    return False

                a_z_target = (T_max_target / m) + g_vec[2]
                if a_z_target <= 1e-6:
                    return False
                t_stop = -v[2] / a_z_target if v[2] < 0 else 0.0
                dz_needed = v[2] * t_stop + 0.5 * a_z_target * (t_stop ** 2)
                safety = 1.2
                has_altitude = alt > safety * abs(dz_needed)
                has_time = t_ballistic > max(burn_window, safety * t_stop)
                return has_altitude and has_time

            # Stage Down 3->2
            if current_stage_max == 3 and (t - t_last_stage_change) >= min_dwell:
                if stage_safe_to_drop(2):
                    current_stage_max = 2
                    t_last_stage_change = t
                    print(
                        f"Stage Down 3->2 at t={t:.2f}, Alt={alt:.1f}m, "
                        f"T_req={T_req_mag/1e3:.1f}kN"
                    )

            # Stage Down 2->1
            if current_stage_max == 2 and (t - t_last_stage_change) >= min_dwell:
                if stage_safe_to_drop(1):
                    current_stage_max = 1
                    t_last_stage_change = t
                    print(
                        f"Stage Down 2->1 at t={t:.2f}, Alt={alt:.1f}m, "
                        f"T_req={T_req_mag/1e3:.1f}kN"
                    )

        # Step 3: Final Cutoff (one-way switch)
        if engines_on and alt < 5.0 and np.linalg.norm(v) < 0.5:
            engines_on = False
            current_stage_max = 0
            T_req_mag = 0.0 # Force zero thrust

        # F. Horizontal Freeze Logic
        v_h = np.linalg.norm(v[:2])
        if freeze_horizontal:
            if horizontal_done or v_h < 2.0:
                horizontal_done = True
                a_cmd[0] = 0.0
                a_cmd[1] = 0.0
        
        # G. Engine Allocation
        T_des = m * a_cmd
        T_des_mag = np.linalg.norm(T_des) if engines_on else 0.0

        # Engine allocation is sticky within a stage: once a stage is selected,
        # hold that engine count and only adjust throttle. This prevents
        # oscillating ignitions.
        if engines_on and current_stage_max > 0:
            n_engines = current_stage_max
            stage_min = n_engines * THROTTLE_MIN * T_ENGINE_MAX
            stage_max = n_engines * THROTTLE_MAX * T_ENGINE_MAX
            T_cmd_mag = float(np.clip(T_des_mag, stage_min, stage_max))
        else:
            n_engines, T_cmd_mag = 0, 0.0

        # H. Apply to Dynamics (Set the min/max limits for the dynamics object)
        if n_engines == 0:
            dyn.thrust_min = 0.0
            dyn.thrust_max = 0.0
        else:
            dyn.thrust_min = n_engines * THROTTLE_MIN * T_ENGINE_MAX
            dyn.thrust_max = n_engines * THROTTLE_MAX * T_ENGINE_MAX

        # I. Clamp thrust direction
        T_vec = clamp_thrust_direction(a_cmd, m, T_cmd_mag)

        # J0. Vertical braking overlay to drive v->0 at alt->0 smoothly
        if engines_on and alt > 0:
            a_stop = (abs(v[2]) ** 2) / max(alt, 1.0)
            a_stop *= 0.5  # convert to v^2/(2*alt)
            a_stop *= 1.1  # safety factor
            # Required thrust acceleration to realize that upward net accel
            a_thrust_req = max(a_cmd[2], a_stop - g_vec[2])
            # Update thrust vector magnitude to meet the higher vertical demand
            if T_cmd_mag > 0:
                desired_Tz = a_thrust_req * m
                desired_Tz = np.clip(desired_Tz, dyn.thrust_min, dyn.thrust_max)

                # Preserve any horizontal authority but enforce the vertical component needed
                T_horizontal = T_vec.copy()
                T_horizontal[2] = 0.0
                T_vec = T_horizontal
                T_vec[2] = desired_Tz

                # Respect gimbal limits while keeping the new magnitude
                T_vec = clamp_direction_to_cone(T_vec, np.array([0.0, 0.0, 1.0]), 20.0)
                if np.linalg.norm(T_vec) > 1e-9:
                    T_vec = (T_vec / np.linalg.norm(T_vec)) * min(T_cmd_mag, dyn.thrust_max)
                    T_cmd_mag = np.linalg.norm(T_vec)

        # J. CRITICAL High T/W Throttle Scaling (The Hover-Lock Fix)
        # Scale thrust back if the minimum throttle would cause upward acceleration.
        if engines_on and m > 0.0:
            # Approximate vertical accel from thrust + gravity (ignore drag)
            a_z_thrust_only = T_vec[2] / m
            a_z_actual = g_vec[2] + a_z_thrust_only 
            
            # Predict next vertical velocity
            v_z_next_pred = v[2] + a_z_actual * dt_sim

            if v_z_next_pred > 0.0 and a_z_thrust_only > 0.0:
                # Calculate scale factor 's' to set v_z_next_pred ≈ 0
                scale = -(v[2] + g_vec[2] * dt_sim) / (a_z_thrust_only * dt_sim)
                scale = np.clip(scale, 0.0, 1.0)
                
                T_vec *= scale
                T_cmd_mag = np.linalg.norm(T_vec) # Update actual thrust magnitude


                    
        # K. Step Integration
        (r, v, m), a_actual, _ = dyn.step((r, v, m), T_vec, dt_sim)

        # Record
        traj["t"].append(t)
        traj["r"].append(r.copy())
        traj["v"].append(v.copy())
        traj["m"].append(m)
        traj["engines"].append(n_engines)
        
        t += dt_sim

    # If we exited with a small positive altitude, project the final free-fall leg to ground
    if r[2] > 0.0:
        gz = g_vec[2]
        t_ground = estimate_ballistic_time_to_ground(r, v, g_vec, tf_fallback=0.0)
        if t_ground > 0:
            r = r + v * t_ground + 0.5 * np.array([0.0, 0.0, gz]) * (t_ground ** 2)
            v = v + np.array([0.0, 0.0, gz]) * t_ground
            t += t_ground
            traj["t"].append(t)
            traj["r"].append(r.copy())
            traj["v"].append(v.copy())
            traj["m"].append(m)
            traj["engines"].append(0)

    # Convert to arrays
    for k in traj:
        traj[k] = np.array(traj[k])

    return traj, ref


if __name__ == "__main__":
    # quick manual test
    r0 = np.array([0.0, 0.0, 2000.0])
    v0 = np.array([100.0, 0.0, -50.0])
    m0 = 1.5e5

    traj, ref = simulate_landing_once(r0, v0, m0)
    print("Sim finished. Final state:")
    print("r:", traj["r"][-1])
    print("v:", traj["v"][-1])
    print("m:", traj["m"][-1])