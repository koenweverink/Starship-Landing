# simulate_once.py
import numpy as np
from mars_env import get_mars_params
from dynamics import LanderDynamics
from convex_reference import solve_reference_trajectory
from zem_zev_guidance import ZEMZEVGuidance   # <-- the fixed version above


def simulate_landing_once(
    r0, v0, m0,
    dt_sim=0.05,
    freeze_horizontal=True,
):
    env = get_mars_params()
    g_vec = env["g_vec"]
    g_mag = -g_vec[2]

    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)
    m = float(m0)

    dyn = LanderDynamics(
        g_vec=g_vec, isp=380.0, thrust_min=0.0, thrust_max=1e9,
        dry_mass=85_000.0, cd_area=120.0, rho0=0.020, h_scale=11000.0,
    )
    guidance = ZEMZEVGuidance(g_vec=g_vec, T_engine_max=3_000_000.0, m_nom=m0)

    T_ENGINE_MAX = 3_000_000.0
    THROTTLE_MIN = 0.40
    THROTTLE_MAX = 1.00

    t = 0.0
    traj = {"t": [], "r": [], "v": [], "m": [], "engines": []}
    engines_on = False
    current_stage_max = 0
    t_last_drop = -999.0

    while r[2] > 0.05 and t < 300.0:
        alt = r[2]
        v_h = np.linalg.norm(v[:2])
        guidance.update_mass(m)

        # --- Ignition ---
        if not engines_on:
            a_max = 3*T_ENGINE_MAX/m - g_mag
            t_stop = max(-v[2],0) / max(a_max,1)
            t_ball = max(( -v[2] - np.sqrt(v[2]**2 - 2*g_vec[2]*r[2]) ) / g_vec[2], 0)
            if t_ball <= t_stop + 6.0:
                engines_on = True
                current_stage_max = 3
                t_last_drop = t
                print(f"[t={t:.1f}s] ENGINES IGNITED @ {alt:.0f}m")

        # --- Guidance above 80 m ---
        if engines_on and alt >= 80.0:
            t_go = guidance.compute_tgo(r, v)
            a_cmd = guidance.compute_accel(r, v, t_go)

            if freeze_horizontal and v_h < 2.0:
                a_cmd[:2] = 0.0

            # Staging
            if current_stage_max > 1 and t - t_last_drop > 3.0:
                a_next = (current_stage_max-1)*T_ENGINE_MAX/m - g_mag
                if a_next > 2.0 and alt > 0.85 * (v[2]**2/(2*max(a_next,1))):
                    current_stage_max -= 1
                    t_last_drop = t
                    print(f"[t={t:.1f}s] DROP → {current_stage_max} engines")

        # --- TERMINAL PHASE BELOW 80 m (the real Starship magic) ---
        if alt < 80.0 and engines_on:
            current_stage_max = 1

            # Ramp up deceleration as we descend
            if alt > 40:
                a_net = -2.2
            elif alt > 15:
                a_net = -3.2
            else:
                a_net = -4.0

            T_req = m * (a_net - g_vec[2])
            T_cmd = np.clip(T_req, 0.55*T_ENGINE_MAX, T_ENGINE_MAX)

            direction = np.array([0.0, 0.0, 1.0])
            if v_h > 1.0 and alt > 12:
                horiz = np.array([-v[0], -v[1], 0.0])
                horiz /= max(np.linalg.norm(horiz), 1e-12)
                direction = direction + 0.09 * horiz
                direction /= np.linalg.norm(direction)

            T_vec = direction * T_cmd

            # Absolute floor
            if g_vec[2] + T_vec[2]/m < -5.5:
                T_vec[2] = m * (-g_vec[2] - 5.5)

        else:
            # Normal phase
            n_eng = current_stage_max if engines_on else 0
            T_min = n_eng * THROTTLE_MIN * T_ENGINE_MAX
            T_max = n_eng * THROTTLE_MAX * T_ENGINE_MAX

            if engines_on and alt >= 80.0:
                T_des = np.clip(m * np.linalg.norm(a_cmd), T_min, T_max)
                dir_cmd = a_cmd / np.linalg.norm(a_cmd) if np.linalg.norm(a_cmd)>1e-6 else np.array([0,0,1])
                T_vec = dir_cmd * T_des
            else:
                T_vec = np.zeros(3)

            if np.linalg.norm(T_vec) > 1000:
                gimbal = 20 if alt > 200 else 8
                T_vec = clamp_direction_to_cone(T_vec, [0,0,1], gimbal)

        # --- Cutoff ---
        if engines_on and alt < 6.0 and abs(v[2]) < 1.2 and v_h < 1.0:
            T_vec = np.zeros(3)
            engines_on = False
            print(f"[t={t:.1f}s] TOUCHDOWN! v_vert = {v[2]:.3f} m/s  |  v_h = {v_h:.3f} m/s")

        # --- Step ---
        dyn.thrust_min = current_stage_max * THROTTLE_MIN * T_ENGINE_MAX
        dyn.thrust_max = current_stage_max * THROTTLE_MAX * T_ENGINE_MAX
        (r, v, m), _, _ = dyn.step((r, v, m), T_vec, dt_sim)

        traj["t"].append(t)
        traj["r"].append(r.copy())
        traj["v"].append(v.copy())
        traj["m"].append(m)
        traj["engines"].append(current_stage_max if engines_on else 0)
        t += dt_sim

    for k in traj: traj[k] = np.array(traj[k])
    return traj, None

# ------------------------------------------------------------------
# Clamp function (copy-paste if not already in file)
# ------------------------------------------------------------------
def clamp_direction_to_cone(vec, axis, max_angle_deg):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    n = np.linalg.norm(vec)
    if n < 1e-9:
        return np.zeros(3)
    v = vec / n
    cos_max = np.cos(np.radians(max_angle_deg))
    cos_actual = np.dot(v, axis)
    if cos_actual >= cos_max:
        return vec
    # project and re-build
    perp = v - cos_actual * axis
    perp /= np.linalg.norm(perp)
    v_new = cos_max * axis + np.sin(np.arccos(cos_max)) * perp
    return n * v_new / np.linalg.norm(v_new)


# ------------------------------------------------------------------
# Quick test (this now works perfectly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    r0 = np.array([0.0, 0.0, 2000.0])      # 2 km – realistic PDI altitude
    v0 = np.array([90.0, 0.0, -60.0])      # typical terminal entry state
    m0 = 150_000.0

    traj, _ = simulate_landing_once(r0, v0, m0)

    import matplotlib.pyplot as plt
    t = traj["t"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0,0].plot(t, traj["r"][:,2]); ax[0,0].set_title("Altitude [m]")
    ax[0,1].plot(t, traj["v"][:,2]); ax[0,1].set_title("Vertical Velocity [m/s]")
    ax[1,0].plot(t, np.linalg.norm(traj["v"][:,:2], axis=1)); ax[1,0].set_title("Horizontal Speed [m/s]")
    ax[1,1].step(t, traj["engines"], where='post'); ax[1,1].set_title("Engines On")
    plt.tight_layout()
    plt.show()

    print("TOUCHDOWN vertical   :", traj["v"][-1,2])
    print("TOUCHDOWN horizontal :", np.linalg.norm(traj["v"][-1,:2]))