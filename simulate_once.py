import numpy as np
from mars_env import get_mars_params
from dynamics import LanderDynamics
from convex_reference import solve_reference_trajectory
from zem_zev_guidance import ZEMZEVGuidance


# --------------------------------------------------------
# Reference interpolator
# --------------------------------------------------------
def build_ref_interpolator(ref, g_vec):
    t_ref = ref["t"]
    r_ref = ref["r"]
    v_ref = ref["v"]
    a_ref_total = ref["a"]

    dt = t_ref[1] - t_ref[0]
    N = len(t_ref) - 1

    def interp(t_phase):
        if t_phase <= 0.0:
            rr, vv, aa = r_ref[0], v_ref[0], a_ref_total[0]
        elif t_phase >= t_ref[-1]:
            rr, vv, aa = r_ref[-1], v_ref[-1], a_ref_total[-1]
        else:
            i = int(t_phase / dt)
            if i >= N:
                i = N - 1
            tau = (t_phase - t_ref[i]) / dt
            tau = np.clip(tau, 0.0, 1.0)
            rr = (1 - tau) * r_ref[i] + tau * r_ref[i + 1]
            vv = (1 - tau) * v_ref[i] + tau * v_ref[i + 1]
            aa = (1 - tau) * a_ref_total[i] + tau * a_ref_total[i + 1]
        u = aa - g_vec  # thrust-only accel from convex plan
        return rr, vv, aa, u

    return interp, t_ref[-1]


# --------------------------------------------------------
# Ballistic time-to-ground (no drag)
# --------------------------------------------------------
def ballistic_ttg(r, v, g_vec):
    z, vz = r[2], v[2]
    if z <= 0.0:
        return 0.0
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


# --------------------------------------------------------
# Engine allocation: approximate requested thrust magnitude
# with N engines, each in [T_e_min, T_e_max] or off.
# Returns (T_total_alloc, n_eng_on).
# --------------------------------------------------------
def allocate_engine_thrust(T_req_mag, n_eng_max, T_e_min, T_e_max):
    if T_req_mag <= 0.0:
        return 0.0, 0  # all off

    best_n = 0
    best_T = 0.0
    best_err = None

    for n in range(1, n_eng_max + 1):
        per = T_req_mag / n
        # Per-engine thrust must be within [T_e_min, T_e_max]
        per_clamped = np.clip(per, T_e_min, T_e_max)
        T_total = n * per_clamped
        err = abs(T_total - T_req_mag)
        if (best_err is None) or (err < best_err):
            best_err = err
            best_T = T_total
            best_n = n

    return best_T, best_n


# --------------------------------------------------------
# Main simulation
# --------------------------------------------------------
def simulate_landing_once(
    r0,
    v0,
    m0,
    dt_sim=0.05,
    N_ref=60,
    freeze_horizontal=True,
):
    env = get_mars_params()
    g_vec = env["g_vec"]
    g_mag = -g_vec[2]

    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)
    # Attitude: quaternion (w,x,y,z) and body rates
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.zeros(3)
    m = float(m0)
    m_initial = m  # for fuel usage metrics

    # ---- Engine cluster: 3 engines ----
    N_ENG = 3
    T_E_MAX = 1_000_000.0                # per-engine max thrust [N]
    T_E_MIN_FRAC = 0.4                   # per-engine min throttle
    T_E_MIN = T_E_MIN_FRAC * T_E_MAX     # per-engine min thrust [N]
    T_CLUSTER_MAX = N_ENG * T_E_MAX      # cluster max thrust

    # Lateral accel, attitude, and body-rate limits
    A_LAT_MAX = 12.0                     # max lateral accel [m/s^2]
    TILT_HIGH_DEG = 28.0                 # pitch-over cap during early braking
    TILT_MID_DEG = 26.0                  # aggressive lateral kill mid-phase
    TILT_LOW_DEG = 20.0                  # stand-up begins
    TILT_FINAL_DEG = 10.0                # stay upright near the ground
    BODY_RATE_LIMIT_DEG = 15.0
    BODY_RATE_LIMIT_RAD = np.radians(BODY_RATE_LIMIT_DEG)

    dyn = LanderDynamics(
        g_vec=g_vec,
        isp=380.0,
        thrust_min=0.0,                  # we'll enforce mins in allocator
        thrust_max=T_CLUSTER_MAX,
        dry_mass=85_000.0,
        cd_area=120.0,
        rho0=0.020,
        h_scale=11000.0,
    )

    guidance = ZEMZEVGuidance(
        g_vec=g_vec,
        T_engine_max=T_E_MAX,            # per-engine, used for scaling
        m_nom=m0,
    )

    def tilt_limit_for_phase(altitude, v_h_speed):
        if altitude > 300.0:
            limit = np.radians(TILT_HIGH_DEG)
        elif altitude > 120.0:
            limit = np.radians(TILT_MID_DEG)
        elif altitude > 60.0:
            limit = np.radians(TILT_LOW_DEG)
        else:
            limit = np.radians(TILT_FINAL_DEG)

        if v_h_speed < 3.0 and altitude < 80.0:
            limit = min(limit, np.radians(8.0))

        # If we're still carrying significant horizontal speed late in the burn,
        # allow a slight increase in tilt to shed it before stand-up, but cap to
        # the early pitch-over limit.
        if altitude < 80.0 and v_h_speed > 5.0:
            limit = min(np.radians(TILT_HIGH_DEG), max(limit, np.radians(12.0)))

        return limit

    def horizontal_speed_cap(altitude):
        if altitude > 200.0:
            return 20.0
        if altitude > 120.0:
            return 10.0
        if altitude > 60.0:
            return 4.0
        return 1.0

    t = 0.0
    engines_on = False
    t_burn_start = None

    ref_interp = None
    tf_ref = None

    traj = {
        "t": [],
        "r": [],
        "v": [],
        "q": [],
        "w": [],
        "m": [],
        "engines": [],
        # telemetry
        "g_load": [],
        "thrust_mag": [],
        "fuel_used": [],
    }

    while r[2] > 0.05 and t < 300.0:
        alt = r[2]
        v_h = v[:2]
        v_h_mag = np.linalg.norm(v_h)
        guidance.update_mass(m)
        tilt_limit_rad = tilt_limit_for_phase(alt, v_h_mag)

        # ----------------- Ignition logic (vertical-only) -----------------
        if not engines_on:
            t_ball = ballistic_ttg(r, v, g_vec)

            a_net_max = T_CLUSTER_MAX / max(m, 1.0) - g_mag
            if a_net_max <= 0.0:
                a_net_max = 1.0

            t_stop = -v[2] / a_net_max if v[2] < 0.0 else 0.0

            # slightly conservative ignition rule
            if t_ball <= 1.05 * t_stop + 4.0:
                engines_on = True
                t_burn_start = t
                print(f"[t={t:.1f}s] ENGINES IGNITED @ {alt:.0f}m")

                tf_guess = max(2.0 * t_stop, 30.0)
                try:
                    ref = solve_reference_trajectory(
                        r, v, g_vec, tf_guess, N_ref
                    )
                    ref_interp, tf_ref = build_ref_interpolator(ref, g_vec)
                except Exception as e:
                    print(f"[WARN] Convex infeasible at ignition: {e}")
                    ref_interp, tf_ref = None, None

        n_engines_lit = 0
        T_vec = np.zeros(3)
        torque_cmd = np.zeros(3)

        # ----------------- Powered guidance & thrust -----------------
        if engines_on:
            v_z = v[2]
            safe_alt = max(alt, 1.0)

            # === Vertical: braking law (baseline) ===
            if v_z < 0.0:
                a_z_net = (v_z * v_z) / (2.0 * safe_alt)
            else:
                a_z_net = 0.5

            if alt > 80.0:
                a_z_net = np.clip(a_z_net, 0.5, 8.0)
            elif alt > 20.0:
                a_z_net = np.clip(a_z_net, 0.5, 12.0)
            else:
                a_z_net = np.clip(a_z_net, 1.0, 18.0)

            # Required vertical thrust (upwards)
            T_z_req = (a_z_net + g_mag) * m
            if T_z_req < 0.0:
                T_z_req = 0.0
            if T_z_req > T_CLUSTER_MAX:
                T_z_req = T_CLUSTER_MAX

            # === Horizontal: GN&C-produced lateral accel ===
            a_lat_cmd = np.zeros(2)

            if freeze_horizontal:
                if alt > 150.0:
                    # pitch-over: aggressively bleed horizontal speed early in the burn
                    if v_h_mag > 0.2:
                        vhu = v_h / v_h_mag
                        a_time_const = 10.0
                        a_side = np.clip(v_h_mag / max(a_time_const, 1.0), 0.75, A_LAT_MAX)
                        a_lat_cmd = -a_side * vhu

                    if ref_interp is not None:
                        t_phase = t - t_burn_start
                        _, _, _, u_ref_t = ref_interp(t_phase)
                        t_go_rem = tf_ref - t_phase if tf_ref is not None else 20.0
                        t_go_rem = max(t_go_rem, 5.0)

                        a_net_zem = guidance.compute_accel(r, v, t_go_rem)
                        a_thrust_zem = a_net_zem - g_vec

                        alpha = 0.45
                        blend_cmd = (1 - alpha) * u_ref_t[:2] + alpha * a_thrust_zem[:2]
                        if np.linalg.norm(a_lat_cmd) > 1e-6:
                            a_lat_cmd = 0.5 * a_lat_cmd + 0.5 * blend_cmd
                        else:
                            a_lat_cmd = blend_cmd
                elif alt > 80.0:
                    # mid-phase: still allow tilt but bias toward horizontal kill
                    if ref_interp is not None:
                        t_phase = t - t_burn_start
                        _, _, _, u_ref_t = ref_interp(t_phase)
                        t_go_rem = tf_ref - t_phase if tf_ref is not None else 15.0
                        t_go_rem = max(t_go_rem, 5.0)

                        a_net_zem = guidance.compute_accel(r, v, t_go_rem)
                        a_thrust_zem = a_net_zem - g_vec

                        alpha = 0.4
                        a_lat_cmd = (1 - alpha) * u_ref_t[:2] + alpha * a_thrust_zem[:2]
                    if v_h_mag > 0.2 and np.linalg.norm(a_lat_cmd) < 1e-6:
                        a_lat_cmd = -min(A_LAT_MAX, v_h_mag) * (v_h / v_h_mag)
                elif alt > 20.0:
                    # terminal braking: hard lateral kill before stand-up
                    if v_h_mag > 0.15:
                        vhu = v_h / v_h_mag
                        a_side = min(A_LAT_MAX, max(v_h_mag * 1.2, 1.5))
                        a_lat_cmd = -a_side * vhu
                else:
                    # stand-up: keep nearly vertical, only damp residual drift
                    if v_h_mag > 0.08:
                        vhu = v_h / v_h_mag
                        a_side = np.clip(v_h_mag / 2.0, 0.0, 2.5)
                        a_lat_cmd = -a_side * vhu

            # Enforce an altitude-indexed horizontal speed envelope with extra braking
            v_h_cap = horizontal_speed_cap(alt)
            if v_h_mag > v_h_cap:
                vhu = v_h / v_h_mag
                excess = v_h_mag - v_h_cap
                tau = np.clip(alt / 40.0, 0.5, 8.0)
                a_envelope = np.clip(excess / tau, 0.5, A_LAT_MAX)
                brake_cmd = -a_envelope * vhu
                if np.linalg.norm(a_lat_cmd) > 1e-6:
                    a_lat_cmd = 0.5 * a_lat_cmd + 0.5 * brake_cmd
                else:
                    a_lat_cmd = brake_cmd

            # Limit lateral accel
            a_lat_mag = np.linalg.norm(a_lat_cmd)
            if a_lat_mag > A_LAT_MAX:
                a_lat_cmd *= A_LAT_MAX / a_lat_mag

            # Desired lateral thrust
            F_lat_des = m * a_lat_cmd
            F_lat_mag = np.linalg.norm(F_lat_des)

            # Caps on lateral thrust
            F_lat_cap_acc = m * A_LAT_MAX
            F_lat_cap_tilt = T_z_req * np.tan(tilt_limit_rad) if T_z_req > 0.0 else 0.0

            if T_z_req >= T_CLUSTER_MAX:
                F_lat_cap_T = 0.0
            else:
                F_lat_cap_T = np.sqrt(max(T_CLUSTER_MAX**2 - T_z_req**2, 0.0))

            F_lat_cap = min(F_lat_cap_acc, F_lat_cap_tilt, F_lat_cap_T)
            if F_lat_mag > 1e-6 and F_lat_cap > 0.0:
                if F_lat_mag > F_lat_cap:
                    F_lat_des *= F_lat_cap / F_lat_mag
            else:
                F_lat_des[:] = 0.0

            # Requested total thrust vector
            T_req = np.array([F_lat_des[0], F_lat_des[1], T_z_req])
            T_req_mag = np.linalg.norm(T_req)

            if T_req_mag > 1e-6:
                T_dir = T_req / T_req_mag
                # Engine allocation
                T_alloc_mag, n_engines_lit = allocate_engine_thrust(
                    T_req_mag, N_ENG, T_E_MIN, T_E_MAX
                )
                if T_alloc_mag > T_CLUSTER_MAX:
                    T_alloc_mag = T_CLUSTER_MAX
                T_vec = T_dir * T_alloc_mag
            else:
                T_vec = np.zeros(3)
                n_engines_lit = 0

            # Soft-touch cutoff
            if alt < 3.0 and abs(v_z) < 1.0 and v_h_mag < 0.8:
                print(
                    f"[t={t:.1f}s] TOUCHDOWN! "
                    f"v_vert = {v_z:.3f} m/s, v_h = {v_h_mag:.3f} m/s"
                )
                T_vec[:] = 0.0
                engines_on = False
                n_engines_lit = 0

        # ----------------- Integrate translation -----------------
        dyn.thrust_min = 0.0
        dyn.thrust_max = T_CLUSTER_MAX

        # --- 6-DOF attitude control ---
        if np.linalg.norm(T_vec) > 1e-6:
            desired_dir = T_vec / np.linalg.norm(T_vec)
        else:
            desired_dir = np.array([0.0, 0.0, 1.0])

        # Build desired attitude frame with z-axis along desired thrust
        z_b_des = desired_dir
        x_ref = np.array([1.0, 0.0, 0.0]) if abs(np.dot(z_b_des, [1, 0, 0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_b_des = np.cross(x_ref, z_b_des)
        x_b_norm = np.linalg.norm(x_b_des)
        if x_b_norm < 1e-6:
            x_b_des = np.array([1.0, 0.0, 0.0])
        else:
            x_b_des /= x_b_norm
        y_b_des = np.cross(z_b_des, x_b_des)

        R_des = np.column_stack([x_b_des, y_b_des, z_b_des])
        q_des = dyn.dcm_to_quat(R_des)

        # Quaternion error to torque command
        q_conj = dyn.quat_conjugate(q)
        q_err = dyn.quat_multiply(q_des, q_conj)
        if q_err[0] < 0.0:
            q_err = -q_err
        rot_vec = q_err[1:]

        Kp = 5.0e6
        Kd = 1.0e6
        w_mag = np.linalg.norm(w)
        if w_mag > BODY_RATE_LIMIT_RAD:
            # Rate saturation: prioritize damping when already too fast
            torque_cmd = -Kd * w
        else:
            torque_cmd = -Kp * rot_vec - Kd * w

            # Predict the rate after this torque and clamp if it would exceed limits
            w_cross_Jw = np.cross(w, dyn.J * w)
            w_dot_cmd = (torque_cmd - w_cross_Jw) * dyn.J_inv
            w_pred = w + w_dot_cmd * dt_sim
            w_pred_mag = np.linalg.norm(w_pred)
            if w_pred_mag > BODY_RATE_LIMIT_RAD:
                w_delta = w_pred - w
                delta_mag = np.linalg.norm(w_delta)
                if delta_mag > 1e-9:
                    scale = max(0.0, (BODY_RATE_LIMIT_RAD - w_mag) / delta_mag)
                    scale = np.clip(scale, 0.0, 1.0)
                    torque_cmd *= scale

        # Thrust expressed in current body frame
        T_body = dyn.quat_to_dcm(q).T @ T_vec

        (r, v, q, w, m), a, _, _ = dyn.step_6dof((r, v, q, w, m), T_body, torque_cmd, dt_sim)

        # Telemetry metrics
        thrust_mag = np.linalg.norm(T_vec)
        a_mag = np.linalg.norm(a)
        g_load = a_mag / 9.80665
        fuel_used = m_initial - m

        traj["t"].append(t)
        traj["r"].append(r.copy())
        traj["v"].append(v.copy())
        traj["q"].append(q.copy())
        traj["w"].append(w.copy())
        traj["m"].append(m)
        traj["engines"].append(n_engines_lit)
        traj["g_load"].append(g_load)
        traj["thrust_mag"].append(thrust_mag)
        traj["fuel_used"].append(fuel_used)

        t += dt_sim

    for k in traj:
        traj[k] = np.array(traj[k])

    # ---------- Summary metrics ----------
    if len(traj["t"]) > 0:
        v_final = traj["v"][-1]
        v_touch_vert = float(v_final[2])
        v_touch_horiz = float(np.linalg.norm(v_final[:2]))
        fuel_used_total = float(m_initial - traj["m"][-1])
        peak_g = float(np.max(traj["g_load"]))
        max_thrust = float(np.max(traj["thrust_mag"]))
        min_alt = float(np.min(traj["r"][:, 2]))
        # Attitude/aero metrics
        z_axes = np.array([LanderDynamics.quat_to_dcm(q)[:, 2] for q in traj["q"]])
        tilt_angles = np.degrees(np.arccos(np.clip(z_axes[:, 2], -1.0, 1.0)))
        peak_tilt_deg = float(np.max(tilt_angles))
        body_rate_mag = np.linalg.norm(traj["w"], axis=1)
        peak_body_rate_deg = float(np.max(np.degrees(body_rate_mag)))
    else:
        v_touch_vert = v_touch_horiz = fuel_used_total = peak_g = max_thrust = min_alt = 0.0
        peak_tilt_deg = peak_body_rate_deg = 0.0

    summary = {
        "m0": float(m_initial),
        "m_final": float(traj["m"][-1]) if len(traj["m"]) > 0 else float(m_initial),
        "fuel_used_total": fuel_used_total,
        "v_touch_vertical": v_touch_vert,
        "v_touch_horizontal": v_touch_horiz,
        "peak_g_load": peak_g,
        "max_thrust": max_thrust,
        "min_altitude": min_alt,
        "peak_tilt_deg": peak_tilt_deg,
        "peak_body_rate_deg": peak_body_rate_deg,
    }

    return traj, summary


# --------------------------------------------------------
# Quick test
# --------------------------------------------------------
if __name__ == "__main__":
    r0 = np.array([0.0, 0.0, 2000.0])
    v0 = np.array([90.0, 0.0, -60.0])
    m0 = 150_000.0

    traj, summary = simulate_landing_once(r0, v0, m0)

    import matplotlib.pyplot as plt
    t = traj["t"]

    fig, ax = plt.subplots(3, 3, figsize=(14, 10))
    ax[0, 0].plot(t, traj["r"][:, 2]); ax[0, 0].set_title("Altitude [m]")
    ax[0, 1].plot(t, traj["v"][:, 2]); ax[0, 1].set_title("Vertical Velocity [m/s]")
    ax[0, 2].plot(t, np.linalg.norm(traj["v"][:, :2], axis=1)); ax[0, 2].set_title("Horizontal Speed [m/s]")

    ax[1, 0].plot(t, traj["thrust_mag"]); ax[1, 0].set_title("Thrust Magnitude [N]")
    ax[1, 1].step(t, traj["engines"], where="post"); ax[1, 1].set_title("Engines On (count)")
    ax[1, 2].plot(t, traj["fuel_used"]); ax[1, 2].set_title("Fuel Used [kg]")

    # Attitude and loads
    z_axes = np.array([LanderDynamics.quat_to_dcm(q)[:, 2] for q in traj["q"]])
    tilt_deg = np.degrees(np.arccos(np.clip(z_axes[:, 2], -1.0, 1.0)))
    ax[2, 0].plot(t, tilt_deg); ax[2, 0].set_title("Body Tilt vs. Vertical [deg]")

    rate_mag_deg = np.degrees(np.linalg.norm(traj["w"], axis=1))
    ax[2, 1].plot(t, rate_mag_deg); ax[2, 1].set_title("Body Rate Magnitude [deg/s]")

    ax[2, 2].plot(t, traj["g_load"]); ax[2, 2].set_title("G-Load")

    for axes in ax.flat:
        axes.grid(True, linestyle="--", alpha=0.4)
        axes.set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

    print("TOUCHDOWN vertical   :", summary["v_touch_vertical"])
    print("TOUCHDOWN horizontal :", summary["v_touch_horizontal"])
    print("Fuel used [t]        :", summary["fuel_used_total"] / 1000.0)
    print("Peak g-load          :", summary["peak_g_load"])
    print("Max thrust [MN]      :", summary["max_thrust"] / 1e6)
    print("Peak tilt [deg]      :", summary["peak_tilt_deg"])
    print("Peak body rate [deg/s]:", summary["peak_body_rate_deg"])

