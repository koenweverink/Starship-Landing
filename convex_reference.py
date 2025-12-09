# convex_reference.py

import numpy as np
import cvxpy as cp


def solve_reference_trajectory(
    r0,
    v0,
    g_vec,
    tf,
    N=60,
    a_max_3=18.0,
    a_max_2=12.0,
    a_max_1=8.0,
):
    """
    Solve a convex guidance problem for powered descent.

    States:
        r_k, v_k   for k = 0..N (3D)
    Control:
        u_k        for k = 0..N-1  (3D)  = commanded accel due to thrust (m/s^2)

    Dynamics (discrete):
        v_{k+1} = v_k + (g + u_k) * dt
        r_{k+1} = r_k + v_k*dt + 0.5*(g + u_k)*dt^2

    Constraints:
        - initial r_0, v_0 match
        - final r_N = 0, v_N = 0
        - z_k is monotonically non-increasing (descent only)
        - ||u_k||_2 <= a_max_stage(k) for each stage:
            * early: all engines available   → a_max_3
            * mid:   throttled / partial set  → a_max_2
            * late:  single-engine tail       → a_max_1

    Objective:
        minimize sum_k ||u_k||_2^2  (quadratic effort)

    Returns:
        ref: dict with keys "t", "r", "v", "a" (all numpy arrays)
             where "a" = u + g  (total accel including gravity)
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    g_vec = np.asarray(g_vec, dtype=float)

    dt = tf / N

    # Decision variables
    r = cp.Variable((N + 1, 3))
    v = cp.Variable((N + 1, 3))
    u = cp.Variable((N, 3))

    constraints = []

    # Initial conditions
    constraints += [r[0, :] == r0, v[0, :] == v0]

    # Stage-dependent accel bounds (early high thrust → mid → late low thrust)
    # We'll split the N steps into 3 segments by index.
    idx_3_to_2 = int(0.4 * N)   # first 40% of horizon: max thrust phase
    idx_2_to_1 = int(0.7 * N)   # next 30%: throttled/partial engines
    # last 30%: single-engine tail

    for k in range(N):
        # discrete dynamics
        constraints += [
            v[k + 1, :] == v[k, :] + (g_vec + u[k, :]) * dt,
            r[k + 1, :] == r[k, :] + v[k, :] * dt + 0.5 * (g_vec + u[k, :]) * dt * dt,
        ]

        # Monotonic descent in z (no altitude increases)
        constraints += [r[k + 1, 2] <= r[k, 2]]

        # Stage-dependent accel limits
        if k < idx_3_to_2:
            a_max_k = a_max_3
        elif k < idx_2_to_1:
            a_max_k = a_max_2
        else:
            a_max_k = a_max_1

        constraints += [cp.norm(u[k, :], 2) <= a_max_k]

    # Final conditions: land at origin with zero velocity
    constraints += [
        r[N, :] == np.zeros(3),
        v[N, :] == np.zeros(3),
    ]

    # Objective: minimize squared accel usage
    objective = cp.Minimize(cp.sum(cp.sum_squares(u)))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, warm_start=True)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Convex solver failed with status {prob.status}")

    r_ref = r.value
    v_ref = v.value
    u_ref = u.value
    t_ref = np.linspace(0.0, tf, N + 1)

    # Total accel including gravity (for plotting / feedforward)
    # u is thrust-only accel; add g to get inertial accel.
    a_ref = np.zeros_like(r_ref)
    for k in range(N):
        a_ref[k, :] = u_ref[k, :] + g_vec
    a_ref[-1, :] = a_ref[-2, :]

    ref = {
        "t": np.array(t_ref),
        "r": np.array(r_ref),
        "v": np.array(v_ref),
        "a": np.array(a_ref),
    }
    return ref
