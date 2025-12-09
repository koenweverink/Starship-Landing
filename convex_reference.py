# convex_reference.py

import importlib.util
import numpy as np

CVXPY_AVAILABLE = importlib.util.find_spec("cvxpy") is not None
if CVXPY_AVAILABLE:
    import cvxpy as cp
else:
    cp = None


def solve_reference_trajectory(
    r0,
    v0,
    g_vec,
    tf,
    N=60,
    a_max_3=25.0,
    a_max_2=18.0,
    a_max_1=12.0,
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
            * early: 3 engines  → a_max_3
            * mid:   2 engines  → a_max_2
            * late:  1 engine   → a_max_1

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

    if not CVXPY_AVAILABLE:
        # Lightweight fallback: linearly blend toward the ground with finite differences
        t_ref = np.linspace(0.0, tf, N + 1)
        r_ref = np.linspace(r0, np.zeros(3), N + 1)
        v_ref = np.gradient(r_ref, t_ref, axis=0)
        a_ref = np.gradient(v_ref, t_ref, axis=0) + g_vec
        return {
            "t": t_ref,
            "r": r_ref,
            "v": v_ref,
            "a": a_ref,
        }

    # Decision variables
    r = cp.Variable((N + 1, 3))
    v = cp.Variable((N + 1, 3))
    u = cp.Variable((N, 3))

    constraints = []

    # Initial conditions
    constraints += [r[0, :] == r0, v[0, :] == v0]

    # Stage-dependent accel bounds (approx 3→2→1 engines)
    # We'll split the N steps into 3 segments by index.
    idx_3_to_2 = int(0.4 * N)   # first 40% of horizon: up to 3 engines
    idx_2_to_1 = int(0.7 * N)   # next 30%: up to 2 engines
    # last 30%: up to 1 engine

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
