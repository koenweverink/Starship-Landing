# zem_zev_guidance.py
import numpy as np


class ZEMZEVGuidance:
    def __init__(self, g_vec, T_engine_max=3_000_000.0, m_nom=150_000.0):
        self.g = np.array(g_vec, dtype=float)
        self.T_engine_max = float(T_engine_max)   # one Raptor vacuum thrust
        self.m_current = float(m_nom)             # will be updated every step
        self.g_mag = -self.g[2]                   # positive gravity magnitude

    def update_mass(self, m):
        """Call this every simulation step with current mass"""
        self.m_current = float(m)

    def compute_tgo(self, r, v, r_f=None, v_f=None, min_tgo=1.5, n_engines=3):
        if r_f is None:
            r_f = np.zeros(3)
        if v_f is None:
            v_f = np.zeros(3)

        R = np.array(r - r_f, dtype=float)
        V = np.array(v - v_f, dtype=float)

        z = R[2]
        vz = V[2]

        if z <= 1.0:
            return min_tgo

        # Maximum upward acceleration with 3 engines (we always assume we can use 3 for t_go estimate)
        a_max_thrust = n_engines * self.T_engine_max / max(self.m_current, 50_000.0)
        a_net_max = a_max_thrust - self.g_mag

        if a_net_max <= 0.5:                     # impossible to arrest
            return 100.0

        # Analytic optimal bang-bang time-to-go (the one SpaceX really uses)
        discriminant = vz**2 + 2.0 * a_net_max * z
        if discriminant < 0:
            return 100.0

        t_go = (-vz + np.sqrt(discriminant)) / a_net_max

        # Safety: never trust an unrealistically short t_go when still very high/fast
        if z > 10_000.0 and t_go < 15.0:
            t_go = max(t_go, 30.0)

        return max(t_go, min_tgo)

    def compute_accel(self, r, v, t_go, r_f=None, v_f=None):
        if r_f is None:
            r_f = np.zeros(3)
        if v_f is None:
            v_f = np.zeros(3)

        r = np.array(r, dtype=float)
        v = np.array(v, dtype=float)

        ZEM = r_f - (r + v * t_go + 0.5 * self.g * t_go**2)
        ZEV = v_f - (v + self.g * t_go)

        # Classic ZEM/ZEV proportional navigation law
        a_cmd = 6.0 * ZEM / (t_go**2 + 1e-6) - 2.0 * ZEV / (t_go + 1e-6)
        return a_cmd