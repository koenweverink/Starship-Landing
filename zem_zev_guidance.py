import numpy as np

class ZEMZEVGuidance:
    def __init__(self, g_vec):
        # gravity vector g(r)  [m/s^2]
        self.g = np.array(g_vec, dtype=float)

    def compute_accel(self, r, v, t_go, r_f=None, v_f=None):
        """
        Compute ZEM/ZEV *control* acceleration a(t) for dynamics:

            r_ddot = g + a

        r, v      : current position/velocity (3,)
        t_go      : time-to-go = t_f - t
        r_f, v_f  : desired final state (default: 0,0)
        """
        if r_f is None:
            r_f = np.zeros(3)
        if v_f is None:
            v_f = np.zeros(3)

        r = np.array(r, dtype=float)
        v = np.array(v, dtype=float)

        # ZEM and ZEV for constant gravity g
        ZEM = r_f - (r + v * t_go + 0.5 * self.g * t_go**2)
        ZEV = v_f - (v + self.g * t_go)

        # Hawkins/Furfaro ZEM/ZEV law (Eq. 11 in the paper):
        #   a = 6/t_go^2 * ZEM - 2/t_go * ZEV
        a_cmd = 6.0 * ZEM / (t_go**2) - 2.0 * ZEV / t_go
        return a_cmd   # this is control accel a(t), not total

    def accel_to_thrust(self, a_cmd, m):
        """
        Convert control acceleration a_cmd to thrust vector.

        Since r_ddot = g + a and a = T/m,
        => T = m * a_cmd
        """
        return m * a_cmd
