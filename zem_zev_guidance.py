import numpy as np

class ZEMZEVGuidance:
    def __init__(self, g_vec):
        # gravity vector g(r)  [m/s^2]
        self.g = np.array(g_vec, dtype=float)

    def compute_tgo(self, r, v, r_f=None, v_f=None, min_tgo=2.0):
        """
        Estimate optimal time-to-go.
        Handles both "Closing" phase (heuristic) and "Climbing/Hovering" phase (ballistic).
        """
        if r_f is None: r_f = np.zeros(3)
        if v_f is None: v_f = np.zeros(3)

        R_vec = r - r_f
        V_vec = v - v_f
        
        dist = np.linalg.norm(R_vec)
        closing_speed = 0.0
        if dist > 1e-3:
            closing_speed = -np.dot(R_vec, V_vec) / dist

        # --- Physics-based Fallback ---
        # Calculate how long it would take to fall to the ground if we cut engines.
        # z(t) approx = z + vz*t - 0.5*g*t^2
        # This prevents "panic" if we are climbing (vz > 0).
        z = R_vec[2]
        vz = V_vec[2]
        g_mag = np.linalg.norm(self.g) # approx 3.71
        
        t_ballistic = 10.0 # Default
        if z > 0:
            # Roots of 0.5*g*t^2 - vz*t - z = 0
            # t = (vz + sqrt(vz^2 + 2*g*z)) / g
            discriminant = vz**2 + 2 * g_mag * z
            if discriminant >= 0:
                t_ballistic = (vz + np.sqrt(discriminant)) / g_mag

        # --- Decision Logic ---
        if closing_speed > 5.0:
            # We are closing in fast. Use the standard soft-landing heuristic.
            # t_go = 2 * Distance / Speed
            t_opt = 2.0 * dist / closing_speed
        else:
            # We are slow, hovering, or climbing.
            # Use the ballistic time + a "comfort margin" (e.g. 1.2x)
            # This tells guidance: "You have plenty of time, just guide it down gently."
            t_opt = t_ballistic * 1.2

        return max(t_opt, min_tgo)

    def compute_accel(self, r, v, t_go, r_f=None, v_f=None):
        """
        Compute ZEM/ZEV *control* acceleration a(t).
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

        # Hawkins/Furfaro ZEM/ZEV law
        # a_cmd = 6*ZEM/t^2 - 2*ZEV/t
        a_cmd = 6.0 * ZEM / (t_go**2) - 2.0 * ZEV / t_go
        return a_cmd
    
    def accel_to_thrust(self, a_cmd, m):
        return m * a_cmd