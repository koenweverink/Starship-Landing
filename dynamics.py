# dynamics.py
import numpy as np


class LanderDynamics:
    """
    Point-mass lander dynamics with gravity, thrust and simple aerodynamic drag.

    Coordinates:
      r, v are 3D vectors [x, y, z], with z positive upward.
      g_vec should therefore have negative z on Mars (e.g. [0, 0, -3.71]).
    """

    def __init__(
        self,
        g_vec,
        isp,
        thrust_min,
        thrust_max,
        dry_mass,
        cd_area=0.0,
        rho0=0.02,
        h_scale=11000.0,
        inertia_diag=(8.0e6, 8.0e6, 4.0e6),
    ):
        """
        Parameters
        ----------
        g_vec      : array-like, shape (3,)
            Gravity vector [m/s^2].
        isp        : float
            Specific impulse [s].
        thrust_min : float
            Minimum allowed thrust magnitude [N].
        thrust_max : float
            Maximum allowed thrust magnitude [N].
        dry_mass   : float
            Dry mass [kg] (we stop burning if m <= dry_mass).
        cd_area    : float, optional
            C_d * A [m^2] for drag (0 disables drag).
        rho0       : float, optional
            Reference density at z = 0 [kg/m^3].
        h_scale    : float, optional
            Atmospheric scale height [m].
        """
        self.g = np.array(g_vec, dtype=float)
        self.isp = float(isp)
        self.thrust_min = float(thrust_min)
        self.thrust_max = float(thrust_max)
        self.dry_mass = float(dry_mass)

        # Aerodynamic parameters
        self.cd_area = float(cd_area)     # C_d * A
        self.rho0 = float(rho0)
        self.h_scale = float(h_scale)

        # Earth standard gravity for Isp relation
        self.g0_earth = 9.80665

        # Constant diagonal inertia matrix (body frame)
        self.J = np.array(inertia_diag, dtype=float)
        self.J_inv = 1.0 / self.J

    # ------------------------------------------------------------------
    # Quaternion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def quat_normalize(q):
        q = np.asarray(q, dtype=float)
        return q / np.linalg.norm(q)

    @staticmethod
    def quat_multiply(q1, q2):
        """Hamilton product (q1 * q2) with scalar-first convention."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    @staticmethod
    def quat_conjugate(q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    @staticmethod
    def quat_to_dcm(q):
        """Direction cosine matrix (body → inertial) from quaternion."""
        q = LanderDynamics.quat_normalize(q)
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    @staticmethod
    def dcm_to_quat(R):
        """Quaternion (scalar-first) from direction cosine matrix."""
        R = np.asarray(R, dtype=float)
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        q = np.array([w, x, y, z])
        return LanderDynamics.quat_normalize(q)

    # ------------------------------------------------------------------
    # Atmosphere model
    # ------------------------------------------------------------------
    def _atmos_density(self, z):
        """
        Simple exponential atmosphere:
            rho(z) = rho0 * exp(-z / h_scale),  for z >= 0.
        For z < 0 (below "surface") we clamp to rho0.
        """
        if self.cd_area <= 0.0:
            return 0.0

        z_clamped = max(0.0, z)
        return self.rho0 * np.exp(-z_clamped / self.h_scale)

    # ------------------------------------------------------------------
    # Main dynamics step
    # ------------------------------------------------------------------
    def step(self, state, thrust_vec, dt):
        """
        Semi-implicit Euler integration for:
            ṙ = v
            v̇ = g + (T + D) / m
            ṁ = -|T| / (Isp * g0_earth)

        Parameters
        ----------
        state      : tuple (r, v, m)
            r, v : np.array shape (3,)
            m    : scalar mass [kg]
        thrust_vec : np.array shape (3,)
            Desired thrust vector [N] (will be clamped to [thrust_min, thrust_max]).
        dt         : float
            Time step [s].

        Returns
        -------
        (r_new, v_new, m_new) : tuple
            New state after dt.
        a : np.array shape (3,)
            Acceleration used for integration [m/s^2].
        saturated : bool
            True if requested thrust was clamped to min or max.
        """
        r, v, m = state
        r = np.array(r, dtype=float)
        v = np.array(v, dtype=float)
        m = float(m)

        # ---------------- Thrust clamp ----------------
        T = np.linalg.norm(thrust_vec)
        saturated = False

        if T > 0.0:
            if T < self.thrust_min:
                if self.thrust_min > 0.0:
                    thrust_vec = thrust_vec * (self.thrust_min / T)
                    T = self.thrust_min
                    saturated = True
                else:
                    thrust_vec[:] = 0.0
                    T = 0.0
            elif T > self.thrust_max:
                thrust_vec = thrust_vec * (self.thrust_max / T)
                T = self.thrust_max
                saturated = True
        else:
            thrust_vec[:] = 0.0
            T = 0.0

        # ---------------- Mass flow ----------------
        if T > 0.0 and m > self.dry_mass:
            mdot = -T / (self.isp * self.g0_earth)  # [kg/s]
        else:
            # Out of propellant or no thrust: no more burn
            thrust_vec[:] = 0.0
            mdot = 0.0
            T = 0.0

        # ---------------- Aerodynamic drag ----------------
        # Relative velocity = v (no winds yet)
        rho = self._atmos_density(r[2])
        drag_vec = np.zeros(3)
        if self.cd_area > 0.0 and rho > 0.0:
            vmag = np.linalg.norm(v)
            if vmag > 1e-3:
                # F_d = 0.5 * rho * C_d * A * |v|^2 * (-v_hat)
                drag_mag = 0.5 * rho * self.cd_area * vmag**2
                drag_vec = -drag_mag * (v / vmag)

        # ---------------- Acceleration ----------------
        m_eff = max(m, self.dry_mass)
        a = self.g + (thrust_vec + drag_vec) / m_eff

        # ---------------- Integrate (semi-implicit Euler) ----------------
        v_new = v + a * dt
        r_new = r + v_new * dt
        m_new = m + mdot * dt
        if m_new < self.dry_mass:
            m_new = self.dry_mass

        return (r_new, v_new, m_new), a, saturated

    # ------------------------------------------------------------------
    # 6-DOF rigid-body dynamics (translation + attitude)
    # ------------------------------------------------------------------
    def step_6dof(self, state, thrust_body, torque_body, dt):
        """
        Integrate a 6-DOF rigid-body step using semi-implicit Euler.

        State: (r, v, q, w, m)
            r, v : np.array shape (3,) inertial position/velocity [m, m/s]
            q    : quaternion (w, x, y, z) representing body→inertial rotation
            w    : body angular velocity [rad/s]
            m    : mass [kg]

        Control inputs:
            thrust_body : body-frame thrust vector [N]
            torque_body : body-frame control torque [N·m]
        """
        r, v, q, w, m = state
        r = np.array(r, dtype=float)
        v = np.array(v, dtype=float)
        q = self.quat_normalize(np.array(q, dtype=float))
        w = np.array(w, dtype=float)
        m = float(m)

        T_mag = float(np.linalg.norm(thrust_body))
        thrust_body = np.array(thrust_body, dtype=float)
        torque_body = np.array(torque_body, dtype=float)
        saturated = False

        # Clamp thrust magnitude
        if T_mag > 0.0:
            if T_mag < self.thrust_min:
                if self.thrust_min > 0.0:
                    thrust_body = thrust_body * (self.thrust_min / T_mag)
                    T_mag = self.thrust_min
                    saturated = True
                else:
                    thrust_body[:] = 0.0
                    T_mag = 0.0
            elif T_mag > self.thrust_max:
                thrust_body = thrust_body * (self.thrust_max / T_mag)
                T_mag = self.thrust_max
                saturated = True
        else:
            thrust_body[:] = 0.0
            T_mag = 0.0

        # Mass flow
        if T_mag > 0.0 and m > self.dry_mass:
            mdot = -T_mag / (self.isp * self.g0_earth)
        else:
            thrust_body[:] = 0.0
            mdot = 0.0
            T_mag = 0.0

        # Aerodynamic drag
        rho = self._atmos_density(r[2])
        drag_vec = np.zeros(3)
        if self.cd_area > 0.0 and rho > 0.0:
            vmag = np.linalg.norm(v)
            if vmag > 1e-3:
                drag_mag = 0.5 * rho * self.cd_area * vmag**2
                drag_vec = -drag_mag * (v / vmag)

        # Linear acceleration
        R = self.quat_to_dcm(q)  # body -> inertial
        thrust_inertial = R @ thrust_body
        m_eff = max(m, self.dry_mass)
        a = self.g + (thrust_inertial + drag_vec) / m_eff

        # Integrate translation
        v_new = v + a * dt
        r_new = r + v_new * dt

        # Angular dynamics: w_dot = J^{-1}(tau - w x Jw)
        w_cross_Jw = np.cross(w, self.J * w)
        w_dot = (torque_body - w_cross_Jw) * self.J_inv
        w_new = w + w_dot * dt

        # Quaternion kinematics: q_dot = 0.5 * [0, w] * q
        omega_quat = np.array([0.0, *w_new])
        q_dot = 0.5 * self.quat_multiply(q, omega_quat)
        q_new = self.quat_normalize(q + q_dot * dt)

        m_new = m + mdot * dt
        if m_new < self.dry_mass:
            m_new = self.dry_mass

        return (r_new, v_new, q_new, w_new, m_new), a, w_dot, saturated
