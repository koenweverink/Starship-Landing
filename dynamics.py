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
