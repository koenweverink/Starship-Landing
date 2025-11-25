# mars_env.py
import numpy as np
from orekit_setup import init_orekit

# Hard-coded Mars equatorial radius (IAU 2000), in meters
MARS_EQUATORIAL_RADIUS = 3396190.0  # m


def get_mars_params():
    # 1) Ensure JVM + Orekit are initialized
    init_orekit()

    # 2) Import Java classes AFTER init_orekit()
    from org.orekit.bodies import CelestialBodyFactory

    # 3) Get Mars GM from Orekit
    mars = CelestialBodyFactory.getMars()
    mu = mars.getGM()  # m^3/s^2

    # 4) Use our own equatorial radius
    r_eq = MARS_EQUATORIAL_RADIUS

    # 5) Compute surface gravity magnitude (simple spherical approx)
    g0 = mu / (r_eq ** 2)

    # 6) Flat local frame: x,y horizontal, z up; gravity along -z
    return {
        "mu": mu,
        "r_eq": r_eq,
        "g0": g0,
        "g_vec": np.array([0.0, 0.0, -g0]),
    }


if __name__ == "__main__":
    params = get_mars_params()
    print("Mars GM        :", params["mu"])
    print("Mars radius    :", params["r_eq"])
    print("Surface g0     :", params["g0"])
    print("Local g vector :", params["g_vec"])
    print("Mars parameters retrieved successfully!")