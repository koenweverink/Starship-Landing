# orekit_setup.py
import orekit_jpype as orekit

def init_orekit():
    # Start JVM if not already running
    orekit.initVM()

    # Now that JVM is running, we can safely import helpers that use java.*
    from orekit_jpype.pyhelpers import setup_orekit_curdir

    # This will look for an orekit-data directory or zip in the current working directory
    setup_orekit_curdir()


if __name__ == "__main__":
    init_orekit()
    print("Orekit initialized successfully!")
