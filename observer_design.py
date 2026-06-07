#!/usr/bin/env python3
"""
observer_design.py

Computes the observer gain matrix L for the Smith Predictor
used in the distributed Furuta pendulum controller.

The observer is designed by discrete-time pole placement.
The closed-loop observer matrix is:
    A_obs = Ad - L * C

Desired observer poles are chosen to be real, strictly inside
the unit circle, and faster than the closed-loop plant poles
so that the observer converges before the controller acts.

State vector: x = [theta, theta_dot, alpha, alpha_dot]
Measurement:  y = [theta, alpha]  (positions only)

Output:
    L matrix (4x2) ready to paste into controller_process.cpp
"""

import numpy as np
from scipy.signal import place_poles

# ============================================================
# Discrete-time plant model (ZOH, Ts = 0.005 s, 200 Hz)
# Derived from lqr_design_dt.py with same physical parameters
# ============================================================
Ad = np.array([
    [1.0000000000,  0.0049693034,  0.0002132503,  0.0000017572],
    [0.0000000000,  0.9877456579,  0.0851735824,  0.0007723907],
    [0.0000000000, -0.0000173327,  1.0009349977,  0.0049953952],
    [0.0000000000, -0.0069170964,  0.3738049298,  0.9984695615]
])

# Measurement matrix: y = [theta, alpha]
C = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

# ============================================================
# Desired observer poles
# All real, strictly inside the unit circle.
# Chosen faster than closed-loop plant poles for convergence.
# ============================================================
#desired_poles = [0.5, 0.45, 0.4, 0.35]
desired_poles = [0.35, 0.30, 0.25, 0.20]
 

# ============================================================
# Pole placement (dual system: place poles of Ad' - C'*L')
# ============================================================
result = place_poles(Ad.T, C.T, desired_poles)
L = result.gain_matrix.T

print("=" * 60)
print("OBSERVER GAIN MATRIX L  (4x2)")
print("=" * 60)
print(L)
print()

# Verify closed-loop observer eigenvalues
Ao = Ad - L @ C
eigs = np.linalg.eigvals(Ao)
print("Observer closed-loop eigenvalues:")
for e in eigs:
    print(f"  lambda = {e:.8f}   |lambda| = {abs(e):.8f}")

print()
print("All inside unit circle:", np.all(np.abs(eigs) < 1.0))

# ============================================================
# C++ output
# ============================================================
print()
print("=" * 60)
print("C++ array (paste into controller_process.cpp):")
print("=" * 60)
print("static const double SP_L[4][2] = {")
for row in L:
    print(f"    {{ {row[0]:.8f},  {row[1]:.8f}}},")
print("};")