#!/usr/bin/env python3
import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are

np.set_printoptions(precision=6, suppress=True)

# =========================
# Parameters (same source as your file)
# =========================
Lp = 0.129
mp = 0.024
Jp = 1.33e-4
lp = Lp / 2.0
Bp = 0.0001

Lr = 0.085
mr = 0.095
Jr = 2.29e-4
Br = 0.001

Rm = 7.5
kt = 0.0422
km = 0.042

g = 9.81

Ts = 0.005          # 200 Hz
Vmax = 10.0

# =========================
# Continuous-time model (upright)
# x = [theta, thetadot, alpha, alphadot]
# =========================
Jr_total = Jr + mp * (Lr ** 2)
D = Jr_total * Jp + Jr_total * mp * (lp ** 2) + Jp * mp * (Lr ** 2)

A_c = np.array([
    [0, 1, 0, 0],
    [0, -(Br*(Jp + mp*lp**2) + (km*kt)*(Jp + mp*lp**2)/Rm)/D,
        (mp**2*g*lp**2*Lr)/D,
        (Bp*mp*lp*Lr)/D],
    [0, 0, 0, 1],
    [0, -(Br*mp*lp*Lr + (km*kt)*mp*lp*Lr/Rm)/D,
        (mp*g*lp*(Jr_total + mp*Lr**2))/D,
       -(Bp*(Jr_total + mp*Lr**2))/D]
], dtype=float)

B_c = np.array([
    [0],
    [(kt*(Jp + mp*lp**2))/(Rm*D)],
    [0],
    [(kt*mp*lp*Lr)/(Rm*D)]
], dtype=float)

C_c = np.eye(4)
D_c = np.zeros((4, 1))

def print_eigs(label, M, discrete=False):
    eigs = np.linalg.eigvals(M)
    print(f"\n{label}")
    for i, eig in enumerate(eigs):
        if discrete:
            print(f"  λ{i+1} = {eig:.6f}, |λ| = {abs(eig):.6f}")
        else:
            print(f"  λ{i+1} = {eig:.6f}")
    return eigs

print("="*70)
print("QUBE-SERVO 3 - Discrete-time LQR design (thesis version)")
print("="*70)
print(f"Ts = {Ts:.4f} s  (Fs = {1.0/Ts:.1f} Hz)")
print(f"Vmax = {Vmax:.1f} V")

print("\nA_c:\n", A_c)
print("\nB_c:\n", B_c.flatten())

print_eigs("Continuous open-loop eigenvalues:", A_c, discrete=False)

# =========================
# Discretize (ZOH)
# =========================
A_d, B_d, _, _, _ = cont2discrete((A_c, B_c, C_c, D_c), Ts, method="zoh")

print("\nA_d:\n", A_d)
print("\nB_d:\n", B_d.flatten())

print_eigs("Discrete open-loop eigenvalues:", A_d, discrete=True)

# Controllability (discrete)
ctrb = np.hstack([B_d, A_d@B_d, A_d@A_d@B_d, A_d@A_d@A_d@B_d])
rank = np.linalg.matrix_rank(ctrb)
print(f"\nControllability rank = {rank} (should be 4)")

# =========================
# Discrete LQR
# =========================
def dlqr(Ad, Bd, Q, R):
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
    Acl = Ad - Bd @ K
    return K, Acl

designs = [
    ("Conservative", np.diag([1, 1, 10, 1]),   np.array([[1.0]])),
    ("Balanced",     np.diag([50, 1, 100, 5]),   np.array([[1.0]])),
    ("Aggressive",   np.diag([10, 5, 200, 20]),np.array([[0.5]])),
]

print("\n" + "="*70)
print("LQR designs @ 200 Hz")
print("="*70)

for name, Q, R in designs:
    K, Acl = dlqr(A_d, B_d, Q, R)
    eigs = np.linalg.eigvals(Acl)
    stable = np.all(np.abs(eigs) < 1.0)
    print(f"\n--- {name} ---")
    print("Q diag =", np.diag(Q))
    print("R =", float(R[0,0]))
    print("K =", K)
    print("Closed-loop eigenvalues magnitudes =", np.abs(eigs))
    print("Stable =", stable)

    k = K.flatten()
    print("C++:")
    print(f"const double K_{name}[4] = {{{k[0]:.6f}, {k[1]:.6f}, {k[2]:.6f}, {k[3]:.6f}}};")

# Simple “saturation risk” hint (very rough)
print("\n" + "="*70)
print("Note")
print("="*70)
print("If your motor saturates frequently at ±10 V during catch, pick a less aggressive design or increase R.")
