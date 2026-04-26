#!/usr/bin/env python3
"""
estimate_noise.py
=================
Estimates KF noise parameters (Rf, Qf) from a baseline balance CSV log.

Usage:
    python3 estimate_noise.py baseline_delay000_jitter000_run1.csv

What it does:
    1. Loads the CSV log from a baseline balance run
    2. Extracts only the BALANCE mode samples with no delay active
    3. Fits a smooth reference trajectory (Savitzky-Golay filter)
    4. Computes residuals = raw measurement - smooth reference
    5. Estimates Rf from variance of position residuals
    6. Estimates Qf from variance of velocity residuals
    7. Prints ready-to-paste C++ constants
"""

import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# ── Load CSV ──────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python3 estimate_noise.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
print(f"\nLoading: {csv_file}")

df = pd.read_csv(csv_file)
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")

# ── Filter: BALANCE mode only, no delay ──────────────────────────────────
bal = df[(df['mode'] == 'BALANCE') & (df['delay_blend'] < 0.01)].copy()
print(f"BALANCE samples (no delay): {len(bal)}")

if len(bal) < 200:
    print("ERROR: Not enough BALANCE samples. Run longer or check CSV.")
    sys.exit(1)

# ── Extract signals ───────────────────────────────────────────────────────
theta_raw = bal['theta_w_rad'].values
alpha_raw = bal['alpha_rad'].values
theta_dot = bal['theta_dot_rad_s'].values
alpha_dot = bal['alpha_dot_rad_s'].values
t         = bal['t_s'].values

# ── Smooth reference (Savitzky-Golay) ────────────────────────────────────
# Window must be odd and less than data length
win = min(51, len(bal) // 4)
if win % 2 == 0:
    win -= 1
poly = 3

theta_smooth     = savgol_filter(theta_raw, win, poly)
alpha_smooth     = savgol_filter(alpha_raw, win, poly)
theta_dot_smooth = savgol_filter(theta_dot, win, poly)
alpha_dot_smooth = savgol_filter(alpha_dot, win, poly)

# ── Residuals ─────────────────────────────────────────────────────────────
res_theta     = theta_raw - theta_smooth
res_alpha     = alpha_raw - alpha_smooth
res_theta_dot = theta_dot - theta_dot_smooth
res_alpha_dot = alpha_dot - alpha_dot_smooth

# ── Measurement noise Rf ──────────────────────────────────────────────────
# Rf is the variance of position measurement residuals
# We use the average of theta and alpha variances since we use scalar Rf
var_theta = np.var(res_theta)
var_alpha = np.var(res_alpha)
Rf_theta  = var_theta
Rf_alpha  = var_alpha
Rf        = (Rf_theta + Rf_alpha) / 2.0   # scalar Rf for both

# ── Process noise Qf ──────────────────────────────────────────────────────
# Qf_pos: variance of position prediction error (approximated from residuals)
# Qf_vel: variance of velocity residuals
var_theta_dot = np.var(res_theta_dot)
var_alpha_dot = np.var(res_alpha_dot)
Qf_pos = (var_theta + var_alpha) / 2.0 * 0.1   # conservative: 10% of Rf
Qf_vel = (var_theta_dot + var_alpha_dot) / 2.0

# ── Encoder quantization reference ───────────────────────────────────────
RAD_PER_COUNT = 2.0 * np.pi / 2048
Rf_encoder    = RAD_PER_COUNT**2 / 12.0   # theoretical quantization noise

# ── Print results ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("NOISE ESTIMATION RESULTS")
print("="*60)
print(f"\nSamples used        : {len(bal)}")
print(f"Smoothing window    : {win} samples")
print(f"\n-- Measurement noise (Rf) --")
print(f"  var(theta_raw)    = {var_theta:.6e} rad²")
print(f"  var(alpha_raw)    = {var_alpha:.6e} rad²")
print(f"  Rf (measured)     = {Rf:.6e} rad²")
print(f"  Rf (encoder quant)= {Rf_encoder:.6e} rad²")
print(f"  Ratio measured/theoretical = {Rf/Rf_encoder:.1f}x")

print(f"\n-- Process noise (Qf) --")
print(f"  var(theta_dot)    = {var_theta_dot:.6e} (rad/s)²")
print(f"  var(alpha_dot)    = {var_alpha_dot:.6e} (rad/s)²")
print(f"  Qf_pos (estimate) = {Qf_pos:.6e}")
print(f"  Qf_vel (estimate) = {Qf_vel:.6e}")

print("\n" + "="*60)
print("COPY THESE INTO controller_process.cpp:")
print("="*60)
print(f"\nstatic const double KF_Rf_encoder  = {Rf_encoder:.6e};  // encoder quantization")
print(f"static const double KF_Rf_measured = {Rf:.6e};  // measured from hardware")
print(f"static const double KF_Qf_pos      = {Qf_pos:.6e};")
print(f"static const double KF_Qf_vel      = {Qf_vel:.6e};")
print(f"\n// In compute_Rf(), replace KF_Rf_encoder with KF_Rf_measured")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f'Noise Estimation — {csv_file}', fontsize=12)

axes[0,0].plot(t, theta_raw, 'b', alpha=0.5, label='raw')
axes[0,0].plot(t, theta_smooth, 'r', linewidth=2, label='smooth')
axes[0,0].set_title('Theta (arm angle)')
axes[0,0].set_ylabel('rad')
axes[0,0].legend()

axes[0,1].plot(t, alpha_raw, 'b', alpha=0.5, label='raw')
axes[0,1].plot(t, alpha_smooth, 'r', linewidth=2, label='smooth')
axes[0,1].set_title('Alpha (pendulum angle)')
axes[0,1].set_ylabel('rad')
axes[0,1].legend()

axes[1,0].plot(t, res_theta, 'g', alpha=0.7)
axes[1,0].axhline(0, color='k', linewidth=0.5)
axes[1,0].set_title(f'Theta residuals  σ²={var_theta:.2e}')
axes[1,0].set_ylabel('rad')
axes[1,0].set_xlabel('time (s)')

axes[1,1].plot(t, res_alpha, 'g', alpha=0.7)
axes[1,1].axhline(0, color='k', linewidth=0.5)
axes[1,1].set_title(f'Alpha residuals  σ²={var_alpha:.2e}')
axes[1,1].set_ylabel('rad')
axes[1,1].set_xlabel('time (s)')

plt.tight_layout()
plot_file = csv_file.replace('.csv', '_noise.png')
plt.savefig(plot_file, dpi=150)
print(f"\nPlot saved: {plot_file}")
plt.show()