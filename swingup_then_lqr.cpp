////////////////////////////////////////////////////////////////////
//
// swingup_then_lqr.cpp
////////////////////////////////////////////////////////////////////
// QUICK START FOR NEW USERS
//
// 1. Compile:
//      g++ -I/usr/include/quanser swingup_then_lqr.cpp -o swingup_then_lqr \-lhil -lquanser_runtime -lquanser_common -lrt -lpthread -ldl -lm
//
// 2. Run:
//    sudo chrt -f 99 taskset -c 3 ./swingup_then_lqr
//
// 3. First time: press C to calibrate
//    - Move arm to the HOME mark on the base
//    - Let pendulum hang straight down
//    - Press Enter when still
//
// 4. After calibration: press R to run
//    - The arm will home automatically
//    - The pendulum will swing up and balance
//    - Press Ctrl+C to stop safely
//
// The pendulum angle (alpha) and arm angle (theta) are logged
// to run.csv every sample (200 Hz) for your analysis.
////////////////////////////////////////////////////////////////////
//
// PUMPING Swing-Up + LQR Balance Control for Qube-Servo 3
//
// Updated Strategy (no manual steps):
// 0. AUTO-HOME: drive arm slowly to physical home target (shortest path), hold briefly
// 1. Initial kick to start pendulum moving
// 2. Rhythmic pumping in sync with pendulum swing
// 3. Auto-switch to LQR when near upright
//
// Startup menu:
//   C = Calibrate (move arm+pendulum to 0/down by hand, press Enter, saves offsets)
//   R = Run       (loads saved offsets from calibration.cfg)
//
// CSV log columns:
//   t_s, dt_s, sample, mode, event,
//   theta_w_rad, theta_dot_rad_s, alpha_rad, alpha_dot_rad_s,
//   voltage_cmd_V, theta_integral
//
////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
#include <signal.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>

#include "quanser_extern.h"
#include "quanser_signal.h"
#include "quanser_messages.h"
#include "hil.h"

#define TIMING_TEST_MODE 0
#define CALIB_FILE "calibration.cfg"

// ───────────────────────────────────────────────────────── LED RGB channels ──────────────────────────────────────────────────────────
const t_uint32 LED_CH[3] = {11000, 11001, 11002};

static inline void set_led(t_card board, double r, double g, double b)
{
    if (r < 0) r = 0; 
    if (r > 1) r = 1;
    if (g < 0) g = 0; 
    if (g > 1) g = 1;
    if (b < 0) b = 0; 
    if (b > 1) b = 1;
    const t_double rgb[3] = {r, g, b};
    hil_write_other(board, LED_CH, 3, rgb);
}

// ────────────────────────────────────────────────────── System parameters ─────────────────────────────────────────────────────────
// Ts sets the control loop rate. Vmax is the hardware voltage saturation limit.
// ENC_RESOLUTION is 2048 counts/rev for the QUBE-Servo 3 encoders.
const double Ts             = 0.005;           // sampling period [s] = 200 Hz
const double Vmax           = 10.0;            // motor voltage saturation [V]
const int    ENC_RESOLUTION = 2048;            // encoder counts per revolution
const double RAD_PER_COUNT  = 2.0 * M_PI / ENC_RESOLUTION;

// ────────────────────────────────────────────────────────── LQR gains ─────────────────────────────────────────────────────────────────
// Computed offline by lqr_design_dt.py using Q=diag(50,1,100,5), R=1.
// State order: x = [theta, theta_dot, alpha, alpha_dot]
// The scale factor reduces aggressiveness at the moment of catch to avoid
// large transient voltages when switching from pump to balance mode.
const double K1[4] = {-6.489113, -3.387435, 93.254482, 11.128319};
const double *K    = K1;
const double LQR_scale = 0.2;  // applied to full LQR output during balance

// ────────────────────────────────────────────────────────── Swing-up parameters ───────────────────────────────────────────────────────
// KICK_DURATION: the initial kick is very short , it only needs to break
// the pendulum from rest so the energy pump can take over immediately.
//
// KICK_VOLTAGE: low enough to avoid saturating, high enough to move the pendulum.
//
// ALPHA_SWITCH_TO_BALANCE: the pendulum must be within 35 deg of upright before
// LQR is activated. Activating too early (large alpha) causes immediate failure.
//
// ALPHA_SWITCH_TO_PUMP: hysteresis band , if alpha grows beyond 40 deg during
// balance, we give up and return to pumping rather than fighting a lost battle.
// This is wider than the entry threshold to avoid rapid switching near the boundary.
const double KICK_DURATION           = 0.2;                    // [s]
const double KICK_VOLTAGE            = 2.0;                    // [V]
const double ALPHA_SWITCH_TO_BALANCE = 35.0 * M_PI / 180.0;   // [rad]
const double ALPHA_SWITCH_TO_PUMP    = 40.0 * M_PI / 180.0;   // [rad]

// ─────────────────────────────────────────────────────────────── Auto-home parameters ──────────────────────────────────────────────────────
// The arm is driven to theta=0 before every swing-up attempt to ensure
// a consistent starting position and repeatable experimental conditions.
// HOME_HOLD_TIME: the arm must stay within tolerance for this long before
// the controller is satisfied , prevents false positives from overshoots.
// HOME_TIMEOUT: if homing takes too long we proceed anyway, since the
// arm may be close enough even if the tolerance was not cleanly met.
// HOME_VMAX and gains are kept conservative to avoid large arm motions.
const double THETA_HOME_TARGET_DEG = 0.0;
const double THETA_HOME_TARGET_RAD = 0.0;
const double HOME_THETA_TOL        = 2.0  * M_PI / 180.0;   // position tolerance [rad]
const double HOME_DOT_TOL          = 10.0 * M_PI / 180.0;   // velocity tolerance [rad/s]
const double HOME_HOLD_TIME        = 1.0;                    // must hold stable for this long [s]
const double HOME_TIMEOUT          = 4.0;                    // give up and kick anyway [s]
const double HOME_VMAX             = 1.5;                    // max homing voltage [V]
const double HOME_KP               = 3.0;                    // proportional gain
const double HOME_KD               = 0.15;                   // derivative gain

// ────────────────────────────────────────────────────────────── Integral term on arm angle ────────────────────────────────────────────────
// A small integral on theta reduces steady-state arm drift during balance.
// The integrator state is clamped to avoid windup if the arm drifts far.
const double Ki_theta = -0.4;

// ────────────────────────────────────────────────────────────────────────── Globals ───────────────────────────────────────────────────────────────────
static int    stop_flag      = 0;
static double theta_integral = 0.0;

static void signal_handler(int) { stop_flag = 1; }

// ─────────────────────────────────────────────────────────────────── Control mode state machine ────────────────────────────────────────────────
// Transitions: HOME → KICK → PUMP ↔ BALANCE
// The PUMP ↔ BALANCE transition uses a hysteresis band (different alpha
// thresholds for entry and exit) to avoid chattering near the boundary.
enum ControlMode { MODE_HOME, MODE_KICK, MODE_PUMP, MODE_BALANCE };

static const char* mode_str(ControlMode m)
{
    switch (m) {
        case MODE_HOME:    return "HOME";
        case MODE_KICK:    return "KICK";
        case MODE_PUMP:    return "PUMP";
        case MODE_BALANCE: return "BALANCE";
    }
    return "UNKNOWN";
}

// ─────────────────────────────────────────────────────────── Utility functions ─────────────────────────────────────────────────────────
static inline double timespec_to_sec(const timespec &t)
{
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static inline double clamp(double x, double lo, double hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// wrap_pi: maps any angle to the range (-pi, pi]
// Used to ensure theta error and alpha are always in the shortest-path range.
static inline double wrap_pi(double x)
{
    return atan2(sin(x), cos(x));
}

static double sign(double x)
{
    if (x > 0) return  1.0;
    if (x < 0) return -1.0;
    return 0.0;
}

// ─────────────────────────────────────────────────────────────────── Velocity estimator ────────────────────────────────────────────────────────
// Estimates angular velocity from encoder position using finite differences
// smoothed through a first-order low-pass filter (default alpha=0.85).
// The filter reduces noise from encoder quantization at 200 Hz.
class VelocityEstimator
{
private:
    double prev_position;
    double prev_velocity;
    double alpha_filt;

public:
    explicit VelocityEstimator(double filter_alpha = 0.85)
        : prev_position(0.0), prev_velocity(0.0), alpha_filt(filter_alpha) {}

    void reset(double initial_position)
    {
        prev_position = initial_position;
        prev_velocity = 0.0;
    }

    double update(double position, double dt)
    {
        if (dt <= 1e-6) dt = Ts;
        double velocity_raw = (position - prev_position) / dt;
        double velocity = alpha_filt * prev_velocity + (1.0 - alpha_filt) * velocity_raw;
        prev_position = position;
        prev_velocity = velocity;
        return velocity;
    }
};

// ──────────────────────────────────────────────────────────────────────── Calibration ───────────────────────────────────────────────────────────────
// Offsets are recorded once with the arm at the home mark and the pendulum
// hanging down, then saved to file. At runtime the offsets are loaded so
// that theta=0 maps to the home mark and alpha=0 maps to the upright position.
static bool save_calibration(t_int32 theta_offset, t_int32 alpha_offset)
{
    FILE *f = fopen(CALIB_FILE, "w");
    if (!f) { perror("fopen calibration.cfg"); return false; }
    fprintf(f, "theta_offset=%d\n", theta_offset);
    fprintf(f, "alpha_offset=%d\n", alpha_offset);
    fclose(f);
    printf("Calibration saved to %s  (theta_off=%d  alpha_off=%d)\n",
           CALIB_FILE, theta_offset, alpha_offset);
    return true;
}

static bool load_calibration(t_int32 &theta_offset, t_int32 &alpha_offset)
{
    FILE *f = fopen(CALIB_FILE, "r");
    if (!f) {
        printf("WARNING: %s not found. Run calibration first (press C).\n", CALIB_FILE);
        return false;
    }
    if (fscanf(f, "theta_offset=%d\nalpha_offset=%d\n",
               &theta_offset, &alpha_offset) != 2) {
        printf("WARNING: %s is corrupted. Re-calibrate (press C).\n", CALIB_FILE);
        fclose(f);
        return false;
    }
    fclose(f);
    printf("Calibration loaded: theta_off=%d  alpha_off=%d\n",
           theta_offset, alpha_offset);
    return true;
}

// ────────────────────────────────────────────────────────────── Angle decoding ────────────────────────────────────────────────────────────
// Converts raw encoder counts to calibrated angles in radians.
// After calibration: theta=0 at home mark, alpha=0 at upright position.
// Alpha is wrapped to (-pi, pi] so the upright is always at 0 and the
// downward hanging position is at ±pi.
static void decode_angles(const t_int32 counts[2],
                          t_int32 theta_offset, t_int32 alpha_offset,
                          double &theta_out, double &alpha_out)
{
    theta_out = (counts[0] - theta_offset) * RAD_PER_COUNT;

    double alpha_raw = (counts[1] - alpha_offset) * RAD_PER_COUNT;
    alpha_raw = fmod(alpha_raw, 2*M_PI);
    if (alpha_raw < -M_PI) alpha_raw += 2*M_PI;
    if (alpha_raw >  M_PI) alpha_raw -= 2*M_PI;
    alpha_out = alpha_raw;
}

// ───────────────────────────────────────────────────────────────── Home controller ───────────────────────────────────────────────────────────
// Simple PD controller to drive the arm to theta=0 before swing-up.
// Uses conservative gains and a low voltage limit to avoid aggressive motion.
static double home_control(double theta_err_wrapped, double theta_dot)
{
    double u = -(HOME_KP * theta_err_wrapped + HOME_KD * theta_dot);
    return clamp(u, -HOME_VMAX, HOME_VMAX);
}

// ────────────────────────────────────────────────────────────────── Kick controller ───────────────────────────────────────────────────────────
// Applies a brief alternating voltage to break the pendulum from rest.
// The period matches a rough estimate of the pendulum's natural period so
// the kick adds energy in the right direction from the start.
static double kick_control(double /*theta*/, double kick_time)
{
    const double period = 1.0;  // [s]
    double phase = fmod(kick_time, period);
    return (phase < period / 2.0) ? KICK_VOLTAGE : -KICK_VOLTAGE;
}

// ───────────────────────────────────────────────────────────────────── Pump controller ───────────────────────────────────────────────────────────
// Energy-pumping swing-up: applies torque in the direction that increases
// the pendulum's kinetic energy at the top of each half-swing.
//
// The two threshold conditions prevent pumping when:
//   alpha < 120 deg: pendulum is too far from vertical to pump effectively
//   alpha_dot < 100 deg/s: pendulum is moving too slowly, pumping would waste energy
//
// When both conditions are met, a constant torque is applied in the direction
// of -sign(alpha * alpha_dot), which always adds energy to the swing.
// No cable protection is needed here since this file does not use the
// distributed two-process architecture — theta is not constrained in this version.
static double pump_control(double alpha, double alpha_dot)
{
    double alpha_deg     = fabs(alpha)     * 180.0 / M_PI;
    double alpha_dot_deg = fabs(alpha_dot) * 180.0 / M_PI;

    // only pump when pendulum is near vertical and moving fast enough
    if (alpha_deg     < 120.0) return 0.0;
    if (alpha_dot_deg < 100.0) return 0.0;

    double u = -8.0 * sign(alpha * alpha_dot);
    return clamp(u, -Vmax, Vmax);
}

// ────────────────────────────────────────────────────────────── LQR balance controller ────────────────────────────────────────────────────
// Discrete-time LQR around the upright equilibrium.
// Alpha sign is negated to match the encoder convention (upright = 0,
// downward = ±pi) against the linearised model convention used during design.
// A scale factor reduces the output at the moment of catch to avoid
// large transient voltages when switching from pump to balance mode.
static double lqr_control(double theta_w, double theta_dot,
                           double alpha,   double alpha_dot)
{
    double u = -LQR_scale * (  K[0]*theta_w
                              + K[1]*theta_dot
                              + K[2]*(-alpha)
                              + K[3]*(-alpha_dot) );
    return clamp(u, -Vmax, Vmax);
}

// ─────────────────────────────────────────────────────────────────── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    (void)argc; (void)argv;

    static const char board_type[]       = "qube_servo3_usb";
    static const char board_identifier[] = "0";
    static char message[512];

    qsigaction_t action;
    action.sa_handler = signal_handler;
    action.sa_flags   = 0;
    qsigemptyset(&action.sa_mask);
    qsigaction(SIGINT, &action, NULL);

    char log_filename[64];
    snprintf(log_filename, sizeof(log_filename), "run.csv");

    printf("========================================\n");
    printf("QUBE-SERVO 3 PUMPING SWING-UP CONTROLLER\n");
    printf("========================================\n");
    printf("Log: %s\n\n", log_filename);

    // ────────────────────────────────────────────────────────── Startup menu ──────────────────────────────────────────────────────────
    t_int32 theta_offset = 0;
    t_int32 alpha_offset = 0;

    while (true)
    {
        printf("Press C to Calibrate, R to Run: ");
        fflush(stdout);

        char choice = 0;
        while (choice != 'c' && choice != 'C' && choice != 'r' && choice != 'R')
        {
            int ch = getchar();
            if (ch == EOF) { printf("\nEOF, exiting.\n"); return 0; }
            choice = (char)ch;
        }
        printf("\n");

        if (choice == 'c' || choice == 'C')
        {
            t_card board_cal;
            t_int result = hil_open(board_type, board_identifier, &board_cal);
            if (result != 0) {
                msg_get_error_message(NULL, result, message, sizeof(message));
                printf("ERROR opening board: %s\n", message);
                continue;
            }

            const t_uint32 enc_ch[] = {0, 1};
            t_int32 cal_counts[2]   = {0, 0};

            printf("\n=== CALIBRATION ===\n");
            printf("1. Move arm to HOME mark (theta = 0)\n");
            printf("2. Let pendulum hang DOWN freely\n");
            printf("3. Hold still and press Enter...\n");

            { int ch; while ((ch = getchar()) != '\n' && ch != EOF); }
            { int ch; while ((ch = getchar()) != '\n' && ch != EOF); }

            result = hil_read_encoder(board_cal, enc_ch, 2, cal_counts);
            if (result != 0) {
                msg_get_error_message(NULL, result, message, sizeof(message));
                printf("ERROR reading encoders: %s\n", message);
                hil_close(board_cal);
                continue;
            }

            printf("Raw counts at calibration: theta=%d  alpha=%d\n",
                   cal_counts[0], cal_counts[1]);

            // arm at home mark → theta should read 0
            theta_offset = cal_counts[0];
            // pendulum hanging down → alpha should read ±pi, offset shifts it so alpha=0 is upright
            alpha_offset = cal_counts[1] + ENC_RESOLUTION / 2;

            save_calibration(theta_offset, alpha_offset);
            hil_close(board_cal);
            printf("Calibration done!\n\n");
        }
        else
        {
            if (!load_calibration(theta_offset, alpha_offset)) {
                printf("Cannot run without calibration. Please calibrate first (C).\n\n");
                continue;
            }
            break;
        }
    }

    // ───────────────────────────────────────────── Open board for control ────────────────────────────────────────────────
    t_card board;
    t_int result = hil_open(board_type, board_identifier, &board);
    if (result != 0) {
        msg_get_error_message(NULL, result, message, sizeof(message));
        printf("ERROR: %s\n", message);
        return -1;
    }
    printf("Board opened.\n");
    set_led(board, 1.0, 0.0, 0.0);  // red: board open, not yet ready

    const t_uint32 encoder_channels[] = {0, 1};
    const t_uint32 analog_channel     = 0;
    const t_uint32 digital_channel    = 0;

#define NUM_ENCODER_CHANNELS 2
    t_int32   encoder_counts[NUM_ENCODER_CHANNELS];
    t_double  voltage = 0.0;
    t_boolean enable  = 1;
    t_task    task;

    result = hil_write_digital(board, &digital_channel, 1, &enable);
    if (result != 0) {
        msg_get_error_message(NULL, result, message, sizeof(message));
        printf("ERROR: %s\n", message);
        hil_close(board);
        return -1;
    }
    printf("Motor enabled.\n");
    printf("Home target: %.1f deg | Kick: %.1fV for %.1fs | Ts: %.3fs\n\n",
           THETA_HOME_TARGET_DEG, KICK_VOLTAGE, KICK_DURATION, Ts);

    const t_double frequency = 1.0 / Ts;
    result = hil_task_create_encoder_reader(board, frequency,
                                            encoder_channels,
                                            NUM_ENCODER_CHANNELS, &task);
    if (result != 0) {
        msg_get_error_message(NULL, result, message, sizeof(message));
        printf("ERROR: %s\n", message);
        goto shutdown_no_task;
    }

    result = hil_task_start(task, HARDWARE_CLOCK_0, frequency, -1);
    if (result != 0) {
        msg_get_error_message(NULL, result, message, sizeof(message));
        printf("ERROR: %s\n", message);
        hil_task_delete(task);
        goto shutdown_no_task;
    }

    // ──────────────────────────────────────────────────── Main control loop ─────────────────────────────────────────────────────
    {
        FILE *log = fopen(log_filename, "w");
        if (!log) { perror("fopen log"); goto shutdown; }
        static char logbuf[1 << 20];
        setvbuf(log, logbuf, _IOFBF, sizeof(logbuf));

        fprintf(log, "t_s,dt_s,sample,mode,event,"
                     "theta_w_rad,theta_dot_rad_s,"
                     "alpha_rad,alpha_dot_rad_s,"
                     "voltage_cmd_V,theta_integral\n");

        // Read initial encoder state before entering the loop
        double theta = 0.0, alpha = 0.0;
        result = hil_task_read_encoder(task, 1, encoder_counts);
        if (result > 0)
            decode_angles(encoder_counts, theta_offset, alpha_offset, theta, alpha);

        // Velocity estimators must be seeded with the initial position
        // so the first velocity estimate is zero rather than a large spike
        VelocityEstimator theta_vel_est(0.85);
        VelocityEstimator alpha_vel_est(0.85);
        theta_vel_est.reset(theta);
        alpha_vel_est.reset(alpha);

        timespec t_prev, t_now;
        clock_gettime(CLOCK_MONOTONIC, &t_prev);

        set_led(board, 0.0, 0.0, 1.0);  // blue: homing in progress
        printf("=== AUTO-HOME: driving theta to 0 ===\n");

        ControlMode mode        = MODE_HOME;
        int    samples_read     = 0;
        int    mode_switches    = 0;
        double elapsed_time     = 0.0;
        double home_stable_time = 0.0;
        double home_time        = 0.0;
        const char *event       = "";

        while (!stop_flag)
        {
            result = hil_task_read_encoder(task, 1, encoder_counts);

            if (result > 0)
            {
                clock_gettime(CLOCK_MONOTONIC, &t_now);
                double dt_s = timespec_to_sec(t_now) - timespec_to_sec(t_prev);
                t_prev = t_now;
                if (dt_s < 1e-6) dt_s = Ts;
                if (dt_s > 0.05)  dt_s = 0.05;  // cap for robustness

                samples_read++;
                elapsed_time += dt_s;

                decode_angles(encoder_counts, theta_offset, alpha_offset,
                              theta, alpha);

                double theta_w   = wrap_pi(theta);
                double theta_dot = theta_vel_est.update(theta,   dt_s);
                double alpha_dot = alpha_vel_est.update(alpha,   dt_s);
                double theta_err = wrap_pi(theta_w - THETA_HOME_TARGET_RAD);

                // ────────────────────────────────────────── Mode switching ────────────────────────────────────────────
                event = "";

                if (mode == MODE_HOME)
                {
                    home_time += dt_s;

                    // Track how long the arm has been within the tolerance band
                    if (fabs(theta_err) < HOME_THETA_TOL &&
                        fabs(theta_dot) < HOME_DOT_TOL)
                        home_stable_time += dt_s;
                    else
                        home_stable_time = 0.0;

                    // Exit homing when stable long enough OR timeout expires
                    if (home_stable_time >= HOME_HOLD_TIME ||
                        home_time        >= HOME_TIMEOUT)
                    {
                        if (home_time >= HOME_TIMEOUT)
                        {
                            event = "HOME_TIMEOUT_TO_KICK";
                            printf("\n>>> HOME TIMEOUT: starting KICK anyway <<<\n\n");
                        }
                        else
                        {
                            event = "HOME_OK_TO_KICK";
                            printf("\n>>> HOME OK: starting KICK <<<\n\n");
                        }
                        mode = MODE_KICK;
                        mode_switches++;
                        elapsed_time = 0.0;  // reset timer for kick phase
                        theta_vel_est.reset(theta);
                        alpha_vel_est.reset(alpha);
                        theta_integral = 0.0;
                        set_led(board, 0.0, 1.0, 0.0);  // green: running
                    }
                }
                else if (mode == MODE_KICK)
                {
                    // Exit kick after the fixed duration — pump takes over
                    if (elapsed_time >= KICK_DURATION)
                    {
                        event = "KICK_TO_PUMP";
                        mode  = MODE_PUMP;
                        mode_switches++;
                        printf("\n>>> PHASE 2: PUMPING <<<\n\n");
                    }
                }
                else if (mode == MODE_PUMP && fabs(alpha) < ALPHA_SWITCH_TO_BALANCE)
                {
                    // Pendulum has reached the upright region — switch to LQR
                    event = "PUMP_TO_BALANCE";
                    mode  = MODE_BALANCE;
                    theta_vel_est.reset(theta);
                    alpha_vel_est.reset(alpha);
                    theta_integral = 0.0;
                    mode_switches++;
                    printf("\n>>> PHASE 3: BALANCE (a=%.1f deg) <<<\n\n",
                           alpha * 180.0/M_PI);
                }
                else if (mode == MODE_BALANCE && fabs(alpha) > ALPHA_SWITCH_TO_PUMP)
                {
                    // Pendulum fell — return to pumping to try again
                    event = "BALANCE_TO_PUMP";
                    mode  = MODE_PUMP;
                    theta_integral = 0.0;
                    printf("\n>>> BACK TO PUMPING <<<\n\n");
                }

                // ───────────────────────────────────── Control output ────────────────────────────────────────────
                if (mode == MODE_HOME)
                {
                    voltage = home_control(theta_err, theta_dot);
                }
                else if (mode == MODE_KICK)
                {
                    voltage = kick_control(theta, elapsed_time);
                }
                else if (mode == MODE_PUMP)
                {
                    voltage = pump_control(alpha, alpha_dot);
                }
                else  // MODE_BALANCE
                {
                    // LQR output plus integral term on arm angle to reduce drift
                    theta_integral += theta_w * dt_s;
                    theta_integral  = clamp(theta_integral, -3.0, 3.0);

                    voltage  = lqr_control(theta_w, theta_dot, alpha, alpha_dot);
                    voltage -= Ki_theta * theta_integral;
                    voltage  = clamp(voltage, -Vmax, Vmax);
                }

                result = hil_write_analog(board, &analog_channel, 1, &voltage);
                if (result != 0) {
                    msg_get_error_message(NULL, result, message, sizeof(message));
                    printf("ERROR: %s\n", message);
                    break;
                }

                // ─────────────────────────── CSV logging ───────────────────────────────
                fprintf(log, "%.6f,%.6f,%d,%s,%s,"
                             "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                        elapsed_time, dt_s, samples_read,
                        mode_str(mode), event,
                        theta_w, theta_dot,
                        alpha,   alpha_dot,
                        (double)voltage, theta_integral);

#if !TIMING_TEST_MODE
                // Print status every 100 samples (~0.5s)
                if (samples_read % 100 == 0)
                {
                    double th_deg   = theta_w  * 180.0/M_PI;
                    double a_deg    = alpha     * 180.0/M_PI;
                    double adot_deg = alpha_dot * 180.0/M_PI;

                    if (mode == MODE_HOME)
                        printf("HOME   | th:%7.1f  a:%7.1f  u:%6.2f V  err:%6.2fdeg hold:%.2fs\n",
                               th_deg, a_deg, (double)voltage,
                               theta_err*180.0/M_PI, home_stable_time);
                    else if (mode == MODE_KICK)
                        printf("KICK   | th:%7.1f  a:%7.1f  u:%6.2f V  t:%.2fs\n",
                               th_deg, a_deg, (double)voltage, elapsed_time);
                    else if (mode == MODE_PUMP)
                        printf("PUMP   | th:%7.1f  a:%7.1f  adot:%8.1f  u:%6.2f V\n",
                               th_deg, a_deg, adot_deg, (double)voltage);
                    else
                        printf("BALANCE| th:%7.1f  a:%7.1f  u:%6.2f V  I:%6.3f\n",
                               th_deg, a_deg, (double)voltage, theta_integral);

                    fflush(stdout);
                }
#endif
            }
            else if (result < 0)
            {
                msg_get_error_message(NULL, result, message, sizeof(message));
                printf("ERROR: %s\n", message);
                break;
            }
        }

        printf("\n=== SHUTTING DOWN ===\n");
        printf("  Mode switches: %d\n", mode_switches);
        printf("  Total samples: %d\n", samples_read);
        printf("  Run time:      %.1f s\n", elapsed_time);
        fclose(log);
        printf("Log saved: %s\n", log_filename);
    }

shutdown:
    voltage = 0.0;
    hil_write_analog(board, &analog_channel, 1, &voltage);
    hil_task_stop(task);
    hil_task_delete(task);

shutdown_no_task:
    enable = 0;
    hil_write_digital(board, &digital_channel, 1, &enable);
    set_led(board, 1.0, 0.0, 0.0);  // red: shutdown
    hil_close(board);
    printf("Done.\n");
    return 0;
}
