// controller_process.cpp
//
// Distributed LQR controller for the Quanser QUBE-Servo 3 Furuta pendulum.
//
// This process receives encoder measurements from hardware_process over UDP,
// runs one of three control strategies (baseline, kf, sp), injects artificial
// sensor delay and jitter into the feedback path during the balance phase,
// and sends voltage commands back to hardware_process over UDP.
//
// The three strategies share the same LQR gain and swing-up logic.
// They differ only in how delayed sensor feedback is handled:
//   baseline : raw delayed measurement fed directly to LQR
//   kf       : Kalman filter with delayed-measurement replay
//   sp       : Smith Predictor observer-predictor
//
// Delay and jitter are injected in software using a circular buffer.
// Jitter is modeled as a bounded, temporally correlated Gaussian process.
// Injection starts 6 seconds after balance is established so the
// controller is already stable before the delay condition is applied.
//
// All state, metrics, and mode transitions are logged to a CSV file
// named: <strategy>_delay<ms>_jitter<ms>_run<N>.csv
//
// Usage:
//   ./controller_process --delay 20 --jitter 5 --strategy kf --run 1
//   ./controller_process --delay 0  --strategy baseline --run 1
//   [--host 127.0.0.1]

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <cstdlib>

// ── Network ports ─────────────────────────────────────────────────────────────
#define SENSOR_PORT          9001   // receive encoder data from hardware_process
#define CMD_PORT             9002   // send voltage commands to hardware_process
#define DELAY_BUFFER_MAX_STEPS 200  // max delay buffer size (200 * 5ms = 1000ms)

// ── System constants ──────────────────────────────────────────────────────────
static const double Ts   = 0.005;  // sampling period [s] = 200 Hz
static const double Vmax = 10.0;   // motor voltage saturation [V]

// ── LQR gains ─────────────────────────────────────────────────────────────────
// Computed offline by lqr_design_dt.py using Q=diag(50,1,100,5), R=1.
// State order: x = [theta, theta_dot, alpha, alpha_dot]
// Control law: u = -scale * K * x  (alpha sign flipped inside lqr_control)
static const double K[4] = {-6.489113, -3.387435, 93.254482, 11.128319};

// Scale factors applied to LQR output during balance.
// A soft scale is used for the first second after catch to avoid
// aggressive transients; then the balance scale takes over.
static const double LQR_scale_swingup  = 0.2;
static const double LQR_SCALE_SWITCH_S = 1.0;
static const double LQR_scale_balance  = 0.11;

// ── Swing-up parameters ───────────────────────────────────────────────────────
static const double KICK_DURATION           = 0.2;
static const double KICK_VOLTAGE            = 2.0;
static const double ALPHA_SWITCH_TO_BALANCE = 35.0 * M_PI / 180.0;
static const double ALPHA_SWITCH_TO_PUMP    = 40.0 * M_PI / 180.0;

// ── Auto-home parameters ──────────────────────────────────────────────────────
// The arm is driven to theta=0 before swing-up begins.
static const double THETA_HOME_TARGET_RAD = 0.0;
static const double HOME_THETA_TOL  = 6.0  * M_PI / 180.0;
static const double HOME_DOT_TOL    = 10.0 * M_PI / 180.0;
static const double HOME_HOLD_TIME  = 1.0;
static const double HOME_TIMEOUT    = 5.0;
static const double HOME_VMAX       = 3.0;
static const double HOME_KP         = 8.0;
static const double HOME_KD         = 0.3;

// ── Integral term on arm angle ────────────────────────────────────────────────
// Reduces steady-state arm drift. Clamped to avoid windup.
static const double Ki_theta = -0.4;

// ── Delay injection timing ────────────────────────────────────────────────────
// Delay is injected this many seconds after balance is established,
// giving the controller time to stabilize before the experiment begins.
static const double BASELINE_DELAY_INJECT_S = 6.0;
static const double SP_DELAY_INJECT_S       = 6.0;
static const double KF_ACTIVATE_S           = 5.0;
static const double KF_DELAY_INJECT_S       = 6.0;
static const double DELAY_BLEND_DURATION_S  = 0.25;

// ── Kalman filter noise parameters ───────────────────────────────────────────
// Rf measured experimentally from hardware sensor logs over UDP loopback.
// Qf_vel is larger because angular velocity is not directly measured.
// When jitter is active, Rf is increased proportionally to the realized offset.
static const double KF_Rf_base     = 4.626972e-06;
static const double KF_Qf_pos      = 4.626972e-07;
static const double KF_Qf_vel      = 2.955488e-03;
static const double KF_MAX_ANG_VEL = 15.0;

// Compute measurement noise covariance based on realized jitter offset.
// Larger jitter offset means the measurement age is more uncertain,
// so Rf is increased to reduce trust in that measurement.
static inline double compute_Rf_from_offset_steps(int jitter_offset_steps) {
    double jitter_std_s = fabs((double)jitter_offset_steps) * Ts;
    double sigma_angle  = jitter_std_s * KF_MAX_ANG_VEL;
    double Rf = KF_Rf_base + sigma_angle * sigma_angle;
    if (Rf < KF_Rf_base) Rf = KF_Rf_base;
    if (Rf > 1.0e-3)     Rf = 1.0e-3;
    return Rf;
}

// ── Discrete-time plant model (ZOH, Ts=0.005s) ───────────────────────────────
// Derived from lqr_design_dt.py using the physical parameters in Table 1.
// Used by both the KF predict step and the SP observer propagation.
static const double Ad[4][4] = {
    {1.0000000000,  0.0049693034,  0.0002132503,  0.0000017572},
    {0.0000000000,  0.9877456579,  0.0851735824,  0.0007723907},
    {0.0000000000, -0.0000173327,  1.0009349977,  0.0049953952},
    {0.0000000000, -0.0069170964,  0.3738049298,  0.9984695615}
};
static const double Bd[4] = {
    0.0001397044, 0.0557712390, 0.0000788836, 0.0314806814
};

// ── Smith Predictor observer gain matrix ─────────────────────────────────────
// L is a 4x2 matrix mapping position errors [theta_err, alpha_err] to
// state corrections. Computed offline by observer_design.py using pole
// placement with desired observer poles at {0.75, 0.70, 0.65, 0.60}.
static const double SP_L[4][2] = {
    { 0.63497831,  0.04879646},
    {19.13434494,  3.18368196},
    { 0.04521185,  0.65217191},
    { 2.56718512, 21.07001744}
};

// ── Globals ───────────────────────────────────────────────────────────────────
static volatile sig_atomic_t stop_flag = 0;
static double theta_integral = 0.0;

// ── UDP packet structures ─────────────────────────────────────────────────────
// Packed to ensure identical memory layout between processes.
#pragma pack(push, 1)
struct SensorPacket {
    uint32_t seq;
    double   timestamp;
    double   theta_raw;
    double   alpha_raw;
    int32_t  theta_counts;
    int32_t  alpha_counts;
    int32_t  theta_offset;
    int32_t  alpha_offset;
};

struct CmdPacket {
    uint32_t seq;
    double   timestamp;
    double   voltage;
};
#pragma pack(pop)

// ── Control mode state machine ────────────────────────────────────────────────
enum ControlMode { MODE_HOME, MODE_KICK, MODE_PUMP, MODE_BALANCE };

static inline void on_signal(int) { stop_flag = 1; }

static const char* mode_str(ControlMode m) {
    switch (m) {
        case MODE_HOME:    return "HOME";
        case MODE_KICK:    return "KICK";
        case MODE_PUMP:    return "PUMP";
        case MODE_BALANCE: return "BALANCE";
        default:           return "UNKNOWN";
    }
}

// ── Utility functions ─────────────────────────────────────────────────────────
static inline double timespec_to_sec(const struct timespec &t) {
    return (double)t.tv_sec + 1e-9*(double)t.tv_nsec;
}
static inline double clamp(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
static inline double wrap_pi(double x) { return atan2(sin(x), cos(x)); }
static inline double sign_f(double x) {
    return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
}

// ── Velocity estimator ────────────────────────────────────────────────────────
// Estimates angular velocity by finite-differencing encoder positions
// and smoothing with a first-order low-pass filter (coefficient 0.85).
class VelocityEstimator {
    double prev_pos_, prev_vel_, alpha_;
    bool initialized_;
public:
    explicit VelocityEstimator(double a = 0.85)
        : prev_pos_(0.0), prev_vel_(0.0), alpha_(a), initialized_(false) {}

    void reset(double pos) {
        prev_pos_ = pos; prev_vel_ = 0.0; initialized_ = true;
    }

    double update(double pos, double dt) {
        if (!initialized_) { reset(pos); return 0.0; }
        if (dt <= 1e-6) dt = Ts;
        double raw = (pos - prev_pos_) / dt;
        double vel = alpha_ * prev_vel_ + (1.0 - alpha_) * raw;
        prev_pos_ = pos; prev_vel_ = vel;
        return vel;
    }

    double last() const { return prev_vel_; }
};

// ── Delay buffer ──────────────────────────────────────────────────────────────
// Circular buffer storing past sensor measurements.
// Reading at depth d returns the sample from d steps ago,
// simulating a network delay of d * Ts seconds.
struct SensorSample { double theta, alpha; };

class DelayBuffer {
    SensorSample buf_[DELAY_BUFFER_MAX_STEPS];
    int write_idx_, delay_steps_;
public:
    DelayBuffer() : write_idx_(0), delay_steps_(0) {
        for (int i = 0; i < DELAY_BUFFER_MAX_STEPS; i++)
            buf_[i].theta = buf_[i].alpha = 0.0;
    }

    void set_delay_steps(int s) {
        if (s < 0) s = 0;
        if (s >= DELAY_BUFFER_MAX_STEPS) s = DELAY_BUFFER_MAX_STEPS - 1;
        delay_steps_ = s;
    }

    void prefill(double th, double al) {
        for (int i = 0; i < DELAY_BUFFER_MAX_STEPS; i++)
            buf_[i] = {th, al};
        write_idx_ = 0;
    }

    void push(double th, double al) {
        buf_[write_idx_] = {th, al};
        write_idx_ = (write_idx_ + 1) % DELAY_BUFFER_MAX_STEPS;
    }

    SensorSample read_at_depth(int depth) const {
        int n = DELAY_BUFFER_MAX_STEPS;
        int d = (depth < 0) ? 0 : (depth >= n ? n-1 : depth);
        return buf_[(write_idx_ - 1 - d + n*2) % n];
    }
};

// ── Voltage history buffer ────────────────────────────────────────────────────
// Stores past control voltages so KF and SP can replay them
// when forward-predicting through the delay window.
class VoltageHistory {
    double buf_[DELAY_BUFFER_MAX_STEPS];
    int write_idx_;
public:
    VoltageHistory() : write_idx_(0) {
        for (int i = 0; i < DELAY_BUFFER_MAX_STEPS; i++) buf_[i] = 0.0;
    }

    void push(double v) {
        buf_[write_idx_] = v;
        write_idx_ = (write_idx_ + 1) % DELAY_BUFFER_MAX_STEPS;
    }

    double get_past(int n) const {
        int nn = (n < 0) ? 0 : (n >= DELAY_BUFFER_MAX_STEPS ? DELAY_BUFFER_MAX_STEPS-1 : n);
        return buf_[(write_idx_ - nn + DELAY_BUFFER_MAX_STEPS*2) % DELAY_BUFFER_MAX_STEPS];
    }
};

// ── Correlated jitter generator ───────────────────────────────────────────────
// Generates temporally correlated jitter offsets using Box-Muller
// Gaussian samples filtered through a first-order IIR (alpha=0.85).
// Standard deviation = jitter_steps/3 so the specified jitter value
// represents approximately the 3-sigma bound of the offset distribution.
class CorrelatedJitter {
    double prev_offset_, alpha_, sigma_;
public:
    CorrelatedJitter() : prev_offset_(0.0), alpha_(0.85), sigma_(0.0) {}

    void reset(int jitter_steps) {
        prev_offset_ = 0.0;
        sigma_ = (jitter_steps > 0) ? (double)jitter_steps / 3.0 : 0.0;
    }

    int sample(int max_steps) {
        if (max_steps <= 0) return 0;
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        double filtered = alpha_ * prev_offset_ + (1.0 - alpha_) * sigma_ * z;
        prev_offset_ = filtered;
        int offset = (int)round(filtered);
        if (offset >  max_steps) offset =  max_steps;
        if (offset < -max_steps) offset = -max_steps;
        return offset;
    }
};

// ── Smith Predictor ───────────────────────────────────────────────────────────
// Observer-predictor for delay compensation.
// Without delay: runs as a standard current-time observer.
// With delay active:
//   Stage 1 - update observer at t-d using delayed measurement
//   Stage 2 - roll state forward from t-d to t using stored
//             voltages and intermediate delayed measurements
class SmithPredictor {
    int delay_steps_;
    double x_obs_[4];
    bool initialized_;

    void ol_step(const double *xin, double u, double *xout) const {
        for (int i = 0; i < 4; i++) {
            xout[i] = 0.0;
            for (int j = 0; j < 4; j++) xout[i] += Ad[i][j] * xin[j];
            xout[i] += Bd[i] * u;
        }
    }

    void correct(double *xs, double th, double al) const {
        double e0 = th - xs[0], e2 = al - xs[2];
        for (int i = 0; i < 4; i++)
            xs[i] += SP_L[i][0]*e0 + SP_L[i][1]*e2;
    }

public:
    double x[4];  // predicted current state (used by LQR)

    SmithPredictor() : delay_steps_(0), initialized_(false) {
        for (int i = 0; i < 4; i++) x_obs_[i] = x[i] = 0.0;
    }

    void set_delay_steps(int d) {
        delay_steps_ = (d < DELAY_BUFFER_MAX_STEPS) ? d : DELAY_BUFFER_MAX_STEPS-1;
    }

    void reset(double th, double al, double, double) {
        x_obs_[0]=th; x_obs_[1]=0.0; x_obs_[2]=al; x_obs_[3]=0.0;
        x[0]=th;      x[1]=0.0;      x[2]=al;      x[3]=0.0;
        initialized_ = true;
    }

    void step(double th_m, double al_m, bool delay_active,
              const VoltageHistory &vh, const DelayBuffer &db,
              double &th_out, double &al_out) {
        if (!initialized_) { th_out=th_m; al_out=al_m; return; }

        if (!delay_active || delay_steps_ == 0) {
            double xp[4];
            ol_step(x_obs_, vh.get_past(1), xp);
            correct(xp, th_m, al_m);
            for (int i=0; i<4; i++) x_obs_[i] = x[i] = xp[i];
            th_out=th_m; al_out=al_m;
            return;
        }

        // Stage 1: observer update at t-d
        double xp[4];
        ol_step(x_obs_, vh.get_past(delay_steps_+1), xp);
        SensorSample yd = db.read_at_depth(delay_steps_);
        correct(xp, yd.theta, yd.alpha);
        for (int i=0; i<4; i++) x_obs_[i] = xp[i];

        // Stage 2: roll forward from t-d to t
        double xs[4];
        for (int i=0; i<4; i++) xs[i] = x_obs_[i];
        for (int k=1; k<=delay_steps_; k++) {
            double xn[4];
            ol_step(xs, vh.get_past(delay_steps_-k+1), xn);
            SensorSample yk = db.read_at_depth(delay_steps_-k);
            correct(xn, yk.theta, yk.alpha);
            for (int j=0; j<4; j++) xs[j] = xn[j];
        }

        for (int i=0; i<4; i++) x[i] = xs[i];
        th_out=xs[0]; al_out=xs[2];
    }
};

// ── Kalman filter ─────────────────────────────────────────────────────────────
// Full-state estimator measuring only positions (theta, alpha).
// Under delay, uses a practical replay approximation:
//   1. Predict to t-d using stored voltage
//   2. Update on delayed measurement
//   3. Forward-predict from t-d to t using stored voltages
// This is not a full OOSM filter — the covariance is not restored
// retroactively, which is a known limitation under variable jitter.
class KalmanFilter {
public:
    double x[4];
    double P[4][4];

    KalmanFilter() { reset(0.0, 0.0); }

    void reset(double th, double al) {
        x[0]=th; x[1]=0.0; x[2]=al; x[3]=0.0;
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                P[i][j] = (i==j) ? 1.0 : 0.0;
    }

    // Predict: propagate state and covariance one step forward
    void predict(double u) {
        double xp[4]={0,0,0,0};
        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++) xp[i] += Ad[i][j]*x[j];
            xp[i] += Bd[i]*u;
        }
        double AP[4][4]={};
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                for (int k=0; k<4; k++)
                    AP[i][j] += Ad[i][k]*P[k][j];
        double Pp[4][4]={};
        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++) {
                for (int k=0; k<4; k++) Pp[i][j] += AP[i][k]*Ad[k][j];
                if (i==j) Pp[i][j] += (i==1||i==3) ? KF_Qf_vel : KF_Qf_pos;
            }
        }
        for (int i=0; i<4; i++) x[i] = xp[i];
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                P[i][j] = Pp[i][j];
    }

    // Update: correct state using position measurements.
    // Uses Joseph form for numerical stability.
    void update(double th_m, double al_m, double Rf_active) {
        double innov[2] = {th_m-x[0], al_m-x[2]};

        double CP[2][4];
        for (int j=0; j<4; j++) { CP[0][j]=P[0][j]; CP[1][j]=P[2][j]; }

        double S[2][2];
        S[0][0]=CP[0][0]+Rf_active; S[0][1]=CP[0][2];
        S[1][0]=CP[1][0];           S[1][1]=CP[1][2]+Rf_active;

        double det = S[0][0]*S[1][1] - S[0][1]*S[1][0];
        if (fabs(det) < 1e-20) return;

        double Si[2][2];
        Si[0][0]= S[1][1]/det; Si[0][1]=-S[0][1]/det;
        Si[1][0]=-S[1][0]/det; Si[1][1]= S[0][0]/det;

        double PCt[4][2];
        for (int i=0; i<4; i++) { PCt[i][0]=P[i][0]; PCt[i][1]=P[i][2]; }

        double Kg[4][2]={};
        for (int i=0; i<4; i++)
            for (int j=0; j<2; j++)
                for (int k=0; k<2; k++)
                    Kg[i][j] += PCt[i][k]*Si[k][j];

        for (int i=0; i<4; i++)
            x[i] += Kg[i][0]*innov[0] + Kg[i][1]*innov[1];

        double IKC[4][4]={};
        for (int i=0; i<4; i++) IKC[i][i]=1.0;
        for (int i=0; i<4; i++) { IKC[i][0]-=Kg[i][0]; IKC[i][2]-=Kg[i][1]; }

        double IKC_P[4][4]={};
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                for (int k=0; k<4; k++)
                    IKC_P[i][j] += IKC[i][k]*P[k][j];

        double Pnew[4][4]={};
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                for (int k=0; k<4; k++)
                    Pnew[i][j] += IKC_P[i][k]*IKC[j][k];

        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                Pnew[i][j] += Rf_active*(Kg[i][0]*Kg[j][0]+Kg[i][1]*Kg[j][1]);

        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                P[i][j] = Pnew[i][j];
    }
};

// ─────────────────────────────────────── Low-level control functions ───────────────────────────────────────────────

// PD controller to drive arm to theta=0 before swing-up
static double home_control(double th_err, double th_dot) {
    return clamp(-(HOME_KP*th_err + HOME_KD*th_dot), -HOME_VMAX, HOME_VMAX);
}

// Initial kick to start pendulum swinging
static double kick_control(double, double t) {
    return (fmod(t, 1.0) < 0.5) ? KICK_VOLTAGE : -KICK_VOLTAGE;
}

// Energy-pumping: apply torque to add energy to the pendulum.
// Stops if arm drifts too far from home to protect the cable.
static double pump_control(double al, double al_dot, double theta) {
    if (fabs(theta) > 130.0 * M_PI / 180.0) return 0.0;
    if (fabs(al)    * 180.0/M_PI < 120.0)   return 0.0;
    if (fabs(al_dot)* 180.0/M_PI < 100.0)   return 0.0;
    return clamp(-8.0 * sign_f(al * al_dot), -Vmax, Vmax);
}

// LQR balance: u = -scale * K * x
// Alpha sign is flipped to match encoder vs model convention.
static double lqr_control(double th_w, double th_dot, double al, double al_dot, double scale) {
    double u = -scale*(K[0]*th_w + K[1]*th_dot + K[2]*(-al) + K[3]*(-al_dot));
    return clamp(u, -Vmax, Vmax);
}

// ──────────────────────────────────────────────────── Main ─────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {

    // Parse command-line arguments
    int delay_ms=0, jitter_ms=0, run_num=1;
    const char *strategy="baseline", *hw_host="127.0.0.1";

    for (int i=1; i<argc; i++) {
        if      (strcmp(argv[i],"--delay")   ==0 && i+1<argc) delay_ms  = atoi(argv[++i]);
        else if (strcmp(argv[i],"--jitter")  ==0 && i+1<argc) jitter_ms = atoi(argv[++i]);
        else if (strcmp(argv[i],"--strategy")==0 && i+1<argc) strategy  = argv[++i];
        else if (strcmp(argv[i],"--run")     ==0 && i+1<argc) run_num   = atoi(argv[++i]);
        else if (strcmp(argv[i],"--host")    ==0 && i+1<argc) hw_host   = argv[++i];
    }

    bool use_kf = strcmp(strategy,"kf")==0;
    bool use_bl = strcmp(strategy,"baseline")==0;
    bool use_sp = strcmp(strategy,"sp")==0;

    if (!use_kf && !use_bl && !use_sp) {
        fprintf(stderr,"ERROR: unknown strategy '%s'. Use: baseline | kf | sp\n",strategy);
        return 1;
    }

    // Convert ms to buffer steps
    int delay_steps  = (int)((double)delay_ms  / (Ts*1000.0) + 0.5);
    int jitter_steps = (int)((double)jitter_ms / (Ts*1000.0) + 0.5);
    if (delay_steps >= DELAY_BUFFER_MAX_STEPS) {
        fprintf(stderr,"ERROR: delay %d ms too large (max %d ms)\n",
                delay_ms,(int)(DELAY_BUFFER_MAX_STEPS*Ts*1000));
        return 1;
    }

    // Register signal handler for clean shutdown
    struct sigaction sa;
    memset(&sa,0,sizeof(sa));
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM,&sa, NULL);

    // Log filename encodes all condition parameters
    char log_filename[128];
    snprintf(log_filename,sizeof(log_filename),
             "%s_delay%03d_jitter%03d_run%d.csv",
             strategy,delay_ms,jitter_ms,run_num);

    // Set up UDP socket to receive sensor packets
    int sock_sensor = socket(AF_INET,SOCK_DGRAM,0);
    if (sock_sensor<0) { perror("socket sensor"); return 1; }

    int reuse=1;
    setsockopt(sock_sensor,SOL_SOCKET,SO_REUSEADDR,&reuse,sizeof(reuse));

    struct sockaddr_in addr_sensor={};
    addr_sensor.sin_family      = AF_INET;
    addr_sensor.sin_port        = htons(SENSOR_PORT);
    addr_sensor.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(sock_sensor,(struct sockaddr*)&addr_sensor,sizeof(addr_sensor))<0) {
        perror("bind sensor"); close(sock_sensor); return 1;
    }

    // 5ms receive timeout so the loop stays responsive to stop_flag
    struct timeval tv={0,5000};
    if (setsockopt(sock_sensor,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof(tv))<0) {
        perror("setsockopt"); close(sock_sensor); return 1;
    }

    // Set up UDP socket to send voltage commands
    int sock_cmd = socket(AF_INET,SOCK_DGRAM,0);
    if (sock_cmd<0) { perror("socket cmd"); close(sock_sensor); return 1; }

    struct sockaddr_in addr_hw={};
    addr_hw.sin_family      = AF_INET;
    addr_hw.sin_port        = htons(CMD_PORT);
    addr_hw.sin_addr.s_addr = inet_addr(hw_host);

    // Open log file with large write buffer to reduce I/O overhead at 200 Hz
    FILE *log = fopen(log_filename,"w");
    if (!log) { perror("fopen log"); close(sock_sensor); close(sock_cmd); return 1; }
    static char logbuf[1<<20];
    setvbuf(log,logbuf,_IOFBF,sizeof(logbuf));

    fprintf(log,"t_s,dt_s,sample,mode,event,strategy,delay_ms,jitter_ms,delay_steps,"
                "jitter_offset,effective_depth,theta_w_rad,theta_dot_rad_s,alpha_rad,alpha_dot_rad_s,"
                "voltage_cmd_V,theta_integral,delay_blend,Rf_active,"
                "kf_theta,kf_theta_dot,kf_alpha,kf_alpha_dot\n");
    fflush(log);

    printf("========================================\n");
    printf("DISTRIBUTED CONTROLLER PROCESS\n");
    printf("========================================\n");
    printf("Strategy : %s\n",strategy);
    printf("Delay    : %d ms (%d steps)\n",delay_ms,delay_steps);
    printf("Jitter   : %d ms (%d steps, Gaussian correlated, bounded)\n",jitter_ms,jitter_steps);
    printf("Run      : %d\n",run_num);
    printf("Log      : %s\n",log_filename);
    printf("Recv UDP : *:%d\n",SENSOR_PORT);
    printf("Send UDP : %s:%d\n\n",hw_host,CMD_PORT);

    // Instantiate control objects
    DelayBuffer       delay_buf;
    VoltageHistory    volt_hist;
    KalmanFilter      kf;
    SmithPredictor    sp;
    VelocityEstimator theta_vel_est(0.85);
    VelocityEstimator alpha_vel_est(0.85);
    CorrelatedJitter  corr_jitter;

    // Runtime state
    bool initialized=false, kf_active=false, scale_switched=false, delay_active=false;
    uint32_t last_seq=0, cmd_seq=0;
    ControlMode mode=MODE_HOME;
    int samples_read=0;
    double elapsed_time=0.0, home_stable_time=0.0, home_time=0.0, balance_time=0.0;
    double delay_inject_time=-1.0;
    int jitter_offset=0, effective_depth=0;
    double Rf_active=KF_Rf_base;
    const char *event="";

    struct timespec t_prev, t_now;
    clock_gettime(CLOCK_MONOTONIC,&t_prev);

    double theta=0.0, alpha=0.0, voltage=0.0;
    srand((unsigned int)time(NULL));

    // ──────────────────────────────────────────────── Main control loop ─────────────────────────────────────────────────────
    while (!stop_flag) {
        SensorPacket spkt;
        struct sockaddr_in src;
        socklen_t srclen=sizeof(src);

        ssize_t n = recvfrom(sock_sensor,&spkt,sizeof(spkt),0,
                             (struct sockaddr*)&src,&srclen);
        if (n<0) {
            if (errno==EAGAIN||errno==EWOULDBLOCK) continue;
            perror("recvfrom sensor"); break;
        }
        if (n!=(ssize_t)sizeof(SensorPacket)) continue;

        // Discard out-of-order packets
        if (last_seq!=0 && spkt.seq<=last_seq) continue;
        last_seq=spkt.seq;

        clock_gettime(CLOCK_MONOTONIC,&t_now);
        double dt_s = timespec_to_sec(t_now)-timespec_to_sec(t_prev);
        t_prev=t_now;
        if (dt_s<1e-6) dt_s=Ts;
        if (dt_s>0.05)  dt_s=0.05;

        samples_read++;
        elapsed_time+=dt_s;
        theta=spkt.theta_raw;
        alpha=spkt.alpha_raw;

        // One-time initialization on first received packet
        if (!initialized) {
            delay_buf.set_delay_steps(delay_steps);
            delay_buf.prefill(theta,alpha);
            sp.set_delay_steps(delay_steps);
            sp.reset(theta,alpha,0.0,0.0);
            kf.reset(theta,alpha);
            theta_vel_est.reset(theta);
            alpha_vel_est.reset(alpha);
            corr_jitter.reset(jitter_steps);
            kf_active=true;
            initialized=true;
        }

        delay_buf.push(theta,alpha);
        volt_hist.push(voltage);

        // ────────────────────────────────────────────────── Delay injection timing ────────────────────────────────────────────
        if (mode==MODE_BALANCE) {
            balance_time+=dt_s;

            if (use_kf) {
                // KF warms up on undelayed measurements for 5s first
                if (!kf_active && balance_time>=KF_ACTIVATE_S) {
                    kf_active=true;
                    kf.x[0]=theta; kf.x[1]=theta_vel_est.last();
                    kf.x[2]=alpha; kf.x[3]=alpha_vel_est.last();
                    printf("\n>>> KF ACTIVE: %.1fs tdot=%.2f adot=%.2f <<<\n\n",
                           balance_time,kf.x[1],kf.x[3]);
                }
                if (!delay_active && delay_steps>0 && balance_time>=KF_DELAY_INJECT_S) {
                    delay_active=true; delay_inject_time=balance_time;
                    delay_buf.prefill(theta,alpha);
                    corr_jitter.reset(jitter_steps);
                    printf("\n>>> DELAY INJECTION: %d ms +/- %d ms at %.1fs <<<\n\n",
                           delay_ms,jitter_ms,balance_time);
                }
            }
            if (use_bl && !delay_active && delay_steps>0 && balance_time>=BASELINE_DELAY_INJECT_S) {
                delay_active=true; delay_inject_time=balance_time;
                delay_buf.prefill(theta,alpha);
                corr_jitter.reset(jitter_steps);
                printf("\n>>> DELAY INJECTION: %d ms +/- %d ms at %.1fs <<<\n\n",
                       delay_ms,jitter_ms,balance_time);
            }
            if (use_sp && !delay_active && delay_steps>0 && balance_time>=SP_DELAY_INJECT_S) {
                delay_active=true; delay_inject_time=balance_time;
                delay_buf.prefill(theta,alpha);
                corr_jitter.reset(jitter_steps);
                printf("\n>>> SP DELAY INJECTION: %d ms +/- %d ms at %.1fs <<<\n\n",
                       delay_ms,jitter_ms,balance_time);
            }
        }

        // ──────────────────────────────────────────────── Compute delayed measurement ───────────────────────────────────────
        // Reads from buffer at effective depth = nominal delay + jitter offset.
        // Blended in gradually over 0.25s to avoid abrupt feedback step change.
        double theta_meas=theta, alpha_meas=alpha, delay_blend=0.0;
        jitter_offset=0; effective_depth=0; Rf_active=KF_Rf_base;

        if (mode==MODE_BALANCE && delay_active) {
            jitter_offset=corr_jitter.sample(jitter_steps);
            effective_depth=delay_steps+jitter_offset;
            if (effective_depth<0) effective_depth=0;
            if (effective_depth>=DELAY_BUFFER_MAX_STEPS) effective_depth=DELAY_BUFFER_MAX_STEPS-1;

            SensorSample s=delay_buf.read_at_depth(effective_depth);

            double tau=(balance_time-delay_inject_time)/DELAY_BLEND_DURATION_S;
            delay_blend=clamp(tau,0.0,1.0);

            theta_meas=(1.0-delay_blend)*theta + delay_blend*s.theta;
            alpha_meas=(1.0-delay_blend)*alpha + delay_blend*s.alpha;
        }

        // ────────────────────────────────────── Kalman filter predict and update ──────────────────────────────────
        KalmanFilter kf_ctrl;
        if (use_kf && mode==MODE_BALANCE) {
            int replay_depth=delay_active ? effective_depth : 0;
            Rf_active=delay_active ? compute_Rf_from_offset_steps(jitter_offset) : KF_Rf_base;

            kf.predict(volt_hist.get_past(replay_depth+1));
            kf.update(theta_meas,alpha_meas,Rf_active);
            kf_ctrl=kf;

            // Forward-predict from t-d to t to get current-time estimate
            if (delay_active)
                for (int i=replay_depth; i>=1; i--)
                    kf_ctrl.predict(volt_hist.get_past(i));
        }

        // ─────────────────────────────────────── Select state estimates for LQR ────────────────────────────────────
        double theta_c, alpha_c, theta_dot, alpha_dot;

        if (use_kf && mode==MODE_BALANCE && kf_active) {
            theta_c=kf_ctrl.x[0]; alpha_c=kf_ctrl.x[2];
            theta_dot=theta_vel_est.update(theta_meas,dt_s);
            alpha_dot=alpha_vel_est.update(alpha_meas,dt_s);
        } else if (use_sp && mode==MODE_BALANCE) {
            double sp_th, sp_al;
            sp.step(theta_meas,alpha_meas,delay_active,volt_hist,delay_buf,sp_th,sp_al);
            if (delay_active) {
                theta_c=sp_th; alpha_c=sp_al;
                theta_dot=sp.x[1]; alpha_dot=sp.x[3];
            } else {
                theta_c=theta_meas; alpha_c=alpha_meas;
                theta_dot=theta_vel_est.update(theta_meas,dt_s);
                alpha_dot=alpha_vel_est.update(alpha_meas,dt_s);
            }
        } else {
            theta_c=theta_meas; alpha_c=alpha_meas;
            theta_dot=theta_vel_est.update(theta_meas,dt_s);
            alpha_dot=alpha_vel_est.update(alpha_meas,dt_s);
        }

        double theta_w   = wrap_pi(theta_c);
        double theta_err = wrap_pi(theta_w-THETA_HOME_TARGET_RAD);
        event="";

        // ──────────────────────────────────────────────────────────── Mode switching ────────────────────────────────────────────────────
        if (mode==MODE_HOME) {
            home_time+=dt_s;
            if (fabs(theta_err)<HOME_THETA_TOL && fabs(theta_dot)<HOME_DOT_TOL)
                home_stable_time+=dt_s;
            else
                home_stable_time=0.0;

            if (home_stable_time>=HOME_HOLD_TIME || home_time>=HOME_TIMEOUT) {
                event=(home_time>=HOME_TIMEOUT)?"HOME_TIMEOUT_TO_KICK":"HOME_OK_TO_KICK";
                printf("\n>>> HOME %s <<<\n\n",home_time>=HOME_TIMEOUT?"TIMEOUT":"OK");
                mode=MODE_KICK; elapsed_time=0.0;
                theta_vel_est.reset(theta); alpha_vel_est.reset(alpha);
                theta_integral=0.0;
            }
        } else if (mode==MODE_KICK) {
            if (elapsed_time>=KICK_DURATION) {
                event="KICK_TO_PUMP"; mode=MODE_PUMP;
                printf("\n>>> PUMPING <<<\n\n");
            }
        } else if (mode==MODE_PUMP && fabs(alpha)<ALPHA_SWITCH_TO_BALANCE) {
            event="PUMP_TO_BALANCE"; mode=MODE_BALANCE;
            theta_vel_est.reset(theta); alpha_vel_est.reset(alpha);
            theta_integral=0.0; balance_time=0.0;
            kf_active=false; delay_active=false; scale_switched=false;
            delay_inject_time=-1.0; jitter_offset=0; effective_depth=0;
            delay_buf.prefill(theta,alpha);
            kf.reset(theta,alpha);
            sp.reset(theta,alpha,theta_vel_est.last(),alpha_vel_est.last());
            corr_jitter.reset(jitter_steps);
            printf("\n>>> BALANCE (a=%.1f deg) [%s delay=%dms jitter=%dms] <<<\n\n",
                   alpha*180.0/M_PI,strategy,delay_ms,jitter_ms);
        } else if (mode==MODE_BALANCE && fabs(alpha)>ALPHA_SWITCH_TO_PUMP) {
            // Pendulum fell — return to swing-up
            event="BALANCE_TO_PUMP"; mode=MODE_PUMP;
            theta_integral=0.0; balance_time=0.0;
            kf_active=false; delay_active=false; scale_switched=false;
            delay_inject_time=-1.0;
            sp.reset(theta,alpha,0.0,0.0);
            printf("\n>>> BACK TO PUMPING <<<\n\n");
        }

        // ──────────────────────────────────────────────── Compute control voltage ───────────────────────────────────────────
        if (mode==MODE_HOME) {
            voltage=home_control(theta_err,theta_dot);
        } else if (mode==MODE_KICK) {
            voltage=kick_control(theta,elapsed_time);
        } else if (mode==MODE_PUMP) {
            voltage=pump_control(alpha,alpha_dot,theta);
        } else {
            theta_integral+=theta_w*dt_s;
            theta_integral=clamp(theta_integral,-3.0,3.0);

            if (!scale_switched && balance_time>=LQR_SCALE_SWITCH_S) {
                scale_switched=true;
                printf("\n>>> LQR SCALE SWITCH: %.2f -> %.2f (%.1fs) <<<\n\n",
                       LQR_scale_swingup,LQR_scale_balance,balance_time);
            }

            double active_scale=scale_switched ? LQR_scale_balance : LQR_scale_swingup;
            voltage=lqr_control(theta_w,theta_dot,alpha_c,alpha_dot,active_scale);

            // Baseline and SP reduce voltage and integral during the blend window.
            // KF skips this because it already forward-predicts to current time.
            double delay_gain=(!use_kf && delay_active) ? (1.0-0.35*delay_blend) : 1.0;
            voltage*=delay_gain;
            double int_scale=(!use_kf && delay_active) ? (1.0-0.5*delay_blend) : 1.0;
            voltage-=Ki_theta*int_scale*theta_integral;
            voltage=clamp(voltage,-Vmax,Vmax);
        }

        // ─────────────────────────────────────────── Send command to hardware process ──────────────────────────────────
        CmdPacket cpkt;
        memset(&cpkt,0,sizeof(cpkt));
        cpkt.seq=cmd_seq++; cpkt.voltage=voltage;
        ssize_t sent=sendto(sock_cmd,&cpkt,sizeof(cpkt),0,
                            (struct sockaddr*)&addr_hw,sizeof(addr_hw));
        if (sent!=(ssize_t)sizeof(cpkt)) perror("sendto cmd");

        // ───────────────────────────────────────────────── Log one row per sample ─────────────────────────────────────────────
        fprintf(log,"%.6f,%.6f,%d,%s,%s,%s,%d,%d,%d,%d,%d,"
                    "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.9f,"
                    "%.6f,%.6f,%.6f,%.6f\n",
                elapsed_time,dt_s,samples_read,
                mode_str(mode),event,strategy,
                delay_ms,jitter_ms,delay_steps,jitter_offset,effective_depth,
                theta_w,theta_dot,alpha_c,alpha_dot,
                voltage,theta_integral,delay_blend,Rf_active,
                use_kf?kf.x[0]:(use_sp?sp.x[0]:0.0),
                use_kf?kf.x[1]:(use_sp?sp.x[1]:0.0),
                use_kf?kf.x[2]:(use_sp?sp.x[2]:0.0),
                use_kf?kf.x[3]:(use_sp?sp.x[3]:0.0));

        // Print status every 100 samples (~0.5s)
        if (samples_read%100==0) {
            double a_deg=alpha_c*180.0/M_PI, t_deg=theta_w*180.0/M_PI;
            if (mode==MODE_BALANCE) {
                const char *obs=use_kf?(kf_active?"KF":"kf_warm")
                                :use_sp?(delay_active?"SP":"sp_warm"):"raw";
                printf("BALANCE| th:%7.1f a:%7.1f u:%6.2f V [%.0fs OBS:%s DLY:%s off:%d dep:%d SCL:%.2f]\n",
                       t_deg,a_deg,voltage,balance_time,obs,
                       delay_active?"ON":"off",jitter_offset,effective_depth,
                       scale_switched?LQR_scale_balance:LQR_scale_swingup);
            } else {
                printf("%s | th:%7.1f a:%7.1f u:%6.2f V\n",
                       mode_str(mode),t_deg,a_deg,voltage);
            }
            fflush(stdout); fflush(log);
        }
    }

    // Send zero voltage before closing to ensure motor stops safely
    CmdPacket stop_cmd;
    memset(&stop_cmd,0,sizeof(stop_cmd));
    stop_cmd.seq=cmd_seq++; stop_cmd.voltage=0.0;
    sendto(sock_cmd,&stop_cmd,sizeof(stop_cmd),0,
           (struct sockaddr*)&addr_hw,sizeof(addr_hw));

    fclose(log);
    close(sock_sensor);
    close(sock_cmd);

    printf("\nController stopped.\n");
    return 0;
}