// hardware_process.cpp
//
// Process A: hardware-side I/O for QUBE-Servo 3.
//
// This process owns exclusive access to the Quanser hardware. It runs at
// 200 Hz driven by the encoder hardware clock, reads arm and pendulum angles
// from the two encoders, forwards them to the controller process over UDP,
// receives voltage commands back, and applies them to the motor.
//
// It is designed to be as simple and robust as possible. It does no control
// logic — that responsibility belongs entirely to controller_process.
//
// Communication:
//   Sends SensorPacket  to port 9001 (controller_process listens here)
//   Receives CmdPacket  on port 9002  (controller_process sends here)
//
// Safety:
//   If no valid command arrives within CMD_TIMEOUT_S (50 ms), voltage is
//   set to 0 V automatically. This ensures the motor stops if the controller
//   crashes or is not yet started.
//
// LED status indicator:
//   Red    — board just opened or shutdown
//   Blue   — motor enabled, waiting for controller to connect
//   Green  — controller commands arriving, system running normally
//   Yellow — error condition (calibration missing, hardware fault)
//
// Usage:
//   ./hardware_process [--host 127.0.0.1]

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "quanser_extern.h"
#include "quanser_messages.h"
#include "quanser_signal.h"
#include "hil.h"

// ───────────────────────────────────────────── Configuration ──────────────────────────────────────
#define CALIB_FILE    "calibration.cfg"  // encoder offset file written by calibration
#define SENSOR_PORT   9001               // UDP port for outgoing sensor packets
#define CMD_PORT      9002               // UDP port for incoming voltage commands
#define CMD_TIMEOUT_S 0.050              // 50 ms — apply 0 V if no command received

// ──────────────────────────────────────── System constants ────────────────────────────────────────
static const double Ts             = 0.005;                    // sampling period [s]
static const double Vmax           = 10.0;                     // voltage saturation [V]
static const int    ENC_RESOLUTION = 2048;                     // encoder counts/revolution
static const double RAD_PER_COUNT  = 2.0 * M_PI / ENC_RESOLUTION;

static volatile int stop_flag = 0;

// ───────────────────────────────────────── UDP packet structures ────────────────────────────────
// // Packed to keep the binary packet layout consistent between sender and receiver.
// Both processes must use the same struct definitions.
#pragma pack(push, 1)
struct SensorPacket {
    uint32_t seq;           // increments every sample — used to detect dropped packets
    double   timestamp;     // time since hardware process started [s]
    double   theta_raw;     // arm angle after calibration offset [rad]
    double   alpha_raw;     // pendulum angle after calibration offset, wrapped to (-pi, pi] [rad]
    int32_t  theta_counts;  // raw encoder counts (arm)
    int32_t  alpha_counts;  // raw encoder counts (pendulum)
    int32_t  theta_offset;  // calibration offset (arm)
    int32_t  alpha_offset;  // calibration offset (pendulum)
};

struct CmdPacket {
    uint32_t seq;       // sequence number from controller
    double   timestamp; // timestamp from controller
    double   voltage;   // commanded motor voltage [V], clamped to +-Vmax on receipt
};
#pragma pack(pop)

// ───────────────────────────────────────── Signal handler ──────────────────────────────────────
static inline void signal_handler(int) { stop_flag = 1; }

// ─────────────────────────────────────── Utility functions ─────────────────────────────────────
static inline double timespec_to_sec(const struct timespec &t) {
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static inline double clamp(double x, double lo, double hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

// wrap_pi: maps any angle to (-pi, pi] — used to keep alpha in the standard range
static inline double wrap_pi(double x) {
    while (x <= -M_PI) x += 2.0 * M_PI;
    while (x >   M_PI) x -= 2.0 * M_PI;
    return x;
}

// ────────────────────────────────────────── LED helper ───────────────────────────────────────
// Writes RGB values directly to the QUBE-Servo 3 onboard LED.
// Channels 11000/11001/11002 correspond to R/G/B on this hardware.
static inline void set_led(t_card board, double r, double g, double b) {
    const t_uint32 LED_CH[3] = {11000, 11001, 11002};
    t_double rgb[3];
    rgb[0] = clamp(r, 0.0, 1.0);
    rgb[1] = clamp(g, 0.0, 1.0);
    rgb[2] = clamp(b, 0.0, 1.0);
    hil_write_other(board, LED_CH, 3, rgb);
}

// ────────────────────────────────── Calibration ──────────────────────────────────────────────
// Calibration records the raw encoder counts when the arm is at the home mark
// and the pendulum is hanging straight down. These offsets are subtracted at
// runtime so that theta=0 maps to the home mark and alpha=0 maps to upright.
// Offsets are saved to a text file so calibration only needs to be done once.

static bool save_calibration(t_int32 theta_offset, t_int32 alpha_offset) {
    FILE *f = fopen(CALIB_FILE, "w");
    if (!f) { perror("fopen calibration.cfg"); return false; }
    fprintf(f, "theta_offset=%d\n", (int)theta_offset);
    fprintf(f, "alpha_offset=%d\n", (int)alpha_offset);
    fclose(f);
    printf("Calibration saved: theta_off=%d alpha_off=%d\n",
           (int)theta_offset, (int)alpha_offset);
    return true;
}

static bool load_calibration(t_int32 &theta_offset, t_int32 &alpha_offset) {
    FILE *f = fopen(CALIB_FILE, "r");
    if (!f) {
        printf("WARNING: %s not found. Calibrate first (C).\n", CALIB_FILE);
        return false;
    }
    int th = 0, al = 0;
    int ok = fscanf(f, "theta_offset=%d\nalpha_offset=%d\n", &th, &al);
    fclose(f);
    if (ok != 2) {
        printf("WARNING: %s corrupted. Re-calibrate.\n", CALIB_FILE);
        return false;
    }
    theta_offset = (t_int32)th;
    alpha_offset = (t_int32)al;
    printf("Calibration loaded: theta_off=%d alpha_off=%d\n",
           (int)theta_offset, (int)alpha_offset);
    return true;
}

static bool calibrate_offsets(t_int32 &theta_offset, t_int32 &alpha_offset) {
    t_card board_cal = NULL;
    t_int result;
    char msg[512];

    // Open the board temporarily just to read the encoder counts
    result = hil_open("qube_servo3_usb", "0", &board_cal);
    if (result != 0) {
        msg_get_error_message(NULL, result, msg, sizeof(msg));
        printf("ERROR: %s\n", msg);
        return false;
    }

    const t_uint32 enc_ch[] = {0, 1};
    t_int32 counts[2] = {0, 0};

    printf("\n=== CALIBRATION ===\n");
    printf("1. Move arm to HOME mark\n");
    printf("2. Let pendulum hang DOWN\n");
    printf("3. Hold still and press Enter...\n");

    int c;
    while ((c = getchar()) != '\n' && c != EOF) {}
    while ((c = getchar()) != '\n' && c != EOF) {}

    result = hil_read_encoder(board_cal, enc_ch, 2, counts);
    if (result < 0) {
        msg_get_error_message(NULL, result, msg, sizeof(msg));
        printf("ERROR: %s\n", msg);
        hil_close(board_cal);
        return false;
    }

    // theta offset: arm is at home → theta should read 0
    theta_offset = counts[0];
    // alpha offset: pendulum is hanging down → alpha should read ±pi
    // adding half a revolution shifts the zero so that upright = 0
    alpha_offset = counts[1] + ENC_RESOLUTION / 2;

    bool ok = save_calibration(theta_offset, alpha_offset);
    hil_close(board_cal);
    if (ok) printf("Calibration done.\n\n");
    return ok;
}

// ──────────────────────────────── Safe shutdown ──────────────────────────────────────────
// Called on any exit path to ensure the motor is stopped, the motor enable
// signal is deasserted, and all resources are released cleanly.
static void safe_shutdown(t_card board, t_task task,
                          const t_uint32 analog_channel,
                          const t_uint32 digital_channel,
                          int sock_send, int sock_recv) {
    t_double  voltage = 0.0;
    t_boolean enable  = 0;

    if (board) hil_write_analog(board, &analog_channel, 1, &voltage);
    if (task)  { hil_task_stop(task); hil_task_delete(task); }
    if (board) {
        hil_write_digital(board, &digital_channel, 1, &enable);
        set_led(board, 1.0, 0.0, 0.0);  // red: shutdown complete
        hil_close(board);
    }
    if (sock_send >= 0) close(sock_send);
    if (sock_recv >= 0) close(sock_recv);
}

// ──────────────────────────────────────────── Main ─────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    const char *controller_host = "127.0.0.1";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--host") == 0 && i + 1 < argc)
            controller_host = argv[++i];
    }

    // Register Ctrl+C handler so the motor stops cleanly on interrupt
    qsigaction_t action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = signal_handler;
    qsigemptyset(&action.sa_mask);
    qsigaction(SIGINT, &action, NULL);

    t_int32 theta_offset = 0;
    t_int32 alpha_offset = 0;

    // ────────────────────────────────── Startup menu ────────────────────────────────────────────────
    while (true) {
        printf("Press C to Calibrate, R to Run: ");
        fflush(stdout);

        char ch = 0;
        while (ch != 'c' && ch != 'C' && ch != 'r' && ch != 'R') {
            int c = getchar();
            if (c == EOF) { printf("\nEOF\n"); return 0; }
            ch = (char)c;
        }
        printf("\n");

        if (ch == 'c' || ch == 'C') {
            calibrate_offsets(theta_offset, alpha_offset);
            continue;
        }

        if (!load_calibration(theta_offset, alpha_offset)) {
            printf("Cannot run without calibration.\n\n");
            // Yellow LED: signal to the user that calibration is missing.
            // The board must be opened briefly just to drive the LED.
            t_card board_err = NULL;
            if (hil_open("qube_servo3_usb", "0", &board_err) == 0) {
                set_led(board_err, 1.0, 0.8, 0.0);  // yellow: calibration error
                hil_close(board_err);
            }
            continue;
        }

        break;
    }

    // ─────────────────────────────────── Open board ─────────────────────────────────────────────────
    t_card board = NULL;
    t_task task  = NULL;
    t_int  result;
    char   message[512];

    result = hil_open("qube_servo3_usb", "0", &board);
    if (result != 0) {
        msg_get_error_message(NULL, result, message, sizeof(message));
        printf("ERROR: %s\n", message);
        return 1;
    }

    printf("Board opened.\n");
    set_led(board, 1.0, 0.0, 0.0);  // red: board open, motor not yet enabled

    const t_uint32 encoder_channels[] = {0, 1};
    const t_uint32 analog_channel     = 0;
    const t_uint32 digital_channel    = 0;

    t_int32   encoder_counts[2] = {0, 0};
    t_double  voltage           = 0.0;
    t_boolean enable            = 1;

    // Enable the motor drive before starting the encoder task
    result = hil_write_digital(board, &digital_channel, 1, &enable);
    if (result < 0) {
        msg_get_error_message(NULL, result, message, sizeof(message));
        printf("ERROR enabling motor: %s\n", message);
        safe_shutdown(board, NULL, analog_channel, digital_channel, -1, -1);
        return 1;
    }
    printf("Motor enabled.\n");

    // Create a hardware-clocked encoder reader task at 200 Hz.
    // The task blocks on hil_task_read_encoder, which provides accurate timing
    // without busy-waiting or OS scheduler interference.
    const double freq = 1.0 / Ts;
    result = hil_task_create_encoder_reader(board, freq, encoder_channels, 2, &task);
    if (result < 0) {
        msg_get_error_message(NULL, result, message, sizeof(message));
        printf("ERROR create encoder task: %s\n", message);
        safe_shutdown(board, NULL, analog_channel, digital_channel, -1, -1);
        return 1;
    }

    result = hil_task_start(task, HARDWARE_CLOCK_0, freq, -1);
    if (result < 0) {
        msg_get_error_message(NULL, result, message, sizeof(message));
        printf("ERROR start encoder task: %s\n", message);
        safe_shutdown(board, task, analog_channel, digital_channel, -1, -1);
        return 1;
    }

    // ───────────────────────────────── UDP sockets ───────────────────────────────────────
    // sock_send: sends SensorPackets to the controller process
    // sock_recv: receives CmdPackets from the controller process (non-blocking)
    int sock_send = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_send < 0) {
        perror("socket send");
        safe_shutdown(board, task, analog_channel, digital_channel, -1, -1);
        return 1;
    }

    struct sockaddr_in addr_controller;
    memset(&addr_controller, 0, sizeof(addr_controller));
    addr_controller.sin_family      = AF_INET;
    addr_controller.sin_port        = htons(SENSOR_PORT);
    addr_controller.sin_addr.s_addr = inet_addr(controller_host);

    int sock_recv = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_recv < 0) {
        perror("socket recv");
        safe_shutdown(board, task, analog_channel, digital_channel, sock_send, -1);
        return 1;
    }

    struct sockaddr_in addr_cmd;
    memset(&addr_cmd, 0, sizeof(addr_cmd));
    addr_cmd.sin_family      = AF_INET;
    addr_cmd.sin_port        = htons(CMD_PORT);
    addr_cmd.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sock_recv, (struct sockaddr*)&addr_cmd, sizeof(addr_cmd)) < 0) {
        perror("bind recv");
        safe_shutdown(board, task, analog_channel, digital_channel, sock_send, sock_recv);
        return 1;
    }

    // Set non-blocking so the receive loop drains all pending packets without
    // blocking the encoder read cycle
    int flags = fcntl(sock_recv, F_GETFL, 0);
    if (flags >= 0) fcntl(sock_recv, F_SETFL, flags | O_NONBLOCK);

    printf("UDP: sending sensors to %s:%d\n", controller_host, SENSOR_PORT);
    printf("UDP: receiving commands on port %d\n", CMD_PORT);
    set_led(board, 0.0, 0.0, 1.0);  // blue: waiting for controller

    struct timespec t_start, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    uint32_t seq          = 0;
    uint32_t last_cmd_seq = 0;
    double   last_cmd_time = -1.0;  // -1 means no command received yet
    double   last_cmd_v   = 0.0;
    long long samples     = 0;

    // LED state: 0=red, 1=blue, 2=green
    // Only written to hardware when state changes to avoid HIL overhead at 200 Hz
    int led_state = 1;

    printf("\n=== RUNNING -- waiting for controller ===\n");
    printf("(Start controller_process now if not already running)\n\n");

    // ────────────────────────────────────────── Main loop ────────────────────────────────────────────
    // Driven by hil_task_read_encoder which blocks until the next 200 Hz tick.
    // Each iteration: read encoders → send sensor packet → drain command socket
    // → apply voltage → update LED.
    while (!stop_flag) {
        result = hil_task_read_encoder(task, 1, encoder_counts);
        if (result < 0) {
            msg_get_error_message(NULL, result, message, sizeof(message));
            printf("ERROR: %s\n", message);
            set_led(board, 1.0, 0.8, 0.0);  // yellow: encoder hardware fault
            break;
        }

        if (result == 0) continue;  //no new sample returned

        clock_gettime(CLOCK_MONOTONIC, &t_now);
        double ts = timespec_to_sec(t_now) - timespec_to_sec(t_start);
        samples++;

        // ─────────────────────────────── Build and send sensor packet ──────────────────────────────────────
        // Raw counts are included alongside calibrated angles so the controller
        // has full information if it needs to recompute angles differently.
        SensorPacket sp;
        memset(&sp, 0, sizeof(sp));
        sp.seq          = seq++;
        sp.timestamp    = ts;
        sp.theta_counts = encoder_counts[0];
        sp.alpha_counts = encoder_counts[1];
        sp.theta_offset = theta_offset;
        sp.alpha_offset = alpha_offset;
        sp.theta_raw    = (encoder_counts[0] - theta_offset) * RAD_PER_COUNT;
        sp.alpha_raw    = wrap_pi((encoder_counts[1] - alpha_offset) * RAD_PER_COUNT);

        ssize_t sent = sendto(sock_send, &sp, sizeof(sp), 0,
                              (struct sockaddr*)&addr_controller, sizeof(addr_controller));
        if (sent != (ssize_t)sizeof(sp)) perror("sendto sensor");

        // ─────────────────────────────────────── Drain command socket ──────────────────────────────────────────────
        // Multiple packets may have arrived since the last iteration.
        // Only the highest sequence number is kept — older commands are discarded
        // because applying a stale command would be worse than applying nothing.
        CmdPacket cp;
        struct sockaddr_in src;
        socklen_t srclen = sizeof(src);

        while (true) {
            ssize_t n = recvfrom(sock_recv, &cp, sizeof(cp), 0,
                                 (struct sockaddr*)&src, &srclen);
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                perror("recvfrom cmd");
                break;
            }
            if (n != (ssize_t)sizeof(cp)) continue;
            if (last_cmd_seq == 0 || cp.seq > last_cmd_seq) {
                last_cmd_seq  = cp.seq;
                last_cmd_time = ts;
                last_cmd_v    = clamp(cp.voltage, -Vmax, Vmax);
            }
        }

        // ─────────────────────────────────── Apply voltage ─────────────────────────────────────────────────────
        // If no valid command has arrived within CMD_TIMEOUT_S, apply 0 V.
        // This is the safety fallback for controller crash or network failure.
        if (last_cmd_time < 0.0 || (ts - last_cmd_time) > CMD_TIMEOUT_S)
            voltage = 0.0;
        else
            voltage = last_cmd_v;

        result = hil_write_analog(board, &analog_channel, 1, &voltage);
        if (result < 0) {
            msg_get_error_message(NULL, result, message, sizeof(message));
            printf("ERROR writing analog output: %s\n", message);
            set_led(board, 1.0, 0.8, 0.0);  // yellow: analog output fault
            break;
        }

        // ──────────────────────────────── Update LED on state change ────────────────────────────────────────
        // Writing to the LED every sample at 200 Hz would waste HIL bandwidth.
        // Instead, the desired state is computed and the LED is only updated
        // when the state actually changes.
        {
            int new_led;
            if (last_cmd_time < 0.0)
                new_led = 1;  // blue: waiting — controller not yet connected
            else if ((ts - last_cmd_time) > CMD_TIMEOUT_S)
                new_led = 0;  // red: controller was connected but stopped sending
            else
                new_led = 2;  // green: receiving commands normally

            if (new_led != led_state) {
                led_state = new_led;
                if      (led_state == 0) set_led(board, 1.0, 0.0, 0.0);  // red
                else if (led_state == 1) set_led(board, 0.0, 0.0, 1.0);  // blue
                else                     set_led(board, 0.0, 1.0, 0.0);  // green
            }
        }

        // ───────────────────────────────────── Status print every 1 second (200 samples) ─────────────────────────
        if (samples % 200 == 0) {
            printf("HW | t:%.1fs th:%.1f deg al:%.1f deg u:%.2fV seq:%u cmd_age:%.0fms\n",
                   ts,
                   sp.theta_raw * 180.0 / M_PI,
                   sp.alpha_raw * 180.0 / M_PI,
                   (double)voltage,
                   sp.seq,
                   (last_cmd_time < 0.0) ? -1.0 : (ts - last_cmd_time) * 1000.0);
            fflush(stdout);
        }
    }

    printf("\n=== HARDWARE SHUTDOWN ===\n");
    safe_shutdown(board, task, analog_channel, digital_channel, sock_send, sock_recv);
    printf("Done.\n");
    return 0;
}