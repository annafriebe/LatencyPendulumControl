# LatencyPendulumControl

Companion repository for the master's thesis  
**"Comparing the Effects of Latency and Jitter in Distributed Furuta Pendulum Control"**  
Mahmoud Ayoub. Mälardalen University, 2026.

The repository contains the C++ implementation of the distributed control system used in the thesis, the Python scripts used to design controllers and analyse the data, and the full set of CSV logs collected during the experiments.

## Repository layout

```
LatencyPendulumControl/
├── controller_process.cpp     Distributed controller (LQR / KF / SP)
├── hardware_process.cpp       Hardware I/O process (Quanser HIL API)
├── lqr_design_dt.py           Discrete-time LQR design
├── observer_design.py         Smith Predictor observer gain by pole placement
├── Estimate_noise.py          KF noise parameter estimation from logs
├── LICENSE
├── requirements.txt
└── CollectedData/
    ├── analyze_runs.py        Post-processing, metrics, and figures
    ├── Baseline/              Baseline LQR runs, organised by condition
    ├── Kalman Filter/         LQR + KF runs
    ├── Smith Predictor/       LQR + SP runs
    ├── Second_SP/             Extended LQR + SP runs (100 / 300 / 900 ms)
    └── Results/               Generated figures and tables (created by analyze_runs.py)
```

## Hardware and software

The experiments were carried out on a Quanser QUBE-Servo 3 Furuta pendulum connected to a Raspberry Pi 3 Model B+ running Debian GNU/Linux 13 (Trixie). The Quanser HIL SDK must be installed for `hardware_process.cpp` to compile and run; the controller process has no hardware dependency.

The two C++ processes communicate over UDP on the loopback interface, exchanging sensor and command packets at the 200 Hz control rate. Both run as separate Linux processes and can in principle be moved to different hosts.

## Building

The two C++ processes were compiled separately on the Raspberry Pi:

```bash
g++ -I/usr/include/quanser controller_process.cpp -o controller_process \
    -lhil -lquanser_runtime -lquanser_common -lrt -lpthread -ldl -lm

g++ -I/usr/include/quanser hardware_process.cpp -o hardware_process \
    -lhil -lquanser_runtime -lquanser_common -lrt -lpthread -ldl -lm
```

## Running an experiment

Both binaries must be launched on the Raspberry Pi. `hardware_process` requires real-time scheduling because it owns the 200 Hz hardware clock; `controller_process` was also pinned to a separate core for consistency throughout the thesis experiments.

Start the hardware process first:

```bash
sudo chrt -f 99 taskset -c 3 ./hardware_process
```

The first time it runs, press `C` to calibrate the encoders (arm at home, pendulum hanging straight down), then `R` on subsequent runs to use the saved calibration in `calibration.cfg`.

Then, in a second terminal, start the controller for the chosen experimental condition:

```bash
sudo chrt -f 88 taskset -c 2 ./controller_process \
    --strategy kf --delay 10 --jitter 5 --run 1
```

Available arguments:

| Argument     | Values                                  | Description                                  |
|--------------|-----------------------------------------|----------------------------------------------|
| `--strategy` | `baseline`, `kf`, `sp`                  | Control strategy                             |
| `--delay`    | integer milliseconds (0–1000)           | Constant injected sensor-to-controller delay |
| `--jitter`   | integer milliseconds (0 = no jitter)    | Bounded correlated Gaussian jitter amplitude |
| `--run`      | integer                                 | Run number (used in the output filename)     |
| `--host`     | IP address (default `127.0.0.1`)        | Address of `hardware_process`                |

The controller writes one CSV per run, named `<strategy>_delay<ms>_jitter<ms>_run<N>.csv`, into the working directory.

## Analysis

The analysis script `analyze_runs.py` is meant to live inside the `CollectedData/` folder. It auto-locates itself relative to the script's own path, so no paths need to be edited.

Install the Python dependencies once:

```bash
pip install -r requirements.txt
```

Then run the script from anywhere:

```bash
cd CollectedData
python analyze_runs.py
```

The script scans the four strategy folders (`Baseline`, `Kalman Filter`, `Smith Predictor`, `Second_SP`), computes per-run metrics, and writes all figures and tables into `CollectedData/Results/`. The `Results/` folder is created automatically if it does not already exist. CSV files placed inside `Results/` are skipped on re-runs so the script does not re-ingest its own outputs.

The strategy of each CSV file is identified from the top-level folder name, and the filename pattern `<strategy>_delay<NNN>_jitter<NNN>_run<N>.csv` is used to extract the experimental condition. New runs can therefore be added simply by dropping additional CSV files into the appropriate strategy folder.

## Reproducibility notes

All experiments in the thesis used the parameters and gains hard-coded in the source files: a sampling period of 5 ms, the LQR gain `K = [-6.489, -3.387, 93.254, 11.128]` (`Q = diag(50, 1, 100, 5)`, `R = 1`), Kalman filter measurement and process noise estimated from baseline balance logs by `Estimate_noise.py`, and Smith Predictor observer poles `{0.75, 0.70, 0.65, 0.60}` placed by `observer_design.py`. Re-running the design scripts on the same physical model parameters reproduces the gain values used in `controller_process.cpp`.

## Citation

If you use this code or data in academic work, please cite the thesis:

> Mahmoud Ayoub, *Comparing the Effects of Latency and Jitter in Distributed Furuta Pendulum Control*, M.Sc. thesis, Mälardalen University, Västerås, Sweden, 2026.

## License

See [LICENSE](LICENSE).
