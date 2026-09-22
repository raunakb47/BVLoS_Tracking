# BVLoS Tracking Framework

Passive, unsupervised localization and occupancy sensing from IEEE 802.11ac/ax. Beamforming Feedback Information (BFI). Compressed Beamforming Reports are sent in plaintext as 802.11 Action frames and carry a quantized view of the beamformee-to-beamformer channel, so a monitor-mode interface locked to the target channel can read them without association, without a pre-measured reference grid, and without line of sight to the infrastructure.

V-matrix reconstruction is delegated to a modified fork of Wi-BFI, expected as a sibling directory (`WIBFI_DIR` in `config.env`).

## 🚀 Core Features
* **Ego-centric geometry.** Monitor card at the origin. A bearing is relative to the beamformer's array; absent a surveyed `site.json` a nominal half-wavelength uniform linear array is assumed and every result names which was used.
* **Per-packet decode.** Standard, Nc, Nr, channel width, codebook and subcarrier grouping are read from each frame's MIMO Control field, so one capture covers 2x1 through 4x4 and both Wi-Fi 5 and 6 without configuration. Reports are grouped into buckets keyed `{transmitter}_{beamformer}_{Nr}x{Nc}@{bw}`, which keeps a stack rectangular when a device changes configuration mid-capture.
* **Append-only log as the only state.** Stage 2 writes observables once; a stage needing history reads further back rather than carrying its own state across chunk invocations.
* **Capability-gated dispatch.** Each estimator declares what it requires — uniform linear geometry, a minimum report count, a frequency axis — and Stage 3 compares that against facts decoded in Stage 2. An estimator is applicable or not, and both outcomes are recorded with the reason. Adding one is a single registry entry.
* **Estimators reported side by side.** MUSIC, ESPRIT, SPICE and a joint AoD/relative-delay method are run, never reconciled: agreement means the data supports a bearing, spread means it does not, and a single merged number would hide which.
* **Two benches.** `bench_aoa.py` checks recovery of known angles through the standard's own compression; `bench_precision.py` measures how far each estimator moves when asked twice from different real data.

**Not built yet.** No ranging leg: bearings are angular only, with no distance from RSSI. No Stage 4, so nothing renders a map. Stage 1 has never run against hardware.


## 📁 Workflow Structure
```text
/workspace/
│
├── Wi-BFI/                          # Submodule: Modified Extraction Engine
│   ├── capture_reader.py            # pcap/pcapng reader, radiotap and frame selection
│   ├── main.py                      # per-packet MIMO Control decode, V reconstruction
│   ├── vmatrices.py                 # 4x4 to 2x1 Givens Rotation reconstructor
│   ├── bfi_angles.py                # BFI phase/magnitude dequantizer
│   └── utils.py
│
└── BVLoS_Live_Tracker/              # Tracking Architecture
    ├── 1_Stage1_Capture.sh              monitor-mode capture to rotating pcap chunks
    ├── 2_Stage2_Extraction.sh           per-chunk watcher, runs Stage 2 then Stage 3
    ├── observe.py                       Stage 2: frames to an appended observable log
    ├── dispatch.py                      Stage 3: estimator selection, solve, report
    ├── aoa.py                           AoD estimators and array diagnostics
    ├── bench_aoa.py                     synthetic ground-truth bench for aoa.py
    ├── bench_precision.py               estimator precision on captured data
    ├── config.env                       capture interface, chunk period, filter,
    │                                    per-estimator report-count gates
    ├── 0_replay_pcap.sh                 offline replay of an existing capture
    │
    └── 4_Stage4_Inference.py            superseded, kept for reference while
                                         Stage 4 is rewritten; does not import

```

## ⚙️ Prerequisites
Ensure the following system packages and Python libraries are installed before execution:
* **System Utilities:** `tcpdump`, `iw`, `inotify-tools`, and `wireshark-cli` for `editcap` (offline replay only)
* **Python Environment:** `numpy`
* **Hardware:** A network interface card capable of Monitor Mode (e.g., Alfa AWUS036ACS). Recommended dual-antenna Network Interface Card capable of Monitor Mode (e.g., Alfa AWUS036AXM, AWUS036AXML)

```bash
sudo apt install tcpdump wireshark-cli inotify-tools iw
pip install numpy
```

## ▶️ Quick Start Guide

### 1. Monitor Card setup
Put the interface in monitor mode and lock it to the target channel. The capture
script checks this and prints what it finds; if the width is wrong, BFI payloads arrive
truncated.

### 2. Edit `config.env`
```bash
nano BVLoS_Live_Tracker/config.env
```
Set `CAPTURE_INTERFACE` to the monitor-mode interface and `CHUNK_TIME` to the
rotation period in seconds. Standard, MIMO configuration and channel width are
decoded per packet, so they are not configured here. `MIN_REPORTS_<ESTIMATOR>`
sets how many reports a bucket needs before that estimator runs; the file states
the basis of the shipped values.


### 3. Launch the Pipeline (Choose Live or Simulation)
**For Live Physical Capture:**
Open a dedicated terminal and start the capture daemon to begin writing temporal chunks to storage.
```bash
cd BVLoS_Live_Tracker
sudo ./1_Stage1_Capture.sh 
```

**For Offline Simulation:**
Replay an existing capture at real-time rate
```bash
cd BVLoS_Live_Tracker
./0_replay_pcap.sh  path/to/capture.pcap
```

### 4. Initiate the Tracking Daemon
Open a second terminal window and launch the watcher. On each finalized chunk it
runs Stage 2, invoking the `Wi-BFI` extractor as a subprocess and appending the
observables, then Stage 3 over the log. Both stages time themselves. Outputs land
under a per-run session directory: `stage1/` the chunks, `stage2/observe.jsonl`
and its binary sidecar, `stage3/solve.jsonl` and `solve.txt`, plus `timing.jsonl`
and `pipeline.log`.
```bash
cd BVLoS_Live_Tracker
./2_Stage2_Extraction.sh
```
