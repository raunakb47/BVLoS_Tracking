# BVLoS Tracking Framework

Passive, unsupervised localization of 802.11ac/ax devices from Beamforming Feedback Information (BFI). Compressed Beamforming Reports are sent in plaintext as 802.11 Action frames and carry a quantized view of the beamformee-to-beamformer channel, so a monitor-mode interface locked to the target channel can read them without association, without a pre-measured reference grid, and without line of sight to the infrastructure.

The core output is the bearing of each client at its AP's antenna array, extracted from the BFI reports. RSSI at the monitor adds a range band; it supports the bearing and never replaces it.

V-matrix reconstruction is delegated to a modified fork of Wi-BFI, expected as a sibling directory (`WIBFI_DIR` in `config.env`).

## 🚀 Core Features
* **Continuous pipeline.** Each capture chunk runs extraction (Stage 2), bearings (Stage 3), geometry and confidence (Stage 4); the live view redraws as records are appended.
* **Per-packet decode.** Standard, Nc, Nr, width, codebook and grouping come from each frame's MIMO Control field, so one capture covers 2x1 to 4x4, Wi-Fi 5 and 6, without configuration. Reports are bucketed by `{transmitter}_{beamformer}_{Nr}x{Nc}@{bw}`.
* **Append-only logs as the only state.** Each stage writes once and resumes after the last line it processed.
* **Capability-gated estimators.** MUSIC, SPICE, SAMV-2 and IAA on the BFI covariance (Itahara et al., IEEE Access 2022). Each declares its requirements; Stage 3 records whether it ran and why. Each solves its new reports once they meet its gate, and the others run on the same slice for comparison.
* **Ranging from published models.** RSSI to a distance band via ITU-R P.1238-13 / P.1411-13 (NLoS office by default, ±kσ). Out-of-range readings are flagged, not extrapolated. Transmit power is a stated maximum from the client's Power Capability, the Country element or the host's regulatory domain.
* **Geometry from measured angles only.** Without an anchor, bearings stay in the AP's array frame. With the monitor host's associated interface as anchor (`ANCHOR_MAC`), positions are the bearing ray crossing the RSSI ring around the monitor. No world orientation is assumed; both mirror scenes are kept.
* **Confidence reconciliation.** Each track segment is graded by how many independent estimator groups agree on one direction (1 red, 2 amber, 3 green; dotted on a tie) and revised at most once by a longer covering slice. All rules are set in `config.env`.
* **Live view.** Local web page: trajectory, bearing track, rays and AP array frame, with range bands and bearing history.
* **Benches.** `bench_aoa.py` (recovery of known angles) and `bench_precision.py` (repeatability on real data).

**Limitations.**
* Bearings assume a nominal half-wavelength uniform linear array unless `site.json` gives the element positions; every result names which was used.
* Positions need an anchor. Without one, output is bearing and range only.
* Ranging uses site-general models until a site calibration exists.

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
└── BVLoS_Tracking/                  # Tracking Architecture
    ├── 0_replay_pcap.sh                 offline replay of an existing capture
    ├── 1_Stage1_Capture.sh              monitor-mode capture to rotating pcap chunks
    ├── 2_Stage2_Extraction.sh           per-chunk watcher: Stages 2-4 and the live view
    ├── observe.py                       Stage 2: frames to an appended observable log
    │                                    (BFI reports, RSSI, beacons, Power Capability)
    ├── dispatch.py                      Stage 3: estimator selection, carry-until-gate
    │                                    slices, bearings with client RSSI
    ├── aoa.py                           AoD estimators and array diagnostics
    ├── ranging.py                       ITU-R path-loss models, transmit power chain
    ├── locate.py                        Stage 4: range bands, anchor-relative angles,
    │                                    candidate positions
    ├── confidence.py                    Stage 4: confidence reconciliation per segment
    ├── render.py, render.html           live view (local HTTP server and page)
    ├── bench_aoa.py                     synthetic ground-truth bench for aoa.py
    ├── bench_precision.py               estimator precision on captured data
    ├── config.env                       all settings, each with its basis
    │
    └── 4_Stage4_Inference.py            superseded by locate.py and confidence.py,
                                         kept for reference; does not import
```

## ⚙️ Prerequisites
Ensure the following system packages and Python libraries are installed before execution:
* **System Utilities:** `tcpdump`, `iw`, `inotify-tools`, and `wireshark-cli` for `editcap` (offline replay only)
* **Python Environment:** Python 3 with `numpy`. The live view uses the standard library only.
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
nano BVLoS_Tracking/config.env
```
Each setting is documented in the file with its basis. The main ones:

| Setting | Purpose |
|---|---|
| `CAPTURE_INTERFACE`, `CHUNK_TIME` | monitor-mode interface and rotation period in seconds |
| `LIVE_ESTIMATORS`, `MIN_REPORTS_<NAME>` | estimators on the live path and each one's report-count gate |
| `PATH_LOSS_MODEL`, `RANGE_SIGMA_K` | ITU-R model row (default `office_nlos`) and range band width in σ |
| `ANCHOR_MAC` | MAC of the monitor host's own associated interface; empty for no anchor |
| `CONF_GROUPS`, `CONF_TOLERANCE_FACTOR`, `CONF_SUPPORT_DB`, `CONF_REVISION` | confidence reconciliation |
| `RENDER_PORT` | port of the live view; empty to not start it |

Standard, MIMO configuration and channel width are decoded per packet, so they
are not configured here. `SITE_JSON`, when set, gives surveyed array geometry per
beamformer.

### 3. Launch the Pipeline (Choose Live or Simulation)
**For Live Physical Capture:**
Open a dedicated terminal and start the capture daemon to begin writing temporal chunks to storage.
```bash
cd BVLoS_Tracking
sudo ./1_Stage1_Capture.sh
```

**For Offline Simulation:**
Replay an existing capture at real-time rate
```bash
cd BVLoS_Tracking
./0_replay_pcap.sh  path/to/capture.pcap
```

### 4. Initiate the Tracking Daemon
Open a second terminal window and launch the watcher. On each finalized chunk it
runs Stage 2 (invoking the `Wi-BFI` extractor as a subprocess), Stage 3 over the
log, then Stage 4. Outputs land under a per-run session directory:

| Path | Content |
|---|---|
| `stage1/` | the capture chunks |
| `stage2/observe.jsonl` + binary sidecar | observables per report and frame |
| `stage3/solve.jsonl`, `solve.txt` | bearings per estimator and slice, with facts and client RSSI |
| `stage4/locate.jsonl` | range bands, anchor-relative angles, candidate positions |
| `stage4/segments.jsonl` | reconciled segments: step, bearing, state |
| `timing.jsonl`, `pipeline.log` | per-stage timing and log |

```bash
cd BVLoS_Tracking
./2_Stage2_Extraction.sh
```

### 5. Open the live view
With `RENDER_PORT` set, the watcher starts the view at `http://127.0.0.1:8765/`
(bound to the local host; `RENDER_HOST` overrides). It polls the Stage 4 logs
every second, and a reload replays the session from its start. It can also be run
on its own against any session directory:
```bash
python3 render.py <session_dir> 8765
```
