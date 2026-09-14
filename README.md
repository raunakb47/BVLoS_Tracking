# BVLoS Tracking Framework

Passive, unsupervised localization and occupancy sensing from IEEE 802.11ac/ax. Beamforming Feedback Information (BFI). Compressed Beamforming Reports are sent in plaintext as 802.11 Action frames and carry a quantized view of the beamformee-to-beamformer channel, so a monitor-mode interface locked to the target channel can read them without association, without a pre-measured reference grid, and without line of sight to the infrastructure.

V-matrix reconstruction is delegated to a modified fork of Wi-BFI, expected as a sibling directory (`WIBFI_DIR` in `config.env`).

## 🚀 Core Features
* **Ego-Centric Geometry:** Monitor card at the origin, AP on the positive Y-axis by convention. 
* **Modified Wi-BFI Integration:** Customized to natively support low-aperture 2x1 and 2x2 MIMO configurations upto 4x4, extracting topology (wlan.ta and wlan.ra) and RSSI data directly from PCAP headers. Automatically reads matrix dimensions from PCAP headers and groups targets into unique `MAC_Config` buckets to prevent dimension-mismatch crashes when devices change MIMO config.
*  **Triangulation.** Client-AP bearing from BFI, client-monitor range from the client's own frame RSSI, AP-monitor range from Beacon RSSI and the AP's advertised transmit power.
* **Bistatic Ray-Circle Intersection:** Maps client locations by mathematically intersecting the AP's Angle of Departure (AoD) with a dynamically calculated Free Space Path Loss (FSPL) distance radius from the Monitor Card.
* **Algorithm Dispatcher & Registry:** Evaluates incoming tensor metadata (MIMO config, packet density, kinematic energy) and dynamically routes data to the most mathematically viable algorithm (CA-ESPRIT, SpotFi, IAA-APES, or Residual 2D-MUSIC).  SSE resolves bearing and favours fixed APs; KPVT resolves motion and carries the mobile-hotspot case. A low-confidence bearing is withheld while KPVT's occupancy read still stands. CA-ESPRIT, SpotFi, IAA-APES or Residual 2D-MUSIC, selected per bucket on array size, packet density and kinematic energy.
* **Kinematic Phase Variance Tracker (KPVT):** Extracts motion profiles via VSS-LMS background subtraction and PCA to determine client occupancy states. Per-bucket VSS-LMS, scale-normalized against the bucket's own residual-power floor, replacing a static per-chunk mean.
* **Cross-chunk continuity.** A sliding estimation window decoupled from the capture rotation cadence, plus radar-style confirmed/coasting/dropped tracks that hold a client in place through quiet intervals instead of dropping it on the first one.

Each client-AP link is tracked in its own bucket, keyed `{client_mac}_{ap_mac}_{mimo}`.


## 📁 Workflow Structure
```text
/workspace/
│
├── Wi-BFI/                          # Submodule: Modified Extraction Engine
│   ├── main.py                      # Dual-MAC PCAP parser (wlan.ta + wlan.ra + RSSI)
│   ├── vmatrices.py                 # 4x4 to 2x1 Givens Rotation reconstructor
│   ├── bfi_angles.py                # BFI phase/magnitude dequantizer
│   └── utils.py
│
└── BVLoS_Live_Tracker/              # Tracking Architecture
    ├── 1_Stage1_Capture.sh              monitor-mode capture to rotating pcap chunks
    ├──    2_Stage2_Extraction.sh        per-chunk dispatcher (inotify)
    ├── 2_1_Temporal_Sanitizer.py        dropout-threshold segmentation and uniform resampling
    ├──    3_Stage3_Localization.py      estimator routing
    ├──   3_1_Spatial_Algorithms.py      SSE estimators and model-order gating
    ├──   3_2_Kinematic_Tracker.py       KPVT: VSS-LMS background, OS-CFAR occupancy
    ├──  4_Stage4_Inference.py           ray-circle geometry, track state, JSON telemetry
    ├──  state_store.py                  cross-chunk per-bucket state
    ├── tx_power_estimator.py            transmit-power estimates and log-distance ranging
    ├── extract_ap_metadata.py           AP SSID, transmit power and beacon RSSI from Beacons
    ├──   ap_registry.py                 persistent AP registry and derived AP range
    ├──   hotspot_classifier.py          mobile-hotspot classification of the beamformer
    ├──   config.env                     interface, channel, thresholds, state paths
    ├──  0_replay_pcap.sh                offline replay of an existing capture

```

## ⚙️ Prerequisites
Ensure the following system packages and Python libraries are installed before execution:
* **System Utilities:** `tcpdump`, `wireshark-cli` (for `tshark`), `inotify-tools`
* **Python Environment:** `numpy`, `pyshark`, `scipy`, `scikit-learn`
* **Hardware:** A network interface card capable of Monitor Mode (e.g., Alfa AWUS036ACS). Recommended dual-antenna Network Interface Card capable of Monitor Mode (e.g., Alfa AWUS036AXM, AWUS036AXML)

```bash
sudo apt install tcpdump tshark inotify-tools iw
pip install numpy scipy scikit-learn pyshark
```

## ▶️ Quick Start Guide

### 1. Monitor Card setup
Put the interface in monitor mode and lock it to the target channel. The capture
script checks this and prints what it finds; if the width is wrong, BFI payloads arrive
truncated.

Define capture interface, operating channel, and algorithm thresholds in the environment configuration file.
```bash
nano BVLoS_Live_Tracker/config.env
```
### 2. Edit `config.env`.** 
At minimum set `CAPTURE_INTERFACE`, `WIFI_CHANNEL`, `WIFI_STANDARD` and `BANDWIDTH` to match. Then set `TDT_MS` above the interval at which the AP actually sounds its clients — roughly 500 for 802.11ax and 2000 for 802.11ac on the reference traces. Too low and every packet is segmented on its own and nothing comes out; the sanitizer says so on stderr if that happens.


### 2. Launch the Pipeline (Choose Live or Simulation)
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

### 3. Initiate the Tracking Daemon
Open a second terminal window and launch the extraction watchdog. This daemon will automatically trigger the `Wi-BFI` payload extractor, sanitize the tensors, route them through the Stage 3 spatial algorithms, and streams Live JSON telemetry to the dashboard the moment a new PCAP chunk is finalized.
```bash
cd BVLoS_Live_Tracker
./2_Stage2_Extraction.sh
```
