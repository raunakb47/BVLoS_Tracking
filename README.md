# BVLoS Tracking Framework

A passive, hardware-agnostic framework for Beyond Visual Line of Sight (BVLoS) environmental sensing, human occupancy detection, and spatial tracking. This architecture leverages IEEE 802.11ac/ax Beamforming Feedback Information (BFI) to extract positional information without requiring a pre-measured grid or physical line-of-sight to the network infrastructure.

The entire workflow first goes through the modified Wi-BFI repository to get the extracted angles and reconstructed V-matrices, then proceeds to the BVLoS tracker.

## 🚀 Core Features
* **Ego-Centric Auto-Calibration:** Automatically builds a local coordinate system around the user's device (0,0), establishing a relative topological baseline by dynamically tracking Access Point (AP) beacons.
* **Modified Wi-BFI Integration:** Customized to natively support low-aperture 2x1 and 2x2 MIMO configurations upto 4x4, extracting topology (wlan.ta and wlan.ra) and RSSI data directly from PCAP headers. Automatically reads matrix dimensions from PCAP headers and groups targets into unique `MAC_Config` buckets to prevent dimension-mismatch crashes when devices change MIMO config.
* **Bistatic Ray-Circle Intersection:** Maps client locations by mathematically intersecting the AP's Angle of Departure (AoD) with a dynamically calculated Free Space Path Loss (FSPL) distance radius from the Monitor Card.
* **Algorithm Dispatcher & Registry:** Evaluates incoming tensor metadata (MIMO config, packet density, kinematic energy) and dynamically routes data to the most mathematically viable algorithm (CA-ESPRIT, SpotFi, IAA-APES, or Residual 2D-MUSIC).
* **Kinematic Phase Variance Tracker (KPVT):** Extracts motion profiles via VSS-LMS background subtraction and PCA to determine client occupancy states.
* **Temporal Sanitization (TDT):** Enforces a Temporal Dropout Threshold to uniformly interpolate complex phase mapping across a 100Hz grid, preventing ghost trajectories during packet starvation.


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
    ├── config.env                   # Master configuration (Channel, Protocol, Thresholds)
    ├── 1_Stage1_Capture.sh          # Live physical capture daemon (Monitor Mode)
    ├── 2_Stage2_Extraction.sh       # Watchdog event daemon bridging capture to extraction
    ├── 2_1_Temporal_Sanitizer.py    # CSMA/CA jitter correction & TDT enforcer
    ├── 3_Stage3_Localization.py     # KPVT Kinematics & STOC Dispatcher Engine
    ├── 3_1_Spatial_Algorithms.py    # Isolated mathematical registry (ESPRIT, SpotFi, etc.)
    └── 4_Stage4_Inference.py        # Ray-Circle geometry, RPGM filters, and JSON UI Output
```

## ⚙️ Prerequisites
Ensure the following system packages and Python libraries are installed before execution:
* **System Utilities:** `tcpdump`, `wireshark-cli` (for `tshark`), `inotify-tools`
* **Python Environment:** `numpy`, `pyshark`, `scipy`, `scikit-learn`
* **Hardware:** A network interface card capable of Monitor Mode (e.g., Alfa AWUS036ACS). Recommended dual-antenna Network Interface Card capable of Monitor Mode (e.g., Alfa AWUS036AXM, AWUS036AXML)

## ▶️ Quick Start Guide

### 1. Configuration
Define capture interface, operating channel, and algorithm thresholds in the environment configuration file.
```bash
nano BVLoS_Live_Tracker/config.env
```

### 2. Launch the Pipeline (Choose Live or Simulation)
**For Live Physical Capture:**
Open a dedicated terminal and start the capture daemon to begin writing temporal chunks to storage.
```bash
cd BVLoS_Live_Tracker
sudo ./1_capture.sh
```

**For Offline Simulation:**
Drip-feed an existing monolithic PCAP file into the pipeline at a true real-time rate.
```bash
cd BVLoS_Live_Tracker
./0_replay_pcap.sh path/to/historical_capture.pcap
```

### 3. Initiate the Tracking Daemon
Open a second terminal window and launch the extraction watchdog. This daemon will automatically trigger the `Wi-BFI` payload extractor, sanitize the tensors, route them through the Stage 3 spatial algorithms, and streams Live JSON telemetry to the dashboard the moment a new PCAP chunk is finalized.
```bash
cd BVLoS_Live_Tracker
./2_Stage2_Extraction.sh
```
