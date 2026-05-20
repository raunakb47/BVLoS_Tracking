# BVLoS Tracking Framework

A passive, hardware-agnostic framework for Beyond Visual Line of Sight (BVLoS) environmental sensing and spatial bearing resolution. This architecture leverages IEEE 802.11ac Beamforming Feedback Information (BFI) to mathematically resolve the Angle of Arrival (AoA) of multiple unassociated Wi-Fi targets simultaneously using the MUSIC algorithm.

## 🚀 Core Features
* **Environmental Sensing:** Operates completely blindly. Captures and maps every radiating IEEE 802.11 entity in the environment without requiring target MAC addresses.
* **Dynamic MIMO Auto-Detection:** Automatically reads matrix dimensions from PCAP headers and groups targets into unique `MAC_Config` buckets to prevent dimension-mismatch crashes when devices shape-shift to save power.
* **High-Resolution Codebooks:** Extended Givens Rotation math supporting highly complex array configurations up to **4x4** and **4x3**, scaling seamlessly down to **2x1**.
* **Real-Time Continuous Pipeline:** A modular, bash-driven asynchronous pipeline that mimics real-time radar processing using sliding temporal capture windows.

## 📁 Workflow Structure
```text
/workspace/
│
├── Wi-BFI/                      # Submodule: Modified Extraction Engine
│   ├── main.py                  # MAC-agnostic PCAP parser & bucket router
│   ├── vmatrices.py             # 4x4 to 2x1 Givens Rotation reconstructor
│   ├── bfi_angles.py            # BFI phase/magnitude dequantizer
│   └── utils.py
│
└── BVLoS_Live_Tracker/          # Main Application: Tracking Architecture
    ├── config.env               # Master configuration file (Interface, Protocol, Timing)
    ├── 0_replay_pcap.sh         # Offline simulation engine for historical PCAPs
    ├── 1_capture.sh             # Live physical capture daemon (requires Monitor Mode capable WiFi-5+ Adapter)
    ├── 2_watchdog.sh            # inotifywait event daemon bridging capture to extraction
    └── 3_bvlos_tracker.py       # 1D-MUSIC eigenvalue decomposition and UI output
```

## ⚙️ Prerequisites
Ensure the following system packages and Python libraries are installed before execution:
* **System Utilities:** `tcpdump`, `wireshark-cli` (for `editcap`), `inotify-tools`
* **Python Environment:** `numpy`, `pyshark`
* **Hardware:** A network interface card capable of Monitor Mode (e.g., Alfa AWUS036ACS).

## 🚦 Quick Start Guide

### 1. Configuration
Define your capture parameters, channel bandwidth, and physical interface in the master configuration file:
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
Open a second terminal window and launch the watchdog. This daemon will automatically trigger the `Wi-BFI` extraction engine and the `3_bvlos_tracker.py` MUSIC algorithm the moment a new PCAP chunk is finalized.
```bash
cd BVLoS_Live_Tracker
./2_watchdog.sh
```
