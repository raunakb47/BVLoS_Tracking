#!/bin/bash
# ==============================================================================
# Script: 1_capture.sh
# Purpose: Interfaces directly with the physical NIC (Alfa AWUS036ACS) to ingest 
#          raw 802.11 RF frames. Segments the continuous live stream into discrete, 
#          manageable PCAP files for the extraction pipeline to process.
# ==============================================================================

# Ingest global tracking parameters (CAPTURE_INTERFACE, CHUNK_TIME)
source ./config.env

# Define the target directory where the watchdog daemon is listening
WATCH_DIR="./live_chunks"

# Ensure the pipeline ingestion directory exists
mkdir -p $WATCH_DIR

echo "[*] Initializing environmental sensing on ${CAPTURE_INTERFACE}..."
echo "[*] Temporal segmentation resolution: ${CHUNK_TIME} seconds."

# ------------------------------------------------------------------------------
# tcpdump: The standard command-line packet analyzer.
# -i $CAPTURE_INTERFACE : Binds the capture to the specified wireless adapter.
# -I                    : Enforces Monitor Mode (rfmon). Required to capture raw 
#                         802.11 headers (like BFI Action Frames) without associating 
#                         to an Access Point.
# -s 0                  : Sets snaplen to 0 (captures the entire packet, preventing 
#                         truncation of the critical V-Matrix payloads).
# -G $CHUNK_TIME        : The rotation parameter. Instructs tcpdump to automatically 
#                         close the current file and open a new one every X seconds.
# -W 60                 : Implements a circular buffer. Limits the total number of 
#                         saved chunks to 60. Once reached, it overwrites the oldest.
#                         Prevents infinite SSD consumption during long sensing runs.
# -w .../chunk_%S.pcap  : The output write path. '%S' appends the current seconds 
#                         timestamp to the filename for unique chronometric tracking.
# ------------------------------------------------------------------------------
sudo tcpdump -i $CAPTURE_INTERFACE -I -s 0 -G $CHUNK_TIME -W 60 -w $WATCH_DIR/chunk_%S.pcap