#!/bin/bash
# ==============================================================================
# Module: 1_Stage1_Capture.sh
# Purpose: Interfaces directly with the physical NIC to record raw 802.11 RF frames.
#          Assumes the interface has already been locked to the correct target 
#          frequency and VHT bandwidth.
# ==============================================================================


source ./config.env

mkdir -p $WATCH_DIR

echo "[*] Stage 1 (Capture): Initializing RF ingestion on ${CAPTURE_INTERFACE}"

# ------------------------------------------------------------------------------
# Query the netlink interface to verify the external frequency lock.
# If this does not explicitly say "80MHz" or "160MHz" alongside the channel, 
# BFI extraction will fail due to payload truncation.
# ------------------------------------------------------------------------------
ACTUAL_FREQ=$(iw dev "$CAPTURE_INTERFACE" info | grep -E "channel|width")
echo "[*] Interface State: $ACTUAL_FREQ"

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
#sudo tcpdump -i $CAPTURE_INTERFACE -I -s 0 -G $CHUNK_TIME -W 60 -w $WATCH_DIR/chunk_%S.pcap
sudo tcpdump -i "$CAPTURE_INTERFACE" -I -s 0 -G "$CHUNK_TIME" -W 60 -w "$WATCH_DIR/chunk_%S.pcap"