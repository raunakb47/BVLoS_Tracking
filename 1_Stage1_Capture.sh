#!/bin/bash
# ==============================================================================
# Module: 1_Stage1_Capture.sh
# Record raw 802.11 frames to rotating pcap chunks. The interface must already
# be in monitor mode and locked to the target channel and width.
# ==============================================================================


source ./config.env

mkdir -p $WATCH_DIR

echo "[*] Stage 1 (Capture): Initializing RF ingestion on ${CAPTURE_INTERFACE}"

# Verify the external frequency lock. If this does not report the expected
# width alongside the channel, BFI extraction fails on truncated payloads.
ACTUAL_FREQ=$(iw dev "$CAPTURE_INTERFACE" info | grep -E "channel|width")
echo "[*] Interface State: $ACTUAL_FREQ"

echo "[*] Temporal segmentation resolution: ${CHUNK_TIME} seconds."
echo "[*] Capture filter: ${CAPTURE_FILTER}"

# ------------------------------------------------------------------------------
# tcpdump flags:
# -i  capture interface
# -I  monitor mode (rfmon), required to see 802.11 headers without associating
# -s 0  full snaplen, so V-matrix payloads are not truncated
# -G  rotate to a new file every CHUNK_TIME seconds
# -W 60  file cap per tcpdump run; see the restart loop below
# -w chunk_%s.pcap  '%s' is the Unix epoch second, monotonic and unique, so a
#     restart cannot collide with an earlier filename. '%S' (seconds within the
#     minute) wraps every 60 s and does collide if Stage 2 falls a minute behind.
# CAPTURE_FILTER  BPF filter (config.env) keeping Beacon, Probe Response and
#     Action frames only, so a busy environment's data frames are dropped in the
#     kernel rather than written to disk and discarded in Stage 2.
#
# -G with -W but without -C is not a circular buffer: tcpdump prints "Maximum
# file limit reached" and exits once it has written -W files. At CHUNK_TIME=10
# and -W 60 that ends capture after 10 minutes. The loop restarts it on any
# exit; Stage 2's watchdog does not care which process wrote a chunk.
# ------------------------------------------------------------------------------
echo "[*] Starting capture (auto-restarts on exit, e.g. hitting the -W file cap)."
while true; do
    sudo tcpdump -i "$CAPTURE_INTERFACE" -I -s 0 -G "$CHUNK_TIME" -W 60 \
        -w "$WATCH_DIR/chunk_%s.pcap" "$CAPTURE_FILTER"
    EXIT_CODE=$?
    echo "[!] tcpdump exited (code $EXIT_CODE) -- restarting in 1s." >&2
    sleep 1
done
