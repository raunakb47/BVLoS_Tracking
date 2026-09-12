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
echo "[*] Capture filter: ${CAPTURE_FILTER}"

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
# -W 60                 : Caps how many chunk files one tcpdump run produces before
#                         it exits (see the restart loop below for why "exits" and
#                         not "overwrites the oldest").
# -w .../chunk_%s.pcap  : The output write path. '%s' (lowercase) appends the current
#                         Unix epoch second -- a monotonically increasing, globally
#                         unique value -- so restarts (below) can never produce a
#                         filename collision. The original '%S' (uppercase, seconds-
#                         within-the-minute, 00-59) wrapped every 60 seconds; verified
#                         against this system's tcpdump/strftime, that is a real
#                         filename-collision risk if Stage 2 ever falls more than a
#                         minute behind, not just a style choice.
# CAPTURE_FILTER        : BPF filter (config.env) restricting capture at the kernel
#                         level to Beacon, Probe Response, and Action management
#                         frames -- the only frame types this pipeline uses -- instead
#                         of writing every data/control frame in a busy environment to
#                         disk only to discard almost all of it in Stage 2's pyshark
#                         display_filter.
# ------------------------------------------------------------------------------
#
# -G combined with -W but WITHOUT -C does NOT implement the circular/overwrite
# buffer the single-tcpdump-invocation comment above once assumed: verified
# directly in this session (`tcpdump -i lo -G 2 -W 3 ...` against live loopback
# traffic) that tcpdump prints "Maximum file limit reached: N" and EXITS once it
# has written -W files, rather than wrapping around to overwrite the oldest. At
# this file's default settings (CHUNK_TIME=10, -W 60) that means Stage 1 would
# silently stop capturing after exactly 10 minutes, regardless of how long the
# monitoring session is meant to run. The loop below restarts tcpdump every time
# it exits for any reason (hitting that file cap included), so capture continues
# for the whole session; Stage 2's inotify watchdog does not care which tcpdump
# process wrote a given chunk, so a restart is invisible downstream.
echo "[*] Starting capture (auto-restarts on exit, e.g. hitting the -W file cap)."
while true; do
    sudo tcpdump -i "$CAPTURE_INTERFACE" -I -s 0 -G "$CHUNK_TIME" -W 60 \
        -w "$WATCH_DIR/chunk_%s.pcap" "$CAPTURE_FILTER"
    EXIT_CODE=$?
    echo "[!] tcpdump exited (code $EXIT_CODE) -- restarting in 1s." >&2
    sleep 1
done
