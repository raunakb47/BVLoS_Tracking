#!/bin/bash
# ==============================================================================
# Script: 0_replay_pcap.sh
# Purpose: Simulates a real-time environmental sensing stream by taking a single, 
#          monolithic PCAP capture and chronologically drip-feeding it into the 
#          pipeline's watch directory based on the configured CHUNK_TIME.
# ==============================================================================

# Ingest global tracking parameters (e.g., CHUNK_TIME)
source ./config.env

# Assign the first command-line argument ($1) to the INPUT_PCAP variable
INPUT_PCAP=$1

# Sanity check: Ensure the user provided a target file
if [ -z "$INPUT_PCAP" ]; then
    echo "[!] Usage Error: Target capture payload undefined."
    echo "    Syntax: ./0_replay_pcap.sh <historical_capture.pcap>"
    exit 1
fi

# Define local directories for staging and execution
WATCH_DIR="./live_chunks"
TMP_DIR="./tmp_chunks"

# Ensure directories exist (-p prevents errors if they are already present)
mkdir -p $WATCH_DIR $TMP_DIR

echo "[*] Segmenting continuous capture into ${CHUNK_TIME}-second intervals..."

# ------------------------------------------------------------------------------
# editcap: A Wireshark utility for manipulating libpcap files.
# -i $CHUNK_TIME : Instructs editcap to split the file based on chronological 
#                  packet timestamps, creating a new file every X seconds.
# $INPUT_PCAP    : The source monolithic file.
# $TMP_DIR/...   : The output prefix. Editcap appends chronological identifiers 
#                  (e.g., chunk_00000.pcap, chunk_00001.pcap).
# ------------------------------------------------------------------------------
editcap -i $CHUNK_TIME "$INPUT_PCAP" $TMP_DIR/chunk.pcap

echo "[*] Initiating real-time simulation pipeline..."

# Iterate through the newly created temporal chunks in alphabetical/chronological order
for FILE in $TMP_DIR/chunk*.pcap; do
    
    # 1. Move the chunk into the live directory. 
    #    This atomic filesystem action triggers the `inotifywait` daemon in 2_watchdog.sh.
    mv "$FILE" "$WATCH_DIR/"
    echo "[+] Pipeline Ingest: $(basename $FILE)"
    
    # 2. Suspend execution for exactly the chunk duration. 
    #    This ensures the pipeline processes historical data at realistic real-time speeds,
    #    preventing CPU bottlenecking and mimicking live network interface timing.
    sleep $CHUNK_TIME
done

echo "[*] Simulation payload exhausted."