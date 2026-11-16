#!/bin/bash
# ==============================================================================
# Module: 0_replay_pcap.sh
# Replay a recorded capture into WATCH_DIR as CHUNK_TIME-second chunks at the
# capture's own pace, so 2_Stage2_Extraction.sh processes it exactly as it
# would a live capture from 1_Stage1_Capture.sh.
# Usage: ./0_replay_pcap.sh <capture.pcap>
# ==============================================================================
set -e
source ./config.env

INPUT_PCAP="$1"
if [[ -z "$INPUT_PCAP" || ! -f "$INPUT_PCAP" ]]; then
    echo "usage: $0 <capture.pcap>" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d ./replay_chunks.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir -p "$WATCH_DIR"

# editcap -i splits the capture on packet timestamps, one file per CHUNK_TIME
# seconds, numbered in time order.
editcap -i "$CHUNK_TIME" "$INPUT_PCAP" "$TMP_DIR/chunk.pcap"

for FILE in "$TMP_DIR"/chunk*.pcap; do
    # An atomic rename into WATCH_DIR fires inotifywait's moved_to in
    # 2_Stage2_Extraction.sh, never a partial-file close_write.
    mv "$FILE" "$WATCH_DIR/"
    echo "[*] replayed $(basename "$FILE")"
    sleep "$CHUNK_TIME"
done
echo "[*] replay finished"
