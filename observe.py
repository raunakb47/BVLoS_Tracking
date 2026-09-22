#!/usr/bin/env python3
"""
Module: observe.py
Stage 2. Turn one pcap chunk into records appended to the observable log.

The log is the framework's only memory: a stage needing history reads further
back in it rather than carrying its own state.

Two readers feed it, each running only if the chunk holds anything for it.
Beamforming reports come from Wi-BFI's main.py as a subprocess, so Wi-BFI stays
a standalone tool and this module consumes its output contract rather than its
internals; the cost is one interpreter start per feedback type per chunk, about
88 ms. Beacons and Probe Responses are parsed here: the body is a fixed 12-byte
prefix then information elements, so a chunk with no beacons costs nothing
rather than failing on a missing tool.

Frames the card flagged as failing FCS are dropped. The CRC failure invalidates
the frame body, including the addresses identifying whose reading it was;
keeping them populates the AP registry with phantoms.

Output is append-only, in two files:

  <prefix>.jsonl  one record per line: a source record per chunk, then one per
                  report and per beacon
  <prefix>.bin    V-matrices as raw bytes; each bfi record carries the offset,
                  length, shape and dtype to read its own back

V-matrices are held out of the JSON because they dominate volume and a consumer
scanning metadata should not have to parse them.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

# Management frame body: Timestamp(8) + Beacon Interval(2) + Capability(2),
# then information elements. Identical in Beacon and Probe Response.
_MGMT_HEADER_LEN = 24
_FIXED_BODY_LEN = 12
_IE_START = _MGMT_HEADER_LEN + _FIXED_BODY_LEN

_SUBTYPE_BEACON = 8
_SUBTYPE_PROBE_RESPONSE = 5
_SUBTYPE_ACTION = 13
_SUBTYPE_ACTION_NO_ACK = 14

_IE_SSID = 0
_IE_DS_PARAMETER_SET = 3
_IE_COUNTRY = 7
_IE_POWER_CONSTRAINT = 32

def _import_capture_reader(wibfi_dir):
    """
    Import Wi-BFI's capture_reader rather than copying it, so frame iteration,
    the radiotap walk and the FCS flag cannot drift between repositories.
    """
    if wibfi_dir not in sys.path:
        sys.path.insert(0, wibfi_dir)
    import capture_reader
    return capture_reader


def _mac(raw):
    return ":".join(f"{byte:02x}" for byte in raw)


def _information_elements(frame):
    """
    Walk the length-prefixed element list of a Beacon or Probe Response body.

    Stops at the first element claiming more bytes than the frame holds;
    truncation and a corrupt length are indistinguishable and both end the walk.
    """
    elements = {}
    offset = _IE_START
    while offset + 2 <= len(frame):
        element_id, length = frame[offset], frame[offset + 1]
        if offset + 2 + length > len(frame):
            break
        elements.setdefault(element_id, frame[offset + 2:offset + 2 + length])
        offset += 2 + length
    return elements


def _advertised_tx_power_dbm(elements):
    """
    Advertised transmit ceiling as Country maximum - Power Constraint, per
    802.11 Local Maximum Transmit Power. Returns (dbm, country_code), either
    part None when absent.

    The Country element holds a 3-byte country string then (first channel,
    channel count, max power) triplets; the first triplet is taken.

    A regulatory ceiling, not an instantaneous measurement: an AP running
    802.11h transmit power control sits below it with no further signalling.
    """
    country = elements.get(_IE_COUNTRY)
    if country is None or len(country) < 6:
        return None, None
    country_code = country[:2].decode("ascii", errors="replace")
    max_power_dbm = float(country[5])
    constraint = elements.get(_IE_POWER_CONSTRAINT)
    if constraint:
        max_power_dbm -= float(constraint[0])
    return max_power_dbm, country_code


def scan_chunk(path, capture_reader):
    """
    One walk over the chunk, returning (feedback types present, beacon records).

    The feedback types let the extractor run only for a type that has frames;
    an empty run still pays the interpreter start. Combining both answers in one
    walk makes the first free, since the walk is needed for the beacons.

    Reports name the AP only as a receiver address and never measure it, so
    beacons are the only monitor-side signal measurement of an AP available.
    """
    feedback_present = set()
    beacons = []
    for timestamp, buf in capture_reader._records(path):
        if len(buf) < 8:
            continue
        radiotap_len = struct.unpack_from("<H", buf, 2)[0]
        frame = buf[radiotap_len:]
        # Management header plus an Action frame's category and action code;
        # the beacon branch needs more and checks for itself.
        if len(frame) < _MGMT_HEADER_LEN + 2:
            continue
        frame_control = frame[0]
        if frame_control & 0x0c:                       # management frames only
            continue
        subtype = frame_control >> 4

        if subtype in (_SUBTYPE_ACTION, _SUBTYPE_ACTION_NO_ACK):
            category = frame[24]
            if (category in capture_reader.STANDARD_FOR_CATEGORY
                    and frame[25] == capture_reader.ACTION_COMPRESSED_BEAMFORMING
                    and not capture_reader.radiotap_bad_fcs(buf)):
                reported = capture_reader.feedback_type_of(
                    frame, capture_reader.STANDARD_FOR_CATEGORY[category])
                if reported is not None:
                    feedback_present.add(reported)
            continue

        if subtype not in (_SUBTYPE_BEACON, _SUBTYPE_PROBE_RESPONSE):
            continue
        if len(frame) < _IE_START:
            continue
        if capture_reader.radiotap_bad_fcs(buf):
            continue

        elements = _information_elements(frame)
        ssid_bytes = elements.get(_IE_SSID)
        channel = elements.get(_IE_DS_PARAMETER_SET)
        tx_power_dbm, country_code = _advertised_tx_power_dbm(elements)
        signal_chains = [float(v) for v in capture_reader.radiotap_signal_dbm(buf)]

        beacons.append({
            "kind": "beacon" if subtype == _SUBTYPE_BEACON else "probe_response",
            "t": timestamp,
            "bssid": _mac(frame[16:22]),
            "transmitter": _mac(frame[10:16]),
            # A zero-length SSID element means hidden, which differs from
            # no element at all.
            "ssid": (ssid_bytes.decode("utf-8", errors="replace")
                     if ssid_bytes is not None else None),
            "ssid_hidden": ssid_bytes is not None and len(ssid_bytes) == 0,
            "channel": int(channel[0]) if channel else None,
            "tx_power_dbm": tx_power_dbm,
            "country": country_code,
            "rssi": signal_chains[0] if signal_chains else None,
            "chains": signal_chains,
            "element_ids": sorted(elements),
        })

    return feedback_present, beacons


def _run_extractor(pcap_path, feedback, wibfi_dir, scratch):
    """
    Run Wi-BFI's main.py for one feedback type; returns its bucketed output or
    None.

    The three arguments Wi-BFI documents as ignored are passed as placeholders,
    since standard, configuration and width are decoded per packet. Feedback
    type stays live: SU and MU quantise the Givens angles differently.
    """
    # The extractor runs with cwd set to its own directory for its sibling
    # imports, so a relative capture path must be resolved before that.
    pcap_path = os.path.abspath(pcap_path)
    v_path = os.path.join(scratch, f"v_{feedback}.npy")
    angles_path = os.path.join(scratch, f"angles_{feedback}.npy")
    command = [
        sys.executable, os.path.join(wibfi_dir, "main.py"), pcap_path,
        "x", feedback, "x", "x", str(10 ** 9), v_path, angles_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=wibfi_dir)
    if result.returncode != 0:
        print(f"[!] extractor failed for {feedback}: {result.stderr.strip()}",
              file=sys.stderr)
        return None
    if result.stderr.strip():
        # Per-packet skips: undefined width, short angle payload.
        for line in result.stderr.strip().splitlines():
            print(f"[extractor/{feedback}] {line}", file=sys.stderr)
    if not os.path.exists(v_path):
        return None
    return np.load(v_path, allow_pickle=True).item()


def bfi_records(pcap_path, wibfi_dir, scratch, feedback_types):
    """
    Yield one record per beamforming report with its V-matrix.

    Fields encoded in the bucket key are repeated on the record, so a consumer
    need not parse the key.
    """
    for feedback in sorted(feedback_types):
        buckets = _run_extractor(pcap_path, feedback, wibfi_dir, scratch)
        if not buckets:
            continue
        for bucket_key, samples in buckets.items():
            transmitter, receiver, shape = bucket_key.split("_")
            config, bw = shape.split("@")
            for sample in samples:
                if len(sample) < 6:
                    print(f"[!] {bucket_key}: sample has no metadata element; "
                          f"Wi-BFI predates the extended output tuple",
                          file=sys.stderr)
                    return
                timestamp, v_matrix, rssi, stream_snr, chains, meta = sample[:6]
                if meta.get("bad_fcs"):
                    continue
                yield {
                    "kind": "bfi",
                    "t": float(timestamp),
                    "transmitter": transmitter,
                    "beamformer": receiver,
                    "bucket": bucket_key,
                    "standard": meta["standard"],
                    "feedback": meta["feedback"],
                    "nr": meta["nr"],
                    "nc": meta["nc"],
                    "bw": meta["bw"],
                    "nsubc": meta["nsubc"],
                    "codebook": meta["codebook"],
                    "ng": meta["ng"],
                    # Sets the wavelength every bearing depends on. None when
                    # radiotap omits the Channel field.
                    "freq_mhz": meta.get("freq_mhz"),
                    "phi_bit": meta["phi_bit"],
                    "psi_bit": meta["psi_bit"],
                    "rssi": None if rssi is None else float(rssi),
                    "chains": [float(c) for c in chains],
                    "stream_snr": [float(s) for s in stream_snr],
                }, v_matrix


def observe(pcap_path, log_prefix, wibfi_dir):
    """
    Append one chunk's observations to the log.

    Returns per-kind counts plus a timing mapping. The timings separate the
    direct frame walk from the extractor subprocess, because only the second
    scales with report count and only the first is paid on an empty chunk.

    lag_s is the interval between the last frame in the chunk and the moment
    this stage finished with it: how far behind the air the log runs on a live
    capture, or the recording's age when a stored file is replayed.
    """
    started = time.perf_counter()
    capture_reader = _import_capture_reader(wibfi_dir)
    jsonl_path, bin_path = log_prefix + ".jsonl", log_prefix + ".bin"
    counts = {"bfi": 0, "beacon": 0, "probe_response": 0}
    timing = {}

    with tempfile.TemporaryDirectory() as scratch, \
            open(jsonl_path, "a") as log, open(bin_path, "ab") as blob:
        # Offsets recorded below are absolute within the file.
        blob.seek(0, os.SEEK_END)

        mark = time.perf_counter()
        feedback_types, beacons = scan_chunk(pcap_path, capture_reader)
        timing["scan_s"] = time.perf_counter() - mark

        latest = max((b["t"] for b in beacons), default=None)

        log.write(json.dumps({
            "kind": "source",
            "pcap": os.path.abspath(pcap_path),
            "bytes": os.path.getsize(pcap_path),
            "feedback_types": sorted(feedback_types),
        }) + "\n")

        mark = time.perf_counter()
        for record, v_matrix in bfi_records(pcap_path, wibfi_dir, scratch,
                                            feedback_types):
            payload = np.ascontiguousarray(v_matrix).tobytes()
            record["v"] = {
                "offset": blob.tell(),
                "bytes": len(payload),
                "shape": list(v_matrix.shape),
                "dtype": str(v_matrix.dtype),
            }
            blob.write(payload)
            log.write(json.dumps(record) + "\n")
            counts["bfi"] += 1
            if latest is None or record["t"] > latest:
                latest = record["t"]
        timing["extract_s"] = time.perf_counter() - mark

        for record in beacons:
            log.write(json.dumps(record) + "\n")
            counts[record["kind"]] += 1

    timing["total_s"] = time.perf_counter() - started
    timing["lag_s"] = (time.time() - latest) if latest is not None else None
    return counts, timing


def read_v_matrix(record, log_prefix):
    """Read one record's V-matrix back out of the companion blob."""
    location = record["v"]
    with open(log_prefix + ".bin", "rb") as blob:
        blob.seek(location["offset"])
        raw = blob.read(location["bytes"])
    return np.frombuffer(raw, dtype=location["dtype"]).reshape(location["shape"])


def read_log(log_prefix):
    """Yield every record in the log, in the order it was appended."""
    with open(log_prefix + ".jsonl") as log:
        for line in log:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: observe.py <chunk.pcap> <log_prefix> [wibfi_dir] [timing.jsonl]",
              file=sys.stderr)
        raise SystemExit(2)
    wibfi = os.path.abspath(sys.argv[3] if len(sys.argv) > 3
                            else os.environ.get("WIBFI_DIR", "../Wi-BFI"))
    counts, timing = observe(sys.argv[1], sys.argv[2], wibfi)
    lag = "" if timing["lag_s"] is None else f"  lag={timing['lag_s']:.2f}s"
    print(f"[stage2] {os.path.basename(sys.argv[1])}: "
          + "  ".join(f"{k}={v}" for k, v in counts.items())
          + f"   scan={timing['scan_s'] * 1e3:.0f}ms"
          + f" extract={timing['extract_s'] * 1e3:.0f}ms"
          + f" total={timing['total_s'] * 1e3:.0f}ms{lag}")
    if len(sys.argv) > 4:
        with open(sys.argv[4], "a") as handle:
            handle.write(json.dumps({"stage": 2, "chunk": os.path.basename(sys.argv[1]),
                                     "counts": counts, "timing": timing}) + "\n")
