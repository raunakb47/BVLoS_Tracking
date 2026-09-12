#!/usr/bin/env python3
"""
Module: extract_ap_metadata.py
Extract SSID, advertised max transmit power and beacon RSSI from the Beacon
frames in one pcap chunk.

The AP-side counterpart to Wi-BFI/main.py, which only sees Compressed
Beamforming Reports sent client -> AP and so has no visibility into what the
AP broadcasts. Everything read here is plaintext an AP transmits continuously
to every listener, so it needs only the widened BPF filter in
1_Stage1_Capture.sh, no association or active probing.

Shells out to `tshark -T fields` rather than using pyshark's layer objects
like main.py does: several distinct tagged-parameter fields collapse to the
same ".all" leaf one level up in pyshark's tree, resolved by dict insertion
order rather than by which field was meant, which would extract the wrong
value on a different tshark version instead of failing. Wi-BFI's own
2_batch_extract.sh uses the same `-T fields` pattern for VHT MIMO control.

Field names verified against a hand-crafted 802.11 beacon (scapy Country +
Power Constraint + SSID elements) through tshark 4.2.2, not read off the spec:
  wlan.country_info.fnm.mtpl - Country element per-subband Maximum Transmit
                               Power Level (dBm). Present with or without a
                               Power Constraint element and not VHT/HE
                               specific, so it is the primary source for both
                               11ac and 11ax. A Country element may carry
                               several (first-channel, num-channels,
                               max-power) triplets; the first is taken
                               (-E occurrence=f) rather than matching the
                               operating channel.
  wlan.powercon.local        - Power Constraint local constraint (dB),
                               subtracted per 802.11's Local Maximum Transmit
                               Power = Country Max - Power Constraint.
  wlan.ssid                  - hex-encoded SSID bytes for hotspot_classifier.
No HE Transmit Power Envelope field exists in this Wireshark version's
dictionary (tshark -G fields); the VHT-specific wlan.vht.tpe.pwr_constr_* does
but is unused, since Country/Power-Constraint covers both standards.
"""
import sys
import json
import subprocess
import csv
import io
from importlib import import_module

tx_power_estimator = import_module('tx_power_estimator')

FIELDS = [
    "wlan.ta",
    "wlan.ssid",
    "wlan.country_info.fnm.mtpl",
    "wlan.powercon.local",
    "wlan_radio.signal_dbm",
]


def _run_tshark(pcap_file):
    cmd = [
        "tshark", "-r", pcap_file,
        "-Y", "wlan.fc.type==0 && wlan.fc.subtype==8",  # Beacon frames only
        "-T", "fields", "-E", "separator=\t", "-E", "occurrence=f",
    ]
    for field in FIELDS:
        cmd += ["-e", field]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def _decode_ssid(hex_bytes):
    if not hex_bytes:
        return None
    try:
        return bytes.fromhex(hex_bytes).decode("utf-8", errors="replace")
    except ValueError:
        return None


def extract(pcap_file, out_json):
    raw_output = _run_tshark(pcap_file)
    ap_metadata = {}

    for row in csv.reader(io.StringIO(raw_output), delimiter="\t"):
        if len(row) < len(FIELDS):
            continue
        ap_mac, ssid_hex, country_mtpl, power_constraint, rssi = row[:5]
        if not ap_mac:
            continue

        entry = ap_metadata.setdefault(ap_mac, {"ssid": None, "tx_power_dbm": None, "rssi_samples": []})

        if entry["ssid"] is None:
            entry["ssid"] = _decode_ssid(ssid_hex)

        if entry["tx_power_dbm"] is None and country_mtpl:
            try:
                country_max = float(country_mtpl)
                constraint = float(power_constraint) if power_constraint else 0.0
                entry["tx_power_dbm"] = country_max - constraint
            except ValueError:
                pass

        if rssi:
            try:
                entry["rssi_samples"].append(float(rssi))
            except ValueError:
                pass

    result = {
        mac: {
            "ssid": data["ssid"],
            "tx_power_dbm": data["tx_power_dbm"],
            # Averaged in the linear power domain (tx_power_estimator.mean_rssi_dbm),
            # not a naive dB mean, which understates true average received power for
            # a fading signal (Jensen's inequality) and would read as extra path loss.
            "beacon_rssi_mean": (
                tx_power_estimator.mean_rssi_dbm(data["rssi_samples"])
                if data["rssi_samples"] else None
            ),
        }
        for mac, data in ap_metadata.items()
    }

    with open(out_json, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
