#!/usr/bin/env python3
"""
Module: extract_ap_metadata.py
Extracts AP-side metadata (SSID, advertised max transmit power, and beacon
RSSI at the monitor card) from Beacon frames in a captured pcap chunk.

This is the AP-beacon counterpart to Wi-BFI/main.py's client-side BFI
extraction: main.py only ever sees Compressed Beamforming Report frames
(sent client -> AP), so it has no visibility into anything the AP itself
broadcasts. Every field pulled here is standards-mandated plaintext content
an AP transmits in the clear, continuously, to every listener regardless of
association -- capturing it needs only the widened BPF filter in
1_Stage1_Capture.sh (Beacon/Probe Response added alongside BFI action
frames), no association or active probing.

Shells out to `tshark -T fields` directly rather than using pyshark's
FileCapture object model that the sibling main.py uses for BFI extraction:
field access on pyshark's nested layer objects turned out to be ambiguous
for these specific fields when tested against a hand-crafted synthetic
beacon frame -- multiple distinct tagged-parameter fields collapse to the
same ".all" leaf name one level up in pyshark's layer tree, resolved by
Python dict-insertion order rather than by which field was actually meant,
which is exactly the kind of thing that would silently extract the wrong
value on a different tshark version rather than fail loudly. `tshark -T
fields -e <name>` sidesteps that layer-object model entirely and is the
same pattern Wi-BFI's own 2_batch_extract.sh already uses for VHT MIMO
control fields.

Field names verified against a real, hand-crafted 802.11 beacon frame
(scapy-generated Country + Power Constraint + SSID elements) fed through
tshark 4.2.2, not assumed from the 802.11 spec text alone:
  wlan.country_info.fnm.mtpl - Country element's per-subband Maximum
                                Transmit Power Level (dBm); present
                                whether or not a Power Constraint element
                                also is, and not VHT/HE-specific, so this
                                is the primary source for both 802.11ac
                                and 802.11ax APs. A Country element can
                                carry multiple (first-channel, num-channels,
                                max-power) triplets; the first is used
                                (-E occurrence=f) rather than matching the
                                current operating channel specifically, a
                                simplification documented here rather than
                                silently made.
  wlan.powercon.local         - Power Constraint element's local
                                constraint (dB), subtracted from the
                                Country element's value per 802.11's
                                Local Maximum Transmit Power = Country Max
                                - Power Constraint relation.
  wlan.ssid                   - hex-encoded SSID bytes, for
                                hotspot_classifier.py's SSID-pattern signal.
No HE-specific Transmit Power Envelope field was found in this Wireshark
version's field dictionary (tshark -G fields), so this does not depend on
one existing; a VHT-specific field (wlan.vht.tpe.pwr_constr_*) does exist
but is not used here since the Country/Power-Constraint pair already
covers both standards generically.
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
