#!/usr/bin/env python3
"""
Module: hotspot_classifier.py
Best-effort guess at whether a beamformer MAC is a mobile/personal hotspot
(phone, tablet, laptop tethering) rather than fixed infrastructure.

Neither signal used here proves it, so both are returned with their actual
reliability attached rather than as a device-type detection.
"""
import re

# Out-of-the-box SSID naming from iOS Personal Hotspot and Android tethering.
# Renaming the hotspot defeats this, hence MEDIUM rather than HIGH on a match.
_MOBILE_SSID_PATTERNS = [
    re.compile(r"'s iPhone$", re.IGNORECASE),
    re.compile(r"'s iPad$", re.IGNORECASE),
    re.compile(r"^AndroidAP", re.IGNORECASE),
    re.compile(r"^DIRECT-.*Android", re.IGNORECASE),
    re.compile(r"Galaxy.*Hotspot", re.IGNORECASE),
]

# Illustrative, not a device-class database: Apple holds 1553 and Samsung 909
# registered OUI blocks (standards-oui.ieee.org), and these two entries are
# the ones verified by direct lookup. A deployment needs a real export loaded
# through load_mobile_oui_table(); an unverified hardcoded table would
# misclassify silently, which is worse than reporting UNKNOWN.
_ILLUSTRATIVE_MOBILE_OUIS = {
    "F0EE7A": "Apple",
    "A483E7": "Apple",
}


def _normalize_oui(mac_address):
    return mac_address.replace(":", "").replace("-", "").upper()[:6]


def load_mobile_oui_table(csv_path):
    """
    Load an (oui_hex, vendor_name) two-column CSV, the format exported by
    standards-oui.ieee.org and most vendor lookup services, for
    classify_beamformer()'s oui_table.
    """
    table = {}
    with open(csv_path, newline="") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0]:
                table[_normalize_oui(parts[0])] = parts[1].strip()
    return table


def classify_beamformer(ap_mac, ssid=None, oui_table=None):
    """
    Returns (is_mobile, confidence, reason):
      is_mobile  - best-effort guess. False means no mobile signal was seen,
                   not that this is confirmed fixed infrastructure.
      confidence - "MEDIUM" (SSID pattern), "LOW" (OUI only), "UNKNOWN".
      reason     - short justification for logging or the UI.

    ssid comes from the AP registry and is None when no Beacon has been seen
    for that MAC yet, in which case only the OUI signal applies.
    """
    if ssid:
        for pattern in _MOBILE_SSID_PATTERNS:
            if pattern.search(ssid):
                return True, "MEDIUM", f"SSID '{ssid}' matches a default mobile-hotspot naming pattern"

    table = oui_table if oui_table is not None else _ILLUSTRATIVE_MOBILE_OUIS
    vendor = table.get(_normalize_oui(ap_mac))
    if vendor:
        return True, "LOW", f"OUI {_normalize_oui(ap_mac)} is registered to {vendor} (a mobile-device vendor; does not confirm this specific link is a hotspot)"

    return False, "UNKNOWN", "no mobile-hotspot signal observed (this does not confirm a fixed AP)"
