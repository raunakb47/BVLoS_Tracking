#!/usr/bin/env python3
"""
Module: hotspot_classifier.py
Best-effort classification of whether an observed beamformer MAC is likely a
mobile/personal hotspot (phone, tablet, laptop tethering) rather than fixed
infrastructure, replacing the previous check ("HOT" in ap_mac or "MOB" in
ap_mac): MAC addresses are hex strings (0-9, a-f), and H/O/T/M are not hex
digits, so that substring test could never match anything -- it was
unreachable regardless of input.

Neither signal implemented here is exclusive proof of "mobile hotspot"; both
are returned with their actual reliability rather than presented as a
confirmed device-type detector.
"""
import re

# Default SSID naming conventions used out-of-the-box by common mobile
# hotspot implementations (iOS Personal Hotspot, Android tethering). A user
# who renames their hotspot defeats this -- it is a heuristic on default
# behavior, not a hard identifier, hence the MEDIUM (not HIGH) confidence
# returned when it matches.
_MOBILE_SSID_PATTERNS = [
    re.compile(r"'s iPhone$", re.IGNORECASE),
    re.compile(r"'s iPad$", re.IGNORECASE),
    re.compile(r"^AndroidAP", re.IGNORECASE),
    re.compile(r"^DIRECT-.*Android", re.IGNORECASE),
    re.compile(r"Galaxy.*Hotspot", re.IGNORECASE),
]

# Illustrative starting point only -- NOT a usable device-class database.
# Apple alone holds 1553 and Samsung 909 registered OUI blocks as of this
# writing (IEEE OUI registry, standards-oui.ieee.org); this dict covers a
# small handful, each confirmed by direct lookup rather than assumed from
# memory. A real deployment needs a properly populated OUI table (e.g.
# exported from standards-oui.ieee.org or a maintained mirror) loaded via
# load_mobile_oui_table() below. Shipping a large hardcoded table here
# without verifying every entry against that source risked silently
# misclassifying devices on a wrong guess, which is worse than this
# function honestly reporting UNKNOWN.
_ILLUSTRATIVE_MOBILE_OUIS = {
    "F0EE7A": "Apple",
    "A483E7": "Apple",
}


def _normalize_oui(mac_address):
    return mac_address.replace(":", "").replace("-", "").upper()[:6]


def load_mobile_oui_table(csv_path):
    """
    Load an (oui_hex, vendor_name) table from a two-column CSV -- the format
    exported by standards-oui.ieee.org and most MAC-vendor lookup services --
    for use as the oui_table argument to classify_beamformer(). Populating
    this from a real OUI export is required before the OUI signal below is
    anything more than illustrative.
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
      is_mobile  - best-effort bool guess; False means "no mobile signal
                   observed", NOT "confirmed fixed infrastructure AP" --
                   callers should not treat False as positive proof either.
      confidence - "MEDIUM" (SSID default-naming pattern matched), "LOW"
                   (OUI-only match), or "UNKNOWN" (neither available/matched)
      reason     - short human-readable justification, for logging/UI

    ssid requires Beacon/Probe Response capture, which this pipeline's
    Stage 1 does not yet perform (BFI action frames only) -- pass None until
    that capture-filter change lands; this degrades to the OUI table (or
    UNKNOWN with no table configured).
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
