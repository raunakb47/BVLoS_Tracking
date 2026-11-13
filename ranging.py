#!/usr/bin/env python3
"""
Module: ranging.py
Stage 4. Distance from the monitor to a transmitter, from received power,
under a published site-general path-loss model.

Model (ITU-R P.1238-13 eq. 1 indoor; P.1411-13 eq. 1 outdoor; 3GPP TR 38.901
Table 7.4.1-1 InH-Office, same form):

    L(d, f) = 10 alpha log10(d) + beta + 10 gamma log10(f_GHz) + X,  X ~ N(0, sigma)

with d the 3D distance in metres. alpha, beta, gamma and sigma are fitted per
environment and per LoS/NLoS, and each fit holds over a stated distance and
frequency range (MODELS). PATH_LOSS_MODEL in config.env picks the row.
The 38.901 NLoS loss is max(LoS, NLoS') (LOS_FLOOR), so its inverse is the
nearer of the two rows' distances.

Observed loss is L = P_tx - RSSI. Inverting the median gives median_m; the
band [near_m, far_m] is the median inverted at L -/+ k sigma (RANGE_SIGMA_K).
A distance outside the model's range is not extrapolated: it is reported as
None, and "outside" says on which side the median fell.

P_tx of an AP ranging itself (ap_range()) is first the power its own Beacon
or Probe Response states in a TPC Report element: the power used for that
frame, within +/-5 dB of its EIRP (IEEE 802.11 TPC Report element). Without
one, and for every client, P_tx comes from data, first source found
(tx_power()):
  1. the client's own Power Capability element (Association Request), the
     most it can transmit (Stage 3 fact tx_power_capability_dbm)
  2. the Country element of its AP (Beacon, Probe Response)
  3. any Country element heard in the same band
  4. the monitor host's regulatory domain (iw reg get, logged by the watcher)
Every source is a maximum; a transmitter below it is nearer than reported.
The source is carried with every range. Absent all four there is no range.

Antenna gains are folded into P_tx: the limits are EIRP.

Distance model fit (config.env, the default), used ahead of the above when set:
  SITE_RSSI_1M_DBM, SITE_PL_N, SITE_SIGMA_DB, SITE_VALID_M
      a fit of the monitor's RSSI against tape-measured distance for a
      calibration transmitter, RSSI = RSSI_1m - 10 n log10(d). Every
      transmitter, client or AP, is ranged with it, which assumes its
      effective power equals the calibration transmitter's. Only distances
      inside SITE_VALID_M are given. SITE_FIT_NAME labels the fit.
  AP_DISTANCE_M
      measured distance from the monitor to each AP, "bssid=metres;...". An
      AP listed there takes that distance.
"""
import os
import re

import numpy as np

# Site-general coefficients. Indoor: ITU-R P.1238-13 (09/2025) Table 2, both
# stations on the same floor. Outdoor: ITU-R P.1411-13 (09/2025) Table 4,
# below-rooftop (street level). f_ghz and d_m are each fit's stated ranges.
MODELS = {
    "office_los":         ("P.1238-13", 1.47, 34.17, 2.08, 3.68, (0.3, 294.0), (2, 27)),
    "office_nlos":        ("P.1238-13", 2.39, 30.13, 2.40, 5.01, (0.3, 255.0), (4, 30)),
    "corridor_los":       ("P.1238-13", 1.57, 29.46, 2.24, 3.77, (0.3, 300.0), (2, 160)),
    "corridor_nlos":      ("P.1238-13", 2.78, 28.62, 2.54, 7.58, (0.625, 159.0), (3, 94)),
    "industrial_los":     ("P.1238-13", 2.27, 24.79, 2.10, 2.62, (0.625, 294.0), (2, 102)),
    "industrial_nlos":    ("P.1238-13", 2.80, 23.55, 2.16, 5.70, (0.625, 255.0), (3, 110)),
    "conference_los":     ("P.1238-13", 1.56, 30.47, 2.23, 2.92, (0.45, 300.0), (2, 21)),
    "conference_nlos":    ("P.1238-13", 1.40, 39.53, 2.37, 3.33, (0.45, 159.0), (4, 25)),
    "urban_los":          ("P.1411-13", 2.07, 31.23, 2.06, 4.91, (0.45, 300.0), (5, 660)),
    "urban_highrise_nlos": ("P.1411-13", 3.73, 16.02, 2.26, 7.62, (0.8, 159.0), (20, 715)),
    "suburban_nlos":      ("P.1411-13", 4.52, 6.04, 2.14, 8.02, (0.45, 255.0), (10, 250)),
    "residential_nlos":   ("P.1411-13", 3.01, 18.8, 2.07, 3.07, (0.8, 73.0), (30, 170)),
    # 3GPP TR 38.901 V16.1.0 (ETSI TR 138 901, 2020-11) Table 7.4.1-1, InH-Office:
    # LoS 32.4 + 17.3 log10(d) + 20 log10(f); NLoS' 17.30 + 38.3 log10(d) + 24.9 log10(f).
    "office_38901_los":   ("TR 38.901", 1.73, 32.4, 2.0, 3.0, (0.5, 100.0), (1, 150)),
    "office_38901_nlos":  ("TR 38.901", 3.83, 17.30, 2.49, 8.03, (0.5, 100.0), (1, 150)),
}

# NLoS rows whose loss is never below a LoS row's: PL = max(PL_LoS, PL_NLoS').
LOS_FLOOR = {"office_38901_nlos": "office_38901_los"}

# 802.11 band edges, MHz. A Country element heard in one band says nothing
# about another's limits.
BANDS_MHZ = ((2400.0, 2500.0), (5150.0, 5925.0), (5925.0, 7125.0))

_RULE = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*@[^)]*\),\s*"
                   r"\([^,]*,\s*(\d+(?:\.\d+)?)\s*(mBm)?\s*\)")


def model_from_env(environ=None):
    """(name, coefficients) for PATH_LOSS_MODEL; raises on an unknown name."""
    environ = os.environ if environ is None else environ
    name = environ.get("PATH_LOSS_MODEL")
    if name not in MODELS:
        raise ValueError(f"PATH_LOSS_MODEL={name!r} is not one of {sorted(MODELS)}")
    return name, MODELS[name]


def sigma_k(environ=None):
    """RANGE_SIGMA_K from config.env; raises unless a positive number."""
    environ = os.environ if environ is None else environ
    value = environ.get("RANGE_SIGMA_K")
    try:
        k = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"RANGE_SIGMA_K={value!r} is not a number") from None
    if not k > 0:
        raise ValueError(f"RANGE_SIGMA_K={value!r} is not positive")
    return k


def band_of(freq_mhz):
    """Index into BANDS_MHZ, or None outside every band."""
    if freq_mhz is None:
        return None
    for index, (low, high) in enumerate(BANDS_MHZ):
        if low <= freq_mhz < high:
            return index
    return None


def parse_regdomain(text):
    """
    Frequency rules [(start MHz, end MHz, max EIRP dBm)] from `iw reg get`
    output. Only the global section is read; per-phy sections follow it.
    """
    rules = []
    for line in text.splitlines():
        if line.startswith("phy#"):
            break
        match = _RULE.search(line)
        if match:
            eirp = float(match.group(3)) / (100.0 if match.group(4) else 1.0)
            rules.append((float(match.group(1)), float(match.group(2)), eirp))
    return rules


def load_regdomain(path):
    """Rules from a logged `iw reg get`; empty when the file is absent."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as handle:
        return parse_regdomain(handle.read())


def power_ceiling(freq_mhz, beamformer, ap_records, regdomain):
    """
    (ceiling dBm, source) for a transmitter heard at freq_mhz whose AP is
    beamformer (an AP ranging itself passes its own address). source is
    "own_ap", "band", "host_regdomain" or None with a None ceiling.
    """
    band = band_of(freq_mhz)
    own = [r["tx_power_dbm"] for r in ap_records
           if r.get("tx_power_dbm") is not None and beamformer in r.get("beamformers", ())]
    if own:
        return own[-1], "own_ap"
    heard = [r["tx_power_dbm"] for r in ap_records
             if r.get("tx_power_dbm") is not None and band is not None
             and band_of(r.get("freq_mhz")) == band]
    if heard:
        return max(heard), "band"
    if freq_mhz is not None:
        covering = [eirp for start, end, eirp in regdomain if start <= freq_mhz <= end]
        if covering:
            return max(covering), "host_regdomain"
    return None, None


def tx_power(freq_mhz, beamformer, ap_records, regdomain, capability_dbm=None):
    """(dBm, source): the client's own capability first, else power_ceiling()."""
    if capability_dbm is not None:
        return float(capability_dbm), "client_capability"
    return power_ceiling(freq_mhz, beamformer, ap_records, regdomain)


def distance_for_loss_m(loss_db, freq_mhz, model):
    """Distance at which the model's median loss equals loss_db."""
    _, alpha, beta, gamma, _, _, _ = model
    exponent = (loss_db - beta - 10.0 * gamma * np.log10(freq_mhz / 1000.0)) / (10.0 * alpha)
    return float(10.0 ** exponent)


def _distance_m(loss_db, freq_mhz, model_name):
    """distance_for_loss_m under MODELS[model_name], with its LOS_FLOOR if any."""
    d = distance_for_loss_m(loss_db, freq_mhz, MODELS[model_name])
    floor = LOS_FLOOR.get(model_name)
    if floor is not None:
        d = min(d, distance_for_loss_m(loss_db, freq_mhz, MODELS[floor]))
    return d


def estimate(rssi_dbm, power_dbm, freq_mhz, model_name, k):
    """
    Range from one reading under MODELS[model_name]: median_m and the band
    [near_m, far_m] at -/+ k sigma. A value outside the model's distance range
    is None; "outside" is "below" or "above" when the median is, "frequency"
    when freq_mhz is outside the fit, else None. All None when an input is
    missing.
    """
    model = MODELS[model_name]
    recommendation, _, _, _, sigma, (f_low, f_high), (d_low, d_high) = model
    result = {"model": model_name, "recommendation": recommendation,
              "valid_m": [d_low, d_high], "sigma_db": sigma, "k": k,
              "loss_db": None, "median_m": None, "near_m": None, "far_m": None,
              "outside": None}
    if rssi_dbm is None or power_dbm is None or not freq_mhz:
        return result
    if not f_low <= freq_mhz / 1000.0 <= f_high:
        result["outside"] = "frequency"
        return result
    loss = power_dbm - rssi_dbm
    result["loss_db"] = loss

    def within(d):
        return d if d_low <= d <= d_high else None

    median = _distance_m(loss, freq_mhz, model_name)
    result["median_m"] = within(median)
    result["near_m"] = within(_distance_m(loss - k * sigma, freq_mhz, model_name))
    result["far_m"] = within(_distance_m(loss + k * sigma, freq_mhz, model_name))
    if median < d_low:
        result["outside"] = "below"
    elif median > d_high:
        result["outside"] = "above"
    return result


def client_range(output, facts, ap_records, regdomain, model_name, k):
    """Range of a Stage 3 output's client from the monitor, with its power source."""
    summary = output.get("client_rssi")
    fit = site_fit()
    if fit is not None:
        return site_estimate(summary and summary["median_dbm"], fit, k)
    freq = facts.get("centre_freq_mhz")
    power, source = tx_power(freq, facts["beamformer"], ap_records, regdomain,
                             facts.get("tx_power_capability_dbm"))
    return dict(estimate(summary and summary["median_dbm"], power, freq, model_name, k),
                power_dbm=power, power_source=source)


def ap_range(ap_record, ap_records, regdomain, model_name, k):
    """Range of a Stage 3 "ap" record's AP from the monitor, with its power source."""
    summary = ap_record.get("rssi")
    freq = ap_record.get("freq_mhz")
    measured = measured_ap_distances().get(ap_record["bssid"].lower())
    if measured is not None:
        return {"model": "measured", "recommendation": "site input", "valid_m": None,
                "sigma_db": None, "k": k, "loss_db": None, "median_m": measured,
                "near_m": measured, "far_m": measured, "outside": None,
                "power_dbm": None, "power_source": "site_input"}
    fit = site_fit()
    if fit is not None:
        return site_estimate(summary and summary["median_dbm"], fit, k)
    if ap_record.get("tpc_tx_power_dbm") is not None:
        power, source = float(ap_record["tpc_tx_power_dbm"]), "own_ap_tpc"
        return dict(estimate(summary and summary["median_dbm"], power, freq, model_name, k),
                    power_dbm=power, power_source=source)
    power, source = power_ceiling(freq, ap_record["bssid"],
                                  [dict(ap_record, beamformers=[ap_record["bssid"]])]
                                  + list(ap_records), regdomain)
    return dict(estimate(summary and summary["median_dbm"], power, freq, model_name, k),
                power_dbm=power, power_source=source)


def _float_env(environ, key):
    value = (environ.get(key) or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{key}={value!r} is not a number") from None


def site_fit(environ=None):
    """The calibration fit from SITE_* in config.env, or None when unset."""
    environ = os.environ if environ is None else environ
    rssi_1m, n = _float_env(environ, "SITE_RSSI_1M_DBM"), _float_env(environ, "SITE_PL_N")
    if rssi_1m is None or n is None:
        return None
    if n <= 0:
        raise ValueError(f"SITE_PL_N={n} must be positive")
    sigma = _float_env(environ, "SITE_SIGMA_DB") or 0.0
    valid = (environ.get("SITE_VALID_M") or "").strip()
    try:
        low, high = (float(v) for v in valid.split("-")) if valid else (0.0, float("inf"))
    except ValueError:
        raise ValueError(f"SITE_VALID_M={valid!r} is not 'low-high' in metres") from None
    name = (environ.get("SITE_FIT_NAME") or "").strip() or "site_fit"
    return {"name": name, "rssi_1m_dbm": rssi_1m, "n": n, "sigma_db": sigma, "valid_m": [low, high]}


def site_estimate(rssi_dbm, fit, k):
    """Range from one reading under the site fit, in estimate()'s form."""
    low, high = fit["valid_m"]
    result = {"model": fit.get("name", "site_fit"), "recommendation": "site calibration",
              "valid_m": [low, high], "sigma_db": fit["sigma_db"], "k": k,
              "loss_db": None, "median_m": None, "near_m": None, "far_m": None,
              "outside": None, "power_dbm": None, "power_source": "site_fit"}
    if rssi_dbm is None:
        return result

    def distance(rssi):
        return float(10.0 ** ((fit["rssi_1m_dbm"] - rssi) / (10.0 * fit["n"])))

    def within(d):
        return d if low <= d <= high else None

    median = distance(rssi_dbm)
    result["median_m"] = within(median)
    result["near_m"] = within(distance(rssi_dbm + k * fit["sigma_db"]))
    result["far_m"] = within(distance(rssi_dbm - k * fit["sigma_db"]))
    if median < low:
        result["outside"] = "below"
    elif median > high:
        result["outside"] = "above"
    return result


def measured_ap_distances(environ=None):
    """{bssid: metres} from AP_DISTANCE_M ("bssid=metres;...")."""
    environ = os.environ if environ is None else environ
    distances = {}
    for part in filter(None, (p.strip() for p in (environ.get("AP_DISTANCE_M") or "").split(";"))):
        bssid, _, metres = part.partition("=")
        try:
            distances[bssid.strip().lower()] = float(metres)
        except ValueError:
            raise ValueError(f"AP_DISTANCE_M entry {part!r} is not bssid=metres") from None
    return distances
