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
frequency range (MODELS). PATH_LOSS_MODEL in config.env picks one row of
MODELS, one row of TGAX or one fit of PATH_LOSS_FITS.
The 38.901 NLoS loss is max(LoS, NLoS') (LOS_FLOOR), so its inverse is the
nearer of the two rows' distances.

TGAX rows are the IEEE 802.11ax indoor model (IEEE 802.11-14/0882r4, Sec. 3.1
and Table III, adopting the TGn models B and D):

    L(d, f) = L_FS(d, f)                               d <= d_BP
    L(d, f) = L_FS(d_BP, f) + 35 log10(d / d_BP)       d >  d_BP
    plus m L_iw for m walls (PATH_LOSS_WALLS)

with L_FS the free-space loss. Shadowing sigma is 3 dB up to d_BP and the
row's NLoS value beyond it. The document gives no distance limit, so a TGax
range is bounded only below, at 1 m.

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

Monitor check (config.env MONITOR_CHECK_RSSI_1M_DBM, MONITOR_CHECK_FREQ_MHZ):
  the median RSSI the monitor measures from a transmitter held 1 m away
  (monitor_check.py). When set, the published models take their power from it
  instead of the chain above: P_tx = RSSI_1m + L(1 m, f_check), so the selected
  model reproduces the check reading at 1 m. That one effective power stands
  for the transmitter's power, its antenna and the monitor's RSSI offset, which
  the stated maxima leave out; it assumes every transmitter radiates like the
  check transmitter.

Measured fits (config.env PATH_LOSS_FITS):
  "name: RSSI_1m n sigma low-high; ..." declares fits of the monitor's RSSI
  against tape-measured distance for a calibration transmitter,
  RSSI = RSSI_1m - 10 n log10(d), with residual spread sigma (dB) and the
  span low-high (m) in which the fit is applied. When PATH_LOSS_MODEL names a
  fit, every transmitter, client or AP, is ranged with it. That assumes its
  effective power equals the calibration transmitter's, so no P_tx is needed.
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

# IEEE 802.11-14/0882r4 Table III and Sec. 3.1. Fields: source, d_BP (m),
# sigma up to d_BP, sigma beyond d_BP (dB), wall loss L_iw (dB), f range (GHz),
# d range (m; no upper limit). Residential is simulation scenario #1 (model B,
# L_iw 5 dB); enterprise is scenario #2 (model D, L_iw 7 dB). The document's
# models are for the 2.4 and 5 GHz bands.
TGAX = {
    "tgax_residential": ("802.11-14/0882r4", 5.0, 3.0, 4.0, 5.0, (2.4, 5.925), (1, None)),
    "tgax_enterprise":  ("802.11-14/0882r4", 10.0, 3.0, 5.0, 7.0, (2.4, 5.925), (1, None)),
}

# NLoS rows whose loss is never below a LoS row's: PL = max(PL_LoS, PL_NLoS').
LOS_FLOOR = {"office_38901_nlos": "office_38901_los"}

# 802.11 band edges, MHz. A Country element heard in one band says nothing
# about another's limits.
BANDS_MHZ = ((2400.0, 2500.0), (5150.0, 5925.0), (5925.0, 7125.0))

_RULE = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*@[^)]*\),\s*"
                   r"\([^,]*,\s*(\d+(?:\.\d+)?)\s*(mBm)?\s*\)")


def model_from_env(environ=None):
    """
    (name, definition) for PATH_LOSS_MODEL: a MODELS row, or a fit dict from
    PATH_LOSS_FITS. Raises on an unknown name.
    """
    environ = os.environ if environ is None else environ
    name = environ.get("PATH_LOSS_MODEL")
    fits = fits_from_env(environ)
    if name in fits:
        return name, fits[name]
    if name in TGAX:
        return name, TGAX[name]
    if name not in MODELS:
        raise ValueError(f"PATH_LOSS_MODEL={name!r} is not one of "
                         f"{sorted(MODELS) + sorted(TGAX) + sorted(fits)}")
    return name, MODELS[name]


def walls_from_env(environ=None):
    """PATH_LOSS_WALLS: walls between monitor and transmitter, for TGAX rows; 0 when empty."""
    environ = os.environ if environ is None else environ
    value = (environ.get("PATH_LOSS_WALLS") or "").strip()
    if not value:
        return 0
    try:
        walls = int(value)
    except ValueError:
        raise ValueError(f"PATH_LOSS_WALLS={value!r} is not a whole number") from None
    if walls < 0:
        raise ValueError(f"PATH_LOSS_WALLS={value!r} is negative")
    return walls


def monitor_check_from_env(environ=None):
    """
    (RSSI_1m dBm, freq MHz) from MONITOR_CHECK_RSSI_1M_DBM and
    MONITOR_CHECK_FREQ_MHZ, or None when unset. Raises when only one is set.
    """
    environ = os.environ if environ is None else environ
    rssi = _float_env(environ, "MONITOR_CHECK_RSSI_1M_DBM")
    freq = _float_env(environ, "MONITOR_CHECK_FREQ_MHZ")
    if rssi is None and freq is None:
        return None
    if rssi is None or freq is None:
        raise ValueError("MONITOR_CHECK_RSSI_1M_DBM and MONITOR_CHECK_FREQ_MHZ "
                         "are set together or not at all")
    return rssi, freq


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


def free_space_db(d_m, freq_mhz):
    """Free-space loss, 20 log10(4 pi d f / c), with f in MHz."""
    return float(20.0 * np.log10(d_m) + 20.0 * np.log10(freq_mhz) - 27.55)


def tgax_loss_db(d_m, freq_mhz, row, walls):
    """Median loss of a TGAX row at d_m with walls walls."""
    _, d_bp, _, _, wall_db, _, _ = row
    loss = free_space_db(min(d_m, d_bp), freq_mhz)
    if d_m > d_bp:
        loss += 35.0 * np.log10(d_m / d_bp)
    return float(loss + walls * wall_db)


def tgax_distance_m(loss_db, freq_mhz, row, walls):
    """Distance at which a TGAX row's median loss equals loss_db."""
    _, d_bp, _, _, wall_db, _, _ = row
    excess = loss_db - walls * wall_db
    at_bp = free_space_db(d_bp, freq_mhz)
    if excess <= at_bp:
        return float(10.0 ** ((excess - free_space_db(1.0, freq_mhz)) / 20.0))
    return float(d_bp * 10.0 ** ((excess - at_bp) / 35.0))


def loss_db(d_m, freq_mhz, model_name, walls=0):
    """Median loss of a MODELS or TGAX row at d_m."""
    if model_name in TGAX:
        return tgax_loss_db(d_m, freq_mhz, TGAX[model_name], walls)

    def row_loss(row):
        _, alpha, beta, gamma, _, _, _ = row
        return 10.0 * alpha * np.log10(d_m) + beta + 10.0 * gamma * np.log10(freq_mhz / 1000.0)

    loss = row_loss(MODELS[model_name])
    floor = LOS_FLOOR.get(model_name)
    if floor is not None:
        loss = max(loss, row_loss(MODELS[floor]))
    return float(loss)


def check_power(check, model_name):
    """Effective power from the monitor check: RSSI_1m + L(1 m, f_check), no walls."""
    rssi_1m, freq = check
    return float(rssi_1m + loss_db(1.0, freq, model_name))


def distance_for_loss_m(loss_db, freq_mhz, model):
    """Distance at which the model's median loss equals loss_db."""
    _, alpha, beta, gamma, _, _, _ = model
    exponent = (loss_db - beta - 10.0 * gamma * np.log10(freq_mhz / 1000.0)) / (10.0 * alpha)
    return float(10.0 ** exponent)


def _distance_m(loss_db, freq_mhz, model_name, walls=0):
    """Inverse of loss_db(): a TGAX row, or a MODELS row with its LOS_FLOOR if any."""
    if model_name in TGAX:
        return tgax_distance_m(loss_db, freq_mhz, TGAX[model_name], walls)
    d = distance_for_loss_m(loss_db, freq_mhz, MODELS[model_name])
    floor = LOS_FLOOR.get(model_name)
    if floor is not None:
        d = min(d, distance_for_loss_m(loss_db, freq_mhz, MODELS[floor]))
    return d


def estimate(rssi_dbm, power_dbm, freq_mhz, model_name, k, walls=0):
    """
    Range from one reading under MODELS[model_name] or TGAX[model_name]:
    median_m and the band [near_m, far_m] at -/+ k sigma. A value outside the
    model's distance range is None; "outside" is "below" or "above" when the
    median is, "frequency" when freq_mhz is outside the fit, else None. All
    None when an input is missing. A TGAX row's sigma is the one on the
    median's side of d_BP; walls applies to TGAX rows only.
    """
    if model_name in TGAX:
        recommendation, d_bp, sigma_los, sigma_nlos, _, (f_low, f_high), (d_low, d_high) = TGAX[model_name]
        sigma = sigma_los
    else:
        recommendation, _, _, _, sigma, (f_low, f_high), (d_low, d_high) = MODELS[model_name]
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
        return d if d_low <= d and (d_high is None or d <= d_high) else None

    median = _distance_m(loss, freq_mhz, model_name, walls)
    if model_name in TGAX and median > d_bp:
        sigma = sigma_nlos
        result["sigma_db"] = sigma
    result["median_m"] = within(median)
    result["near_m"] = within(_distance_m(loss - k * sigma, freq_mhz, model_name, walls))
    result["far_m"] = within(_distance_m(loss + k * sigma, freq_mhz, model_name, walls))
    if median < d_low:
        result["outside"] = "below"
    elif d_high is not None and median > d_high:
        result["outside"] = "above"
    return result


def client_range(output, facts, ap_records, regdomain, model_name, k):
    """Range of a Stage 3 output's client from the monitor, with its power source."""
    summary = output.get("client_rssi")
    fit = selected_fit()
    if fit is not None:
        return fit_estimate(summary and summary["median_dbm"], fit, k)
    freq = facts.get("centre_freq_mhz")
    walls = walls_from_env()
    check = monitor_check_from_env()
    if check is not None:
        power, source = check_power(check, model_name), "monitor_check"
    else:
        power, source = tx_power(freq, facts["beamformer"], ap_records, regdomain,
                                 facts.get("tx_power_capability_dbm"))
    return dict(estimate(summary and summary["median_dbm"], power, freq, model_name, k, walls),
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
    fit = selected_fit()
    if fit is not None:
        return fit_estimate(summary and summary["median_dbm"], fit, k)
    walls = walls_from_env()
    check = monitor_check_from_env()
    if check is not None:
        power, source = check_power(check, model_name), "monitor_check"
    elif ap_record.get("tpc_tx_power_dbm") is not None:
        power, source = float(ap_record["tpc_tx_power_dbm"]), "own_ap_tpc"
    else:
        power, source = power_ceiling(freq, ap_record["bssid"],
                                      [dict(ap_record, beamformers=[ap_record["bssid"]])]
                                      + list(ap_records), regdomain)
    return dict(estimate(summary and summary["median_dbm"], power, freq, model_name, k, walls),
                power_dbm=power, power_source=source)


def _float_env(environ, key):
    value = (environ.get(key) or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{key}={value!r} is not a number") from None


def fits_from_env(environ=None):
    """{name: fit} from PATH_LOSS_FITS ("name: RSSI_1m n sigma low-high; ...")."""
    environ = os.environ if environ is None else environ
    fits = {}
    for part in filter(None, (p.strip() for p in (environ.get("PATH_LOSS_FITS") or "").split(";"))):
        name, _, values = part.partition(":")
        name = name.strip()
        try:
            rssi_1m, n, sigma, span = values.split()
            low, high = (float(v) for v in span.split("-"))
            fit = {"name": name, "rssi_1m_dbm": float(rssi_1m), "n": float(n),
                   "sigma_db": float(sigma), "valid_m": [low, high]}
        except ValueError:
            raise ValueError(f"PATH_LOSS_FITS entry {part!r} is not "
                             "'name: RSSI_1m n sigma low-high'") from None
        if name in MODELS or name in TGAX:
            raise ValueError(f"PATH_LOSS_FITS name {name!r} is a published model's name")
        if fit["n"] <= 0:
            raise ValueError(f"PATH_LOSS_FITS {name!r}: n must be positive")
        fits[name] = fit
    return fits


def selected_fit(environ=None):
    """The fit PATH_LOSS_MODEL names, or None when it names a published model."""
    environ = os.environ if environ is None else environ
    return fits_from_env(environ).get(environ.get("PATH_LOSS_MODEL"))


def fit_estimate(rssi_dbm, fit, k):
    """Range from one reading under the site fit, in estimate()'s form."""
    low, high = fit["valid_m"]
    result = {"model": fit["name"], "recommendation": "measured fit",
              "valid_m": [low, high], "sigma_db": fit["sigma_db"], "k": k,
              "loss_db": None, "median_m": None, "near_m": None, "far_m": None,
              "outside": None, "power_dbm": None, "power_source": "fit"}
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
