# BVLoS Handoff

Context carried forward so settled questions are not reopened. Two repositories,
both on branch `claude/sweet-cerf-p09l7d`:

- **`raunakb47/Wi-BFI`** — the extractor. Modified and working. Head `2b61ec3`.
- **`raunakb47/BVLoS_Tracking`** — the framework. **No file modified yet.** All
  work in section 4 is unstarted.

Numbers marked **[m]** were measured in the session that produced this file.
Unmarked numbers carry over from earlier sessions and are listed in section 6.

**Attribution discipline.** A position is attributed to the user only where
section 5 marks it **DECIDED** and quotes them. Everything else in this file is
a measurement or a proposal that the user has not ruled on, and must not be
restated later as something the user said, asked for, or expected.

---

## 1. Objectives

- Passive, unsupervised 2-D localization of Wi-Fi devices from plaintext
  Beamforming Feedback Information, through walls included.
- No room calibration. No association required; association is an optional
  accuracy upgrade, never a prerequisite.
- Clients already on the network. Channel chosen by scanning for activity, or
  set to a target SSID's channel.
- Fixed APs and mobile hotspots, 2×2 through 4×4.
- Extract as much positional information as the observables support, however
  infrequent the packets.
- Sessions of 15–30 minutes. Per-session MAC stability is sufficient.
- No hardcoded thresholds, no confidence scores, no speculative intermediate
  steps.

---

## 2. Modified Wi-BFI — complete workflow

Upstream is `kfoysalhaque/Wi-BFI`. The fork is a drop-in replacement: same CLI
signature, same output filenames, V-matrices bit-identical to upstream on every
trace tested.

### 2.1 File inventory

| file | lines | role |
|---|---|---|
| `capture_reader.py` | 235 | **new.** Frame reader: pcap/pcapng → frame records |
| `main.py` | 322 | driver: decode MIMO Control, slice angles, bucket, save |
| `bfi_angles.py` | 34 | unpack the angle bitfield into per-subcarrier angles |
| `vmatrices.py` | 361 | rebuild V from Givens rotations |
| `utils.py` | 28 | `hex2dec`, `flip_hex` |
| `1_capture.sh` | 25 | monitor-mode capture |
| `2_batch_extract.sh` | 91 | run the extractor + report the buckets |
| `3_visualize.py` | 199 | per-bucket diagnostic figures |

### 2.2 `capture_reader.py` — why it exists

**Need.** Upstream `main.py` used pyshark, which spawns a `tshark` subprocess,
loads the full dissector table, dissects every frame in the file, and returns
JSON that Python then re-parses. Everything the extractor actually consumes —
timestamp, transmitter, receiver, signal, and the frame bytes — sits at a fixed
offset in the radiotap or 802.11 header. Dissecting the frame to reach it buys
nothing and costs 328 ms per ten-second chunk against 0.1 ms for a direct read.

**Purpose.** Reads pcap and pcapng directly, either byte order, either timestamp
resolution, and yields one dict per compressed beamforming report.

**Selection logic.** Frames are chosen by **Action Category**, not by frame
subtype:

- management frames only, subtype 13 (Action) **or 14 (Action No Ack)** — real
  captures use 14, and a subtype-13-only match sees nothing;
- category `0x15` → VHT (`"AC"`), category `0x1e` → HE (`"AX"`), action `0x00`;
- optional `feedback_type` filter on the Feedback Type subfield (`"SU"`/`"MU"`).

Because the category byte *is* the standard, one capture may hold both VHT and
HE feedback and both decode in the same pass. This replaces a run-wide CLI
setting with a per-frame fact.

**Record yielded:**

```python
{"timestamp": float,        # epoch seconds
 "standard":  "AC"|"AX",
 "receiver":  "aa:bb:...",  # beamformer (AP)
 "transmitter": "aa:bb:...",# beamformee (client)
 "signal_dbm": [int, ...],  # every radiotap dBm Antenna Signal, in stored order
 "raw":       "hex string", # whole frame incl. radiotap
 "radiotap_len": int}
```

**`radiotap_signal_dbm(buf)`** walks the radiotap present-bit chain with a field
table covering bits 0–27, honouring alignment and namespace continuation. The
table must be complete even for fields whose values are unwanted: radiotap lays
fields out in bit order, so a missing entry leaves the cursor short and every
later field is read at the wrong offset. Bit 28 (variable-length TLVs) is a
bail-out. Validated against tshark on 36,172 frames.

### 2.3 `main.py` — input, processing, output

**CLI (8 positional args, unchanged from upstream):**

```
python3 main.py <pcap> <standard> <SU|MU> <config> <bw> <max_packets> <v_out.npy> <angles_out.npy>
```

`<standard>`, `<config>` and `<bw>` are **accepted and ignored** — all three are
now decoded per packet. They are retained only so the positional signature stays
valid for existing callers. `<SU|MU>` is still live: it selects the Givens angle
bit widths *and* filters which frames the reader yields.

**Processing, per frame:**

1. `capture_reader.beamforming_reports(file, mimo)` yields the record.
2. Signal: the full radiotap list is kept; `rssi = list[0]`. A header with no
   signal field gives `None`, which reaches consumers as NaN — never a stand-in
   value, because downstream code reads it as measured received power.
3. Radiotap length from bytes 2–3 → offset `i` into the hex string.
4. MIMO Control (VHT 3 bytes at `i+52`, HE 5 bytes at `i+52`), byte-reversed by
   `flip_hex`, so binary index *k* holds spec bit B(width−1−k):
   **Nc = B0–B2, Nr = B3–B5, channel width = B6–B7**; codebook at VHT B10 / HE B9.
   All read **per packet** — a capture can hold several configurations, and a
   wrong value corrupts the V-matrices rather than merely mislabelling them.
5. Subcarrier set from `subcarrier_indices(standard, bw)`. Undefined combination
   (e.g. AC at 160 MHz) → frame skipped with a diagnostic on stderr.
6. Average SNR: one signed byte per stream at the head of the Compressed
   Beamforming Report. **dB = 22 + 0.25 × value**, range −10 to 53.75 dB.
7. Angles: payload read from the end of the SNR block **to the end of the frame**
   — no fixed FCS trim. The payload is prefix-sliced to
   `tot_bits_users × NSUBC_VALID` bits, so anything trailing is already ignored
   (an FCS where the adapter appends one, the MU Delta SNR block on an MU report).
8. Short-payload guard: a frame carrying fewer angle bits than its configuration
   requires is reported with the exact shortfall and skipped.
9. `bfi_angles()` → `vmatrices()` → V of shape `(NSUBC_VALID, Nr, Nc)`.

**Bucketing.** Samples are grouped by

```
"{transmitter}_{receiver}_{Nr}x{Nc}@{bw}"
e.g. "f6:b1:4f:a6:7b:7c_78:0c:f0:7a:5e:6e_4x2@40"
```

Channel width belongs in the key or the stack is ragged. `@` keeps the key at
three `_`-separated fields with `{Nr}x{Nc}` still parseable.

**Output — two `.npy` files, each a pickled dict keyed by bucket:**

```python
v_matrix.npy : {bucket_key: [(timestamp, v_matrix, rssi, stream_snr, signal_chains), ...]}
angles.npy   : {bucket_key: [(timestamp, angle), ...]}
```

| element | type | meaning |
|---|---|---|
| `timestamp` | float | epoch seconds, from the capture record |
| `v_matrix` | complex128 `(NSUBC, Nr, Nc)` | columns orthonormal (verified 1.0 ± 1e-15) **[m]** |
| `rssi` | float or `None` | `signal_chains[0]`; **monitor → client** |
| `stream_snr` | tuple of float, len Nc | dB; **beamformer → beamformee (AP → client)** |
| `signal_chains` | tuple of float | every radiotap value, combined figure first |
| `angle` | `(NSUBC, n_angles)` | raw quantised Givens angles |

Load with `np.load(path, allow_pickle=True).item()`.

**Subcarrier counts** (no collision between standards at any width, so the count
alone identifies the standard):

| width | AC | AX |
|---|---|---|
| 20 | 52 | 64 |
| 40 | 108 | 122 |
| 80 | 234 | 250 |
| 160 | — | 500 |

### 2.4 The two power observables are independent

| | measures | measured by |
|---|---|---|
| `rssi` | transmitter → **monitor** | the capture card |
| `stream_snr` | **beamformer → beamformee** (AP → client) | the client itself |

Live capture, same monitor RSSI, very different link quality **[m]**:

```
46:41:67:3a:d1:d0   monitor -84 dBm   report SNR 29.9-32.3 dB   (far from monitor, near AP)
46:86:23:85:4b:fe   monitor -84 dBm   report SNR  5.5-13.3 dB   (far from both)
```

### 2.5 Divergences from upstream, with cause

| change | cause |
|---|---|
| channel width decoded per packet | a 20 MHz frame inside a 40 MHz capture killed the run (`invalid literal for int() with base 2: ''`) |
| fixed 4-byte FCS trim removed | AWUS036AXM reports `radiotap.flags.fcs = False`; the trim destroyed 32 bits of real feedback on every frame |
| MU accepted on 11ax | field layout and angle tables identical to VHT; upstream gated MU to VHT only |
| `capture_reader.py` replaces pyshark | 328 ms/chunk → 0.1 ms; also fixes Action No Ack selection and per-frame standard detection |
| signal taken from radiotap chain `[0]` not `[-1]` | last chain ran 3.5–13.9 dB low and varied by transmitter |
| missing signal → `None`, not `-65.0` | `-65.0` was invented, not upstream; indistinguishable downstream from a real reading |
| `hex2dec` = `int(x, 16)` | the wrap/join round trip is the identity on whitespace-free input; 10.47 s → 7.68 s |
| per-stream SNR stored | upstream slices `packet_snr` and never reads it — a dead assignment in both |
| full radiotap chain list stored | what `[0]` *means* is driver-dependent (section 5.3) |

**Verification.** V-matrices bit-identical to the pre-change extractor across
11ac SU, 11ac MU and 11ax SU at 20/40/80/160 MHz plus the live capture; every
trace decodes every frame with no skips. SNR decode checked against tshark's own
dissection on 1,800 frames, 0 mismatches **[m]**.

### 2.6 Workflow scripts

**`1_capture.sh`** — monitor-mode capture, 15 min default, into
`../bfi-workspace/captures/`.

```bash
tcpdump -i wlan0 -s 0 -n "wlan[24:2] == 0x1500 or wlan[24:2] == 0x1e00" -w "$FILENAME"
```

`wlan[24]` is the Action Category and `wlan[25]` the Action code, at the same
offset in Action and Action No Ack. **The old VHT-only filter matched 0 of the
581 HE reports in the bundled 11ax trace; the widened one matches all 581** **[m]**.
False positives are byte coincidences in data frames: 4 of 191,748 and 2 of 317 **[m]**.
**This filter still excludes beacons — see section 4.1.**

**`2_batch_extract.sh`** — one pass per feedback type, then a bucket report.
Takes an optional pcap path; otherwise the newest capture. `MAX_PACKETS` env var.

The old version enumerated `(MAC, BW, Nc, Nr)` tuples with tshark, cut a subset
pcap per tuple, and ran the extractor once per subset — necessary only because
upstream took one configuration per run and needed a MAC filter. Both are now
decoded per packet, so the subsets are redundant; it also passed 9 positional
args including the MAC, which no longer exists in the signature.

The bucket report checks each stack against the key naming it: array shape must
carry the key's `(Nr, Nc)`, and the subcarrier count must be one the key's width
defines. Output on the live capture **[m]**:

```
SU:
  6e:09:60:37:f3:63 -> 78:0c:f0:7a:5e:6e  4x2 @ 40 MHz  AC
    reports    176 over    290.9 s (0.60/s)  V(176, 108, 4, 2) ok
    monitor RSSI  -89.0 ..  -83.0 dBm (0 unmeasured)   report SNR 14.12 .. 20.38 dB
  ... 9 buckets, 315 reports, all ok, including one stray 4x2@20 frame
MU: no reports
```

**`3_visualize.py`** — one PNG per bucket, four panels: monitor RSSI vs time;
reported SNR per stream vs time; transmit-antenna power share of stream 1 vs
time; that share per subcarrier as a heatmap. Reads the bucketed dict directly
(the old version loaded a flat array plus a separate `rssi.csv`, neither of which
the extractor produces any more). Skips buckets with fewer than two reports.

The power share is well posed because V's columns are orthonormal, so the
per-antenna squared magnitudes of one column sum to 1 — measured at 1.0 ± 1e-15 **[m]**.

---

## 3. Framework architecture

Four stages, one append-only log as the sole state, calibration as an optional
input that raises the output tier rather than a prerequisite.

```
1 CAPTURE  →  2 OBSERVE  →  3 SOLVE  →  4 VIEW
   monitor      raw decode    dispatcher   plan view
   mode, BPF    AoD roots     + geometry   + diagnostics
   rotating     append-only   emits its
   pcap chunks  log           tier
                                  ↑
                        OPTIONAL site.json
                        walk: AP position, n
                        anchors: array geometry
```

Chunking is an I/O mechanism only. With the log as state, chunk duration stops
being an input to any algorithm.

### Stage 1 — Capture (air → rotating pcap chunks)

| script | status | note |
|---|---|---|
| `1_Stage1_Capture.sh` | fix filter | match on Action Category, not frame control; both categories; **plus beacons** |
| `0_replay_pcap.sh` | fix | watchdog listens for `close_write` but the script delivers with `mv`, which emits `MOVED_TO` only — has never triggered. Needs `-e close_write,moved_to` |

### Stage 2 — Observe (pcap chunk → append-only observable log)

Turns frames into a lossless, typed record of everything the air carried.
Nothing thresholded, nothing discarded, every field timestamped. This file is
the system's only memory, which is what lets the state store, the sliding window
and the coast counter disappear.

| module | status | note |
|---|---|---|
| `observe.py` | new | direct reader, no subprocess. Per report: timestamp, transmitter, beamformer, Nr, Nc, width, grouping, codebook, per-stream SNR, per-chain and combined RSSI, AoD candidate set with eigenvalues. Per beacon: BSSID, SSID, advertised TX power, per-chain RSSI. Absorbs `extract_ap_metadata.py` and `ap_registry.py` |
| `Wi-BFI/main.py` | **done** | used as a library, not a subprocess |
| `aoa.py` | new | angle estimation parameterised by array geometry as element positions, using each subcarrier's own wavelength. Returns the full candidate set with eigenvalues, never an argmax. Plus `ambiguity()` and `sensitivity()` computed from geometry alone |

### Stage 3 — Solve (log + optional `site.json` → positions)

| module | status | note |
|---|---|---|
| `dispatch.py` | new | selects which estimators are *applicable* from decoded metadata: Nr, Nc, subcarrier count, grouping, codebook, mutually-coherent packet count, whether a geometry is known. Capability checks, not thresholds |
| `solve.py` | new | always emits the angular product — bearing relative to the beamformer plus exact radial ratios from the dB legs. Adds a bearing line when `site.json` supplies AP position and array orientation. Absorbs the two propagation formulas from `tx_power_estimator.py`; the client transmit-power default and its DEFAULT tier are removed, not replaced |
| `calibrate.py` | new, optional | not in the live path. Closed-loop walk → AP position and path-loss exponent from beacons; anchor fit → array element positions from reports at known bearings. Both report the fit residual so a degenerate input is visible |

### Stage 4 — View

| module | status | note |
|---|---|---|
| `view.py` | new | renders each tier as what it is — a bearing line as a line, not a point pretending to be a fix. Diagnostic strip: BFI rate per client, candidate count, eigenvalue spread, coherent-packet count, which measurements were present |

Built **before** the ground-truth campaign: every remaining calibration question
is faster to answer with a picture than a log.

### Removed

`3_2_Kinematic_Tracker.py`, `hotspot_classifier.py`, `2_1_Temporal_Sanitizer.py`,
`state_store.py`, `3_Stage3_Localization.py`, `4_Stage4_Inference.py`,
`3_1_Spatial_Algorithms.py` (→ `aoa.py`), `ap_registry.py` and
`extract_ap_metadata.py` (→ `observe.py`), `tx_power_estimator.py` (formulas →
`solve.py`).

With them: `KE_THRESHOLD`, `CFAR_REFERENCE_CELLS`, `CFAR_PFA`, `TDT_MS`,
`WINDOW_CHUNKS`, `TRACK_COAST_LIMIT`, `STATE_FILE`, `TRACK_STATE_FILE`,
`PACKET_STARVATION_LIMIT`, `SSE_EIGENVALUE_DOMINANCE_RATIO`,
`SSE_MDL_MAX_SOURCES`, `DEFAULT_AP_DISTANCE_M`, `MOBILE_OUI_TABLE_PATH`,
`PATH_LOSS_EXPONENT` — the last because it moves into `site.json` as a measured
quantity or is absent.

### Estimator set

| method | array geometry | snapshots | coherent sources | verdict |
|---|---|---|---|---|
| **IAA-APES** | arbitrary, known | one | yes | **default** |
| **Spectral MUSIC** | arbitrary, known | several | no | keep |
| **SpotFi** (Hankel-smoothed 2-D AoA×ToF MUSIC, 2015) | shift-invariant, for the smoothing | manufactures them from one packet | yes | keep, verify |
| root-MUSIC | uniform linear only | several | no | **drop** |
| ESPRIT / CA-ESPRIT | shift-invariant only | several | no | **drop** |

**Why root-MUSIC goes.** Rooting a polynomial in `z = exp(−jπ sin θ)` requires
element *k* to contribute `z^k` — uniform spacing by construction. It cannot be
corrected by supplying element positions, because there are none it can accept.
Assuming λ/2 ULA against realistic geometries gives 20–77° RMS and two of nine
test geometries fold. Spectral MUSIC gives the same estimate where a ULA does
hold and keeps working where it does not.

**`3_1_Spatial_Algorithms.py:148` is misnamed.** It builds its spectrum against
`exp(−jπ·n·sin θ)` over `nt` elements: ordinary 1-D spatial MUSIC in spectral
form. No second dimension, no residual stage. It is not an upgrade over
root-MUSIC in capability — same algorithm, grid search versus rooting. Keep the
spectral form, for the reason above, not for novelty.

**SpotFi, unverified in two places.** On the real captures, removing a linear
across-subcarrier phase ramp reduced residual phase spread by only 3–25 %, so
the delay dimension may buy less here than in the CSI setting it was designed
for. And its Hankel smoothing slides a window across antennas assuming they are
interchangeable — the same shift-invariance that disqualifies ESPRIT. It earns a
place in the set but not the default slot.

---

## 4. Work, in order

### 4.1 Fix the capture filters — blocking

In `config.env`, match on the Action Category byte instead of frame control:

```
wlan[24:2] == 0x1500 or wlan[24:2] == 0x1e00 or wlan[0] == 0x80
```

The last clause is the **beacon addition decided in this session** (section 5.5).
Byte semantics verified: `wlan[0] == 0x80` selected exactly the 810 beacons out
of 191,092 frames in the bundled 11ax trace, no false positives **[m]**. The
three-clause BPF has not been compiled here — `tcpdump` is absent from the
analysis environment — so check it once on the capture host.

In `0_replay_pcap.sh`, add `moved_to` to the inotify events.

**Why first.** `config.env` currently selects Action frames with
`wlan[0] & 0xfc == 0xd0`. Every frame in the live capture has a frame-control
byte of `0xe0` — subtype 14, Action No Ack. `0xe0 & 0xfc` is `0xe0`, so the test
fails on all 315. Stage 1 as configured records nothing, and every later stage is
untestable. The replay path has never triggered, so this is also the only way to
exercise the pipeline without hardware.

### 4.2 Capture once with the widened filter

15–30 minutes on the target channel, beacons included. Settles three things at
once: whether the network produces 11ax feedback at all (the VHT-only filter has
been hiding it); whether any of it is MU, which closes the last open question in
the 11ax MU path; and how many distinct APs are sounding on the captured channel,
and with which clients — the quantity the identifiability rows in 5.6 and 5.6a
are indexed by.

### 4.3 Build the observable log, drop pyshark from the hot path

`observe.py` with `capture_reader` and `Wi-BFI/main.py` as libraries. Records
every field listed in Stage 2, timestamped, append-only.

### 4.4 Demolition, against a regression bench

Write the bench first — synthetic ground-truth angle sweep across the nine array
geometries, plus a replay of the real captures — then delete the Removed list.
The sweep already caught two sign errors and a mirrored covariance; deleting
against a bench makes each removal falsifiable rather than a matter of taste.

### 4.5 Estimators and dispatcher

`aoa.py` behind one geometry-parameterised interface; `dispatch.py` selecting on
decoded metadata. Correct the CA-ESPRIT sign and rebuild the smoothed variant
against the literature's Hankel structure rather than the shipped version.

### 4.6 Solve, with the tier gate

Angular product always; bearing line when `site.json` is present. Each record
names its tier and the measurements used. There is no code path that can invent
a number, so a missing measurement shows up as a downgraded output rather than a
confident wrong one.

### 4.7 Calibration tools — optional

`calibrate.py`: closed-loop walk, and the anchor fit using the laptop's own
beamforming reports captured while associated on its second NIC.

### 4.8 View, then ground truth

`view.py` first, then marked positions at three traffic levels, reporting
accuracy against effective BFI count rather than wall time.

---

## 5. Session record

Two different kinds of entry, marked so they are not confused:

- **DECIDED** — the user directed it, or approved it explicitly. Settled.
- **FINDING** — a measurement produced in this session and reported. Not put to
  the user, not confirmed by the user. Treat as evidence to weigh, not as a
  ruling, and do not cite it as something the user agreed.


### 5.1 Per-stream SNR — landed in `c1d8d30`

*Status: landed in answer to the user's question about whether `packet_snr`
would be available. Not explicitly directed, not objected to.*

Upstream slices `packet_snr` and never reads it; the fork inherited the dead
assignment. Now decoded and stored. `dB = 22 + 0.25 × int8`, checked against
tshark on 1,800 frames across VHT SU, VHT MU and HE at all four widths, 0
mismatches **[m]**. Kept where the per-chain list was initially dropped because
it is a *different observable*, not more detail on RSSI — it is the only figure
in the frame describing the AP→client path.

### 5.2 Monitor radiotap RSSI — DECIDED, kept (was never removed)

*User: "lets keep the radiotap rssi for later use in bvlos".*

Element 3, 100 % coverage on the live capture, 0 unmeasured across all buckets **[m]**.

### 5.3 Per-chain radiotap list — DECIDED to store; exclusion from geometry not ruled on

*User: "land the storage change only" — that is the storage decision. The
recommendation to keep it out of the geometry was argued and questioned by the
user, and the user did not state a ruling either way.*

Landed in `2b61ec3`. The investigation and its verdicts:

**What it is.** `[combined, chain0, chain1]` — verified against Wireshark's
dissection: the first value carries no antenna index, the rest are indexed **[m]**.

**The one real finding — `value[0]` is driver-dependent** **[m]**:

| capture | `value[0]` = sum of chains | = max of chains |
|---|---|---|
| live, AWUS036AXM (mt7921au), 315 | **100.0 %** | 67.9 % |
| bundled 11ac_SU_3x1_40, 631 | 18.5 % | **100.0 %** |
| bundled 11ac_MU_3x1_80, 33,366 | 0.1 % | **100.0 %** |

`sum − max` is not a constant a calibration absorbs: 0.21–3.01 dB on the live
capture, per-transmitter medians 0.97–2.12 dB, IQR 0.55–1.57 dB, moving with the
chain balance. Stored alone, the combined figure silently changes meaning by more
than a dB when the adapter changes — the scale the ranging leg works at.

**Monitor-side bearing from the chain differential — REJECTED** **[m]**:

- within-transmitter IQR of Δ = **4.00 dB**; between-transmitter spread of
  medians = **1.92 dB**. Signal smaller than noise.
- Not a noise-floor artifact: correlation(median RSSI, median Δ) = +0.27; chain 1
  median −89 dBm against a −100 dBm floor, only 1 % within 2 dB of it.
- Physics: two identical vertical dipoles at λ/2 (3 cm at 5 GHz) differ by
  **0.026 dB at 10 m**, 0.087 dB at 3 m — two orders below the 1 dB
  quantisation. Amplitude-comparison DF requires deliberately *different*
  patterns (crossed loops, squinted beams); for identical co-located elements
  bearing lives in phase, and radiotap carries no per-chain phase.

**Association fingerprinting from Δ — REJECTED** (AUC, 0.5 = coin flip) **[m]**:

| window | n | RSSI only | Δ only | RSSI + Δ |
|---|---|---|---|---|
| 5 reports | 60 | 0.801 | 0.539 | 0.754 |
| 10 reports | 29 | **0.801** | 0.588 | 0.805 |
| 20 reports | 13 | 0.706 | 0.622 | 0.832 |

Near chance alone; hurts at W=5, neutral at W=10. The W=20 cell rests on 13
windows and RSSI-only drops there too, marking it small-sample noise.

**Retained uses:** detecting which convention the driver reports, and spotting a
dead or badly imbalanced chain. Nothing geometric.

**If monitor-side bearing is wanted later it is a hardware change, not a code
change** — two antennas with deliberately different patterns (tilted or
orthogonal dipoles) give a calibratable `f(x)`; or a card with CSI extraction
gives per-chain phase.

### 5.4 RSSI averaging has a floor — FINDING, not reviewed

sd of window means, live capture **[m]**:

| transmitter | n | sd | lag-1 r | W=2 | W=4 | W=8 | W=16 |
|---|---|---|---|---|---|---|---|
| 6e:09:60:37:f3:63 | 176 | 1.37 | +0.580 | 1.20 | 1.08 | 0.97 | **0.90** |
| f6:b1:4f:a6:7b:7c | 41 | 1.30 | +0.553 | 1.17 | 1.03 | 0.89 | 0.41 |
| 46:86:23:85:4b:fe | 41 | 1.64 | +0.555 | 1.44 | 1.24 | 1.11 | **1.06** |

White noise would take 1.37 → 0.34 by W=16. It reaches 0.90. Lag-1 r ≈ +0.55
says this is slow shadowing and environment drift, not white noise. **Floor
around 0.9–1.1 dB that averaging does not break** — roughly 4 m of irreducible
scatter at the measured knee, before the shadowing bias that produces the 17 m
blind figure. At 0.05–0.6 BFI/s, W=16 already costs 30 s to 5 minutes.

**Consequence:** "collect more packets" is not the lever for ranging accuracy.

### 5.5 Beacons — DECIDED, in the plan, not yet implemented

*User: "Include the inclusion of beacons in the plan".*

The APs never transmit beamforming reports. Live capture **[m]**:

```
TX addresses: 8 clients, 315 reports
RX addresses: 78:0c:f0:7a:5e:6e (307), 78:72:5d:90:2d:ae (8)
Addresses appearing as both TX and RX: none
Frames transmitted by either AP, whole file: 2   (byte coincidences)
Beacons: 0
```

So there is currently **no monitor-side RSSI for the AP at all**. Beacons supply
it, at ~9.69/s against 0.05–0.6 BFI/s — the highest-rate observable available by
two orders of magnitude, and the only one referenced to the monitor's position
that describes the AP. Added to the Stage 1 filter in section 4.1. Beacons also
carry SSID, BSSID and advertised transmit power, already consumed by `observe.py`.

### 5.6 Identifiability of the static single-AP model — FINDING, not reviewed

Jacobian rank of the observation model (monitor at origin, RSSI to clients and
AP, AoD at the AP). Deficit = unknowns − rank; above 0 means a continuum of
solutions fits the data *exactly*, at any noise level **[m]**.

| configuration | unknowns | rank | deficit |
|---|---|---|---|
| 3 clients, per-device EIRP | 14 | 7 | 7 |
| 8 clients, per-device EIRP | 29 | 17 | 12 |
| 20 clients, per-device EIRP | 65 | 41 | **24** |
| 3 clients, one shared EIRP | 12 | 7 | 5 |
| 20 clients, one shared EIRP | 46 | 41 | **5** |

A property of the model worth recording: the deficit does not fall as clients
are added, because each client contributes 2 equations and 2 unknowns and is
exactly neutral. With per-device transmit powers it rises. Stated as a property
of the model, not as a correction to anyone's expectation.

Null-space projection names the unresolved directions **[m]**:

| fraction unobservable | direction |
|---|---|
| **1.000** | global rotation about the monitor |
| **1.000** | global scale, absorbed by the EIRP constant |
| 0.898 | path-loss exponent traded against EIRP |

Rotation and scale are **exact symmetries of the whole observation set**, not
estimation weaknesses — no quantity of extra bearings or ranges from the same
setup can break them. Rotation because nothing anywhere defines a reference
direction (the monitor cannot supply one, per 5.3). Scale because
`RSSI = C − 10n·log₁₀(d)` means scaling every distance is absorbed by `C`.

**What does reduce the deficit** **[m]**:

| setup | deficit |
|---|---|
| 1 AP | 5 |
| 2 APs, no clients in common | 8 |
| 2 APs, 3 of 9 clients seen by both | 5 |
| 2 APs, **every** client seen by both | **2** |
| 1 AP, monitor observing from 2 places | **2** |
| 3 APs (no gain over 2) | 2 |
| 1 AP, with absolute ToF | 1 |

Two routes reach 2, and 2 is exactly rotation + scale: **the map is fully
determined in shape, unknown in orientation and size.** That is a real
deliverable — relative geometry, proximity, which side of the AP a device sits
on. It is not absolute coordinates.

**A relative map up to rotation and scale was the intended product from the
outset.** This measurement does not change the target; it names which two gauges
are free and shows they are exact symmetries rather than accuracy limits, so no
estimator effort can recover them and none should be spent trying.

### 5.6a Correction to 5.6: what the two-AP row actually requires

The model above gave a client a bearing from each AP that "sees" it. **That is
the wrong condition.** A compressed beamforming report is sent only to the AP
that sounded the client, which is the AP it is associated with. Visibility to
other APs — which is normal, and was never in question — produces no beamforming
report to those APs and therefore no second AoD. A client associates with one AP
at a time, so on a single captured channel the usual case is one AoD per client,
from its own AP.

The two-AP row therefore describes a case requiring the *same* client to be
sounded by *two* APs on the captured channel. Within one session that arises only
on a roam, and is useful only if the client is stationary across it. It is a
narrow case, not a property of the environment and not something a capture choice
selects for. **The monitor-from-two-places row is the reachable route to deficit
2**, and that is the calibration walk, which is optional by design.

The live capture's split — 307 reports to one AP, 8 to another — reflects which
APs were sounding on the captured channel. It says nothing about which APs the
clients could see.

### 5.7 Absolute ToF is structurally absent from BFI — FINDING, not reviewed

The rank test shows ToF is exactly the missing ingredient (deficit 5 → 1). It is
not obtainable. A propagation delay multiplies every entry of `H(f)` by one
scalar per subcarrier; `V` holds right singular vectors, which a scalar cannot
change, and the compression additionally fixes a phase gauge per subcarrier.
Verified on the reconstructed matrices **[m]**:

```
max |imag(last row of V)| = 0.00e+00      min real(last row) = +0.0292
```

Exactly real and non-negative on every subcarrier of every report. Literature
describes the same: BFI is the SVD of CSI "discarding amplitude and absolute
phase". What survives in BFI is AoD and relative multipath delay structure.

Status: measured and reported in this session; **the user has not reviewed or
ruled on it.** It bears on whether any absolute-range leg can rest on BFI, which
remains the user's call.

### 5.8 Corrections to statements made in this session

Recorded so they are not carried forward as established.

**"Monitor RSSI pins the transform down."** Too strong. It
fixes translation only — trivially, by placing the monitor at the origin — and
leaves rotation and scale exactly unobservable (5.6).

---

## 6. Measured ledger

Carried from earlier sessions unless marked **[m]** (this session).

**Cadence and coherence**
- Per-client BFI rate, live capture: 0.05–0.6 reports/s. Beacons 9.69/s.
  MU-MIMO bursts 21.8–86.1/s.
- Coherent span of the feedback: seconds. Beyond ~15 s subspaces are unrelated
  (41–56° apart). Per-measurement angular noise floor 6.6–22°.

**Ranging**
- Blind self-calibration: 17.2 m median, 215 m p90.
- Closed-loop calibration walk, 8 stops: AP to 1.13 m, `n` to ±0.19. Straight-line
  walk is degenerate: 11.18 m.
- Noise knee: 0.3 dB ↔ 0.85 m; 1.0 dB ↔ 4.16 m. Measured per-packet noise
  1.30–1.64 dB **[m]**, with a floor of 0.90–1.06 dB after averaging **[m]**.
- Metric client range for an uncooperative client: 4–9 m, against 6.2 m for
  guessing the centre of a 24 m room.

**Angles**
- Bearing-only triangulation, AP 1° / monitor 10°: 1.83 m median.
- Non-ULA geometries assumed λ/2 ULA: 20–77° RMS; two of nine fold.
- Phone antennas 1.38–2.42 λ apart at 5 GHz — fundamentally ambiguous.
- Ground-truth estimator test: CA-ESPRIT sign-inverted; SpotFi mirrored and
  collapsing to 0.00° at ±15°; Res-2D-MUSIC and IAA-APES correct.

**Kinematic energy (removed)**
- Within-client IQR 0.375 against between-client 0.012 — 31× the wrong way.
- 0 % of consecutive reports bit-identical at real cadence.

**Tooling**
- pyshark 328 ms/chunk against 0.1 ms direct. End-to-end only 1.2–1.9× because
  `textwrap.wrap`/`hex2dec` dominated; `hex2dec` simplification took 10.47 s → 7.68 s.
- Shipped BVLoS pipeline produces **zero position fixes** on 815 s of real 11ac data.

---

## 7. Open decisions and unresolved issues

**Blocking the merge**
1. **Wi-BFI master was reverted** to read Nc/Nr from dissector attributes that do
   not resolve on the current tshark, so it falls back to the CLI config on every
   packet — putting all 315 live frames in one bucket when 3 are 4×1 and 1 is at
   20 MHz. Merging `claude/sweet-cerf-p09l7d` conflicts in `main.py`, mechanically:
   this branch has no reference to the dissector packet object at all, so
   resolving in its favour is a clean replacement, verified 315/315. **Not yet done.**

**Needs a capture to settle**
2. **11ax MU subcarrier set.** The MU path is enabled and the angle tables are
   shared with VHT, but how many subcarriers an HE MU report covers is unverified:
   correct if the sounding is full-bandwidth at Ng=4, wrong if Ng=16 or scoped to
   a narrower resource unit. No 11ax MU capture exists. The short-payload guard
   makes this safe rather than silent — it reports the shortfall in bits, and that
   number pins the real value.
3. **Whether the target network produces HE feedback at all** — hidden until now
   by the VHT-only filter.
4. **How many APs are sounding on the captured channel, and whether any client
   is sounded by more than one of them within a session** (5.6a). An observation
   to make, not a requirement placed on the environment.
5. **Whether the λ/2 ULA assumption holds on a real AP.** Only ground truth
   settles it, which is why raw V-matrices stay in the log.
6. **Whether AoD from V maps to physical bearing on a real AP** — untested.

**Design questions not decided**
7. **How to handle unequal link reliability without a confidence score.** One
   link in the live capture has RSSI sd 3.5 dB against 1.3–1.6 for the others.
   A joint estimator would normally weight; that conflicts with the no-scores
   constraint. Candidate resolution: a robust loss (median-of-residuals) gives
   the same protection without a score. **Not decided.**
8. **How Stage 4 presents a rotation-and-scale-free map** in the plan view.
9. **Whether to keep the 3 ignored positional CLI args** in `main.py` or break
   compatibility once BVLoS calls it as a library.
10. **Whether multipath AoD helps identifiability.** Section 5.6 modelled one
    bearing per client per AP. BFF-MUSIC resolves several paths, which adds
    equations but also unknown scatterer positions; net effect untested. It
    cannot touch the rotation and scale gauges, which are exact symmetries, but
    could close some of the remaining 3.

**Settled in the approved plan** — discussed with the user and agreed:
- root-MUSIC, ESPRIT/CA-ESPRIT dropped from the estimator set (section 3).
- Kinematic energy, KPVT, OS-CFAR, occupancy counter, CEP heuristic,
  resampling, TDT, state store, hotspot classifier (section 3, Removed).

**Measured against and not pursued — my own exploration, never put to the user
as an option and never ruled on by the user.** Recorded so the measurement is
not repeated, not as a closed question:
- Monitor-side AoA from the chain amplitude ratio (5.3).
- The chain differential as an association feature (5.3).
- Averaging more packets to improve ranging accuracy (5.4).
- Any absolute-range leg resting on BFI (5.7).

---

## 8. Known limits

- **Metric client range is not available for an uncooperative client.** Per-client
  transmit power and noise-floor offset are never transmitted. The angular product
  is the honest output.
- **A 2-antenna hotspot is ambiguous at 5 GHz.** Output is a candidate set,
  pruned by continuity and the dB legs — not a bearing.
- **The coherent span is seconds.** Integrating longer averages different channel
  states together.
- **11ax MU is unproven end to end.** So is Stage 1 against real hardware beyond
  the single trace analysed.
- **Not implemented, deliberately:** coarser subcarrier groupings, HE feedback
  scoped to a narrow resource unit, VHT 160 MHz. Each is skipped with a
  diagnostic rather than guessed.
- **Rotation and scale are exact symmetries of the observation set** without
  `site.json` (5.6). The relative map was always the intended product; what is
  new is that these two gauges are provably unrecoverable rather than merely
  hard, so no estimator work should be aimed at them.

---

## 9. Literature

- **Wi-BFI** — extraction of 802.11 beamforming feedback from commercial devices.
  ACM WiNTECH 2023. <https://dl.acm.org/doi/10.1145/3615453.3616514>
- **BFF-based model-driven AoD estimation** — MUSIC on beamforming feedback
  alone, model-driven (no database), AoD error comparable to CSI-based MUSIC.
  IEEE, 2022. <https://ieeexplore.ieee.org/document/9787542/>
  *This is the precedent for the framework's core observable.*
- **LeakyBeam / "Lend Me Your Beam"** — privacy implications of plaintext
  beamforming feedback. NDSS 2025. Delivers **occupancy detection** through
  walls, **not** a position fix.
  <https://www.ndss-symposium.org/ndss-paper/lend-me-your-beam-privacy-implications-of-plaintext-beamforming-feedback-in-wifi/>
- **Enabling Ubiquitous WiFi Sensing with Beamforming Reports** — SIGCOMM 2023.
  Sensing, not localization. <https://dl.acm.org/doi/10.1145/3603269.3604817>
- **SpotFi** — Kotaru et al., SIGCOMM 2015. Joint AoA×ToF MUSIC with Hankel
  smoothing, on CSI.

**No peer-reviewed blind, passive, uncalibrated 2-D position fix from BFI was
found.** That is a statement about the search, not a proof none exists — but it
is consistent with the identifiability result in 5.6.
