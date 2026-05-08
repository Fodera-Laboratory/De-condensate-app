"""
preprocessing.py — Spectral preprocessing and file I/O for the PEARL app.

Covers:
  - Loading linescan .txt files from raw bytes
  - Per-spectrum baseline correction, spike removal, normalisation
  - Wavenumber masking and batch matrix preprocessing
  - Scan-mode detection and cumulative distance computation
"""

import os
import sys
import tempfile

import numpy as np
from chemotools.baseline import AsLs
from chemotools.scatter import StandardNormalVariate
from chemotools.scale import MinMaxScaler, NormScaler
from chemotools.smooth import SavitzkyGolayFilter
from chemotools.derivative import SavitzkyGolay as SavitzkyGolayDeriv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raman_preprocessing import (
    rubberband_correction,
    rolling_ball_baseline,
    endpoint_baseline,
    linear_baseline,
    area_normalization,
    spike_removal_scp,
)
from raman_io import load_line_scan_from_txt


# ─────────────────────────────────────────────────────────────────────────────
# File loading
# ─────────────────────────────────────────────────────────────────────────────

def load_linescan_bytes(file_bytes: bytes, filename: str = "scan.txt"):
    """Load a linescan from raw bytes (e.g. a Streamlit UploadedFile)."""
    suffix = os.path.splitext(filename)[-1] or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        wn, X, positions = load_line_scan_from_txt(tmp_path)
    finally:
        os.unlink(tmp_path)
    return wn, X, positions


# ─────────────────────────────────────────────────────────────────────────────
# Per-spectrum preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_spectrum(I: np.ndarray, wn: np.ndarray, settings: dict) -> np.ndarray:
    """Apply spike removal → baseline → smoothing → normalisation to one spectrum."""
    baseline     = settings.get("baseline",     "rubberband")
    smooth       = settings.get("smooth",       "none")
    normalize    = settings.get("normalize",    "minmax")
    spike_remove = settings.get("spike_remove", True)

    if spike_remove:
        I = spike_removal_scp(I, threshold=settings.get("spike_threshold", 8.0))

    # ── Baseline correction ───────────────────────────────────────────────
    if baseline == "rubberband":
        I = I - rubberband_correction(wn, I)
    elif baseline == "rolling_ball":
        I = I - rolling_ball_baseline(I, ball_radius=settings.get("ball_radius", 50))
    elif baseline == "als":
        I = AsLs(lam=settings.get("als_lam", 1e5),
                 penalty=settings.get("als_p", 0.01)).fit_transform(I.reshape(1, -1))[0]
    elif baseline == "endpoint":
        I = I - endpoint_baseline(I)
    elif baseline == "linear":
        I = I - linear_baseline(I)
    elif baseline in ("arpls", "airpls", "iasls", "drpls",
                      "imodpoly", "modpoly", "poly"):
        import ramanspy.preprocessing.baseline as _rsb
        import ramanspy as _rs
        _lam_kw  = {"lam": settings.get("rs_lam", 1e5)}
        _poly_kw = {"poly_order": settings.get("rs_poly_order", 2)}
        _cls_map = {
            "arpls":    (_rsb.ARPLS,    _lam_kw),
            "airpls":   (_rsb.AIRPLS,   _lam_kw),
            "iasls":    (_rsb.IASLS,    _lam_kw),
            "drpls":    (_rsb.DRPLS,    _lam_kw),
            "imodpoly": (_rsb.IModPoly, _poly_kw),
            "modpoly":  (_rsb.ModPoly,  _poly_kw),
            "poly":     (_rsb.Poly,     _poly_kw),
        }
        _cls, _kw = _cls_map[baseline]
        I = _cls(**_kw).apply(_rs.Spectrum(I, wn)).spectral_data
    # baseline == "none": skip

    # ── Smoothing (optional) ──────────────────────────────────────────────
    if smooth == "savgol":
        I = SavitzkyGolayFilter(window_length=settings.get("sg_window", 11),
                                polyorder=settings.get("sg_poly", 3)).fit_transform(I.reshape(1, -1))[0]
    elif smooth == "gaussian":
        from scipy.ndimage import gaussian_filter
        I = gaussian_filter(I.astype(float), sigma=settings.get("gaussian_sigma", 1))
    elif smooth == "fft_lowpass":
        cutoff = settings.get("fft_cutoff", 0.1)
        n   = len(I)
        F   = np.fft.rfft(I)
        cut = max(1, int(cutoff * len(F)))
        F[cut:] = 0
        I = np.fft.irfft(F, n=n)
    # smooth == "none": skip

    # ── Normalisation ─────────────────────────────────────────────────────
    if normalize == "minmax":
        I = MinMaxScaler().fit_transform(I.reshape(1, -1))[0]
    elif normalize == "snv":
        I = StandardNormalVariate().fit_transform(I.reshape(1, -1))[0]
    elif normalize == "area":
        I = area_normalization(I, wn)
    elif normalize == "vector":
        I = NormScaler(l_norm=2).fit_transform(I.reshape(1, -1))[0]
    # normalize == "none": skip

    # ── Second derivative (optional, applied last) ────────────────────────
    if settings.get("second_deriv", False):
        sd_win  = int(settings.get("sd_window", 11))
        sd_poly = int(settings.get("sd_poly",    2))
        I = SavitzkyGolayDeriv(window_length=sd_win, polyorder=sd_poly,
                               deriv=2).fit_transform(I.reshape(1, -1))[0]

    return I


# ─────────────────────────────────────────────────────────────────────────────
# Wavenumber masking
# ─────────────────────────────────────────────────────────────────────────────

def build_mask(
    wn_original: np.ndarray,
    wn_min: float,
    wn_max: float,
    wn_cut_min: float = None,
    wn_cut_max: float = None,
) -> np.ndarray:
    """Boolean mask for wavenumber selection with optional gap cut."""
    mask = (wn_original >= wn_min) & (wn_original <= wn_max)
    if wn_cut_min is not None and wn_cut_max is not None:
        cut  = (wn_original >= wn_cut_min) & (wn_original <= wn_cut_max)
        mask = mask & ~cut
    return mask


def preprocess_matrix(
    matrix: np.ndarray,
    wn_original: np.ndarray,
    settings: dict,
    is_salt: bool = False,
) -> tuple:
    """
    Preprocess a 2-D spectral matrix (n_spectra × n_wavenumbers).

    Returns
    -------
    X_proc : ndarray  (n_spectra × n_wavenumbers_trimmed)
    wn     : ndarray  trimmed wavenumber axis
    mask   : bool ndarray applied to wn_original
    """
    if is_salt:
        wn_min = settings.get("salt_wn_min", 300)
        wn_max = settings.get("salt_wn_max", 1050)
        mask   = build_mask(wn_original, wn_min, wn_max)
        eff_settings = dict(
            settings,
            baseline  = settings.get("salt_baseline",  "none"),
            smooth    = settings.get("salt_smooth",     "none"),
            normalize = settings.get("salt_normalize",  "none"),
        )
        wn = wn_original[mask]
        rows = [_preprocess_spectrum(matrix[i][mask].copy(), wn, eff_settings)
                for i in range(matrix.shape[0])]
        return np.vstack(rows), wn, mask

    # Non-salt path — hybrid order:
    # 1. Range mask (wn_min/wn_max)
    # 2. Spike removal → baseline → smoothing  (on the full kept range)
    # 3. Gap exclusion  (remove silent region before normalisation)
    # 4. Normalisation → second derivative

    wn_min = settings.get("wn_min", 700)
    wn_max = settings.get("wn_max", 3900)
    range_mask = build_mask(wn_original, wn_min, wn_max)
    wn_full = wn_original[range_mask]

    # Steps 2: spike removal, baseline, smoothing — skip normalise and 2nd-deriv for now
    pre_settings = dict(settings, normalize="none", second_deriv=False)
    rows = [_preprocess_spectrum(matrix[i][range_mask].copy(), wn_full, pre_settings)
            for i in range(matrix.shape[0])]
    X_proc = np.vstack(rows)

    # Step 3: gap exclusion
    use_cut = settings.get("use_cut", False)
    cut_min = settings.get("wn_cut_min") if use_cut else None
    cut_max = settings.get("wn_cut_max") if use_cut else None
    if cut_min is not None and cut_max is not None:
        gap_keep   = (wn_full < cut_min) | (wn_full > cut_max)
        X_proc     = X_proc[:, gap_keep]
        wn         = wn_full[gap_keep]
        final_mask = range_mask.copy()
        final_mask[np.where(range_mask)[0][~gap_keep]] = False
    else:
        wn         = wn_full
        final_mask = range_mask

    # Step 4: normalisation and optional second derivative
    post_settings = dict(settings, baseline="none", smooth="none", spike_remove=False)
    rows = [_preprocess_spectrum(X_proc[i].copy(), wn, post_settings)
            for i in range(X_proc.shape[0])]
    return np.vstack(rows), wn, final_mask


# ─────────────────────────────────────────────────────────────────────────────
# Scan geometry
# ─────────────────────────────────────────────────────────────────────────────

def detect_scan_mode(positions: dict) -> str:
    """Auto-detect 'z' or 'xy' by comparing the range of z to the xy diagonal."""
    x = np.array(positions.get("x", [0]))
    y = np.array(positions.get("y", [0]))
    z = np.array(positions.get("z", [0]))
    z_range  = float(np.ptp(z))
    xy_range = float(np.sqrt(np.ptp(x) ** 2 + np.ptp(y) ** 2))
    return "z" if z_range > xy_range else "xy"


def compute_cumulative_distance(positions: dict, scan_mode: str) -> np.ndarray:
    """Return the position axis (µm) from a positions dict."""
    if scan_mode == "z":
        return np.array(positions["z"])
    x    = np.array(positions["x"])
    y    = np.array(positions["y"])
    dists = np.insert(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2), 0, 0.0)
    cum   = np.cumsum(dists)
    return cum - cum[-1] / 2.0
