"""
preprocessing.py — Spectral preprocessing and file I/O for the De-condensate app.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raman_preprocessing import (
    snv_normalization,
    rubberband_correction,
    als_baseline_correction,
    rolling_ball_baseline,
    Min_max_normalisation,
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
    """Apply baseline → spike removal → normalisation to one spectrum."""
    baseline     = settings.get("baseline",     "rubberband")
    normalize    = settings.get("normalize",    "minmax")
    spike_remove = settings.get("spike_remove", True)

    if baseline == "rubberband":
        I = I - rubberband_correction(wn, I)
    elif baseline == "rolling_ball":
        I = I - rolling_ball_baseline(I, ball_radius=settings.get("ball_radius", 50))
    elif baseline == "als":
        I = I - als_baseline_correction(
            I,
            lam=settings.get("als_lam", 1e5),
            p=settings.get("als_p", 0.01),
        )
    # baseline == "none": skip

    if spike_remove:
        I = spike_removal_scp(I)

    if normalize == "minmax":
        I = Min_max_normalisation(I)
    elif normalize == "snv":
        I = snv_normalization(I)
    # normalize == "none": skip

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
            normalize=settings.get("salt_normalize", "none"),
        )
    else:
        wn_min  = settings.get("wn_min", 700)
        wn_max  = settings.get("wn_max", 3900)
        cut_min = settings.get("wn_cut_min") if settings.get("use_cut") else None
        cut_max = settings.get("wn_cut_max") if settings.get("use_cut") else None
        mask    = build_mask(wn_original, wn_min, wn_max, cut_min, cut_max)
        eff_settings = settings

    wn   = wn_original[mask]
    rows = []
    for i in range(matrix.shape[0]):
        I = matrix[i][mask].copy()
        I = _preprocess_spectrum(I, wn, eff_settings)
        rows.append(I)
    return np.vstack(rows), wn, mask


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
