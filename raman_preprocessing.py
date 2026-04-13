"""
raman_preprocessing.py
-----------------------
Spectral preprocessing routines for Raman data:
  - Spike / cosmic ray removal
  - Baseline correction  (linear, rubber-band, ALS, rolling-ball)
  - Normalisation (min-max, SNV)
  - Savitzky-Golay smoothing
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import interpolate
from scipy.ndimage import grey_opening


# ── Spike / cosmic-ray removal ─────────────────────────────────────────────

def spike_removal(y,
                  width_threshold,
                  prominence_threshold=20,
                  moving_average_window=10,
                  width_param_rel=0.8,
                  interp_type='linear'):
    """
    Detect and replace cosmic spikes in a spectrum with interpolated values.

    Algorithm first published by N. Coca-Lopez in Analytica Chimica Acta.
    https://doi.org/10.1016/j.aca.2024.342312

    Parameters
    ----------
    y : array_like
        Input spectrum intensity.
    width_threshold : float
        Peaks narrower than this are classified as spikes.
    prominence_threshold : float
        Minimum prominence for a peak to be considered a spike candidate.
    moving_average_window : int
        Half-width of the interpolation window around each spike.
    width_param_rel : float
        Relative height used when measuring peak extent.
    interp_type : str
        Interpolation kind passed to scipy.interpolate.interp1d
        ('linear', 'quadratic', 'cubic').

    Returns
    -------
    y_out : ndarray
        Spectrum with spikes replaced by interpolated values.
    """
    peaks, _ = find_peaks(y, prominence=prominence_threshold, distance=4, width=1)
    spikes = np.zeros(len(y))
    widths = peak_widths(y, peaks)[0]
    widths_ext_a = peak_widths(y, peaks, rel_height=width_param_rel)[2]
    widths_ext_b = peak_widths(y, peaks, rel_height=width_param_rel)[3]

    for _, width, ext_a, ext_b in zip(range(len(widths)), widths, widths_ext_a, widths_ext_b):
        if width < width_threshold:
            spikes[int(ext_a) - 1: int(ext_b) + 2] = 1

    y_out = y.copy()
    for i, spike in enumerate(spikes):
        if spike != 0:
            window_start = max(0, i - moving_average_window)
            window_end   = min(len(y), i + moving_average_window + 1)
            window = np.arange(window_start, window_end)
            window = window[(window >= 0) & (window < len(y))]
            window_no_spikes = window[spikes[window] == 0]
            if len(window_no_spikes) >= 2:
                f = interpolate.interp1d(
                    window_no_spikes, y[window_no_spikes],
                    kind=interp_type, bounds_error=False, fill_value="extrapolate",
                )
                y_out[i] = f(i)
    return y_out


def spike_removal_scp(data, kernel_size=3, threshold=8):
    """
    Remove cosmic spikes using the Whitaker-Hayes algorithm as implemented in RamanSPy.

    Detects spikes via modified Z-score on the first difference of the spectrum,
    then iteratively replaces each spike with the mean of its non-spike neighbours.

    Reference: Whitaker & Hayes (2018). A simple algorithm for despiking Raman
    spectra. Chemometrics and Intelligent Laboratory Systems, 179, 82–84.
    https://doi.org/10.1016/j.chemolab.2018.06.009

    Parameters
    ----------
    data : array_like, shape (n_wavenumbers,)
        Single spectrum.
    kernel_size : int
        Half-width of the neighbourhood window for replacement (default: 3).
    threshold : float
        Modified Z-score threshold for spike detection (default: 8).

    Returns
    -------
    ndarray, shape (n_wavenumbers,)
        Despiked spectrum.
    """
    y = np.asarray(data, dtype=float).copy()
    n = len(y)

    # Modified Z-score on first difference
    d   = np.diff(y)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    if mad == 0:
        return y
    z      = np.abs(0.6745 * (d - med) / mad)
    # first diff has n-1 points; pad to n by repeating last value
    z      = np.append(z, z[-1])
    spikes = z > threshold

    # Iteratively replace spikes with mean of non-spike neighbours
    while spikes.any():
        changed = False
        for i in np.where(spikes)[0]:
            neighbours = np.arange(max(0, i - kernel_size),
                                   min(n, i + kernel_size + 1))
            good = neighbours[~spikes[neighbours]]
            if len(good) == 0:
                continue
            y[i]      = np.mean(y[good])
            spikes[i] = False
            changed   = True
        if not changed:
            break

    return y


# ── Baseline correction ────────────────────────────────────────────────────

def linear_background_subtraction(x, y, fit_region=None):
    """
    Subtract a linear baseline fitted to the spectral edges.

    Parameters
    ----------
    x : array_like
        Wavenumber axis.
    y : array_like
        Intensity values.
    fit_region : list of (start, stop) tuples, optional
        Index slices used for fitting (default: first and last 5 % of points).

    Returns
    -------
    y_corrected : ndarray
    baseline : ndarray
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if fit_region is None:
        n = len(x)
        fit_region = [(0, int(0.05 * n)), (-int(0.05 * n), None)]

    x_fit = np.concatenate([x[s:e] for s, e in fit_region])
    y_fit = np.concatenate([y[s:e] for s, e in fit_region])
    coeffs   = np.polyfit(x_fit, y_fit, deg=1)
    baseline = np.polyval(coeffs, x)
    return y - baseline, baseline


def _lower_convex_hull(y):
    """Return the lower convex hull of a 1-D spectrum as a baseline array."""
    n = len(y)
    pts = [(i, float(y[i])) for i in range(n)]

    # Andrew's monotone chain — lower hull only
    lower = []
    for p in pts:
        while len(lower) >= 2:
            o, a = lower[-2], lower[-1]
            cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)

    hull_x = np.array([p[0] for p in lower])
    hull_y = np.array([p[1] for p in lower])
    return np.interp(np.arange(n), hull_x, hull_y)


def rubberband_correction(x, y):
    """
    Estimate baseline using the rubber-band (lower convex hull) method.

    Parameters
    ----------
    x : array_like
        Wavenumber axis (unused in computation, kept for API compatibility).
    y : array_like
        Intensity values.

    Returns
    -------
    baseline : ndarray
    """
    y = np.asarray(y, dtype=float)
    return _lower_convex_hull(y)


def als_baseline_correction(y, lam=1e5, p=0.01, niter=50):
    """
    Asymmetric least-squares (ALS) baseline correction.

    Original algorithm: Eilers & Boelens (2005). Leiden University Medical Centre Report.

    Parameters
    ----------
    y : array_like
        Intensity values.
    lam : float
        Smoothness parameter (1e4–1e7; higher → smoother baseline).
    p : float
        Asymmetry parameter (0.001–0.1; lower → baseline hugs minima).
    niter : int
        Maximum number of iterations.

    Returns
    -------
    baseline : ndarray
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    H = lam * D.T.dot(D)
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = sparse.diags(w, 0)
        z = spsolve(W + H, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def rolling_ball_baseline(y, ball_radius=50):
    """
    Rolling-ball baseline correction via morphological opening.

    Parameters
    ----------
    y : array_like
        Intensity values.
    ball_radius : int
        Radius of the structuring element (larger → smoother).

    Returns
    -------
    baseline : ndarray
    """
    y = np.asarray(y)
    struct_size = int(2 * ball_radius + 1)
    return grey_opening(y, size=struct_size)


# ── Normalisation ──────────────────────────────────────────────────────────

def Min_max_normalisation(I_data):
    """Scale a spectrum to [0, 1] using min-max normalisation."""
    I_data = np.asarray(I_data)
    return (I_data - I_data.min()) / (I_data.max() - I_data.min())


def snv_normalization(data):
    """
    Standard Normal Variate (SNV) normalisation.

    Parameters
    ----------
    data : array_like
        1-D (single spectrum) or 2-D array (samples × features).

    Returns
    -------
    ndarray, same shape as input.
    """
    data = np.asarray(data)
    if data.ndim == 1:
        return (data - data.mean()) / data.std()
    elif data.ndim == 2:
        return (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
    else:
        raise ValueError("Input must be a 1-D or 2-D array.")


# ── Smoothing ──────────────────────────────────────────────────────────────

def Savgol_filter(y, window_length, polyorder):
    """
    Savitzky-Golay smoothing.

    Parameters
    ----------
    y : array_like
        Intensity values.
    window_length : int
        Length of the filter window (must be odd).
    polyorder : int
        Polynomial order.

    Returns
    -------
    ndarray
        Smoothed spectrum.
    """
    y = np.asarray(y, dtype=float)
    return savgol_filter(y, window_length, polyorder)
