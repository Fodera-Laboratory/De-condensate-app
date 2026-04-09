"""
raman_io.py
-----------
File I/O and utility functions for Raman data:
  - Reading single spectrum files (WITec / DataOrigin format)
  - Loading line-scan .txt files
  - Spectral integration (area under the curve)
  - Axis formatting helper
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── File reading ───────────────────────────────────────────────────────────

def read_raw_single_file(location, file_name):
    """
    Read wavenumber, intensity, and XYZ position from a WITec/DataOrigin
    single-spectrum text file.

    Parameters
    ----------
    location : str
        Directory containing the file.
    file_name : str
        File name.

    Returns
    -------
    wavenumbers : list of float
    intensities : list of float
    position : dict with keys 'x', 'y', 'z' (float or None)
    """
    file_path = os.path.join(location, file_name)
    wavenumbers = []
    intensities = []
    position = {"x": None, "y": None, "z": None}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data_section = False
        for line in f:
            line = line.strip()
            if line.startswith("PositionX"):
                position["x"] = float(line.split("=")[-1].strip())
            elif line.startswith("PositionY"):
                position["y"] = float(line.split("=")[-1].strip())
            elif line.startswith("PositionZ"):
                position["z"] = float(line.split("=")[-1].strip())

            if line.startswith("[Data]"):
                data_section = True
                next(f)   # skip column headers
                next(f)
                continue

            if data_section and line:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        wavenumbers.append(float(parts[0]))
                        intensities.append(float(parts[1]))
                except ValueError:
                    pass

    return wavenumbers, intensities, position


def load_line_scan_from_txt(input_file):
    """
    Load line-scan data from a raw WITec .txt export.

    Parameters
    ----------
    input_file : str
        Path to the .txt file.

    Returns
    -------
    wavenumbers : ndarray, shape (n_wavenumbers,)
    spectra : ndarray, shape (n_points, n_wavenumbers)
    positions : dict with keys 'x', 'y', 'z' — each an ndarray of length n_points
    """
    header = {}
    skiprows = 0
    with open(input_file, "r", encoding="latin1") as f:
        for i, line in enumerate(f):
            if line.strip() == "[Data]":
                skiprows = i + 2   # skip "[Data]" line + column-header line
                break
            if "=" in line:
                key, val = line.split("=", 1)
                header[key.strip()] = val.strip()

    n_points = int(header["SizeX"])
    x0, y0 = float(header["ScanStartX"]), float(header["ScanStartY"])
    x1, y1 = float(header["ScanStopX"]),  float(header["ScanStopY"])
    z0 = float(header.get("ScanStartZ", 0))
    z1 = float(header.get("ScanStopZ",  0))

    xs = np.linspace(x0, x1, n_points)
    ys = np.linspace(y0, y1, n_points)
    zs = np.linspace(z0, z1, n_points)

    df = pd.read_csv(input_file, sep="\t", skiprows=skiprows, encoding="latin1")
    wavenumbers = df.iloc[:, 0].astype(float).values
    spectra = df.iloc[:, 1:].to_numpy().T   # (n_points, n_wavenumbers)

    return wavenumbers, spectra, {"x": xs, "y": ys, "z": zs}


# ── Signal processing utility ──────────────────────────────────────────────

def integrate_wn_range(wavenumbers, intensities, wn_min, wn_max):
    """
    Integrate the area under the spectrum between wn_min and wn_max
    using the trapezoidal rule.

    Parameters
    ----------
    wavenumbers, intensities : array_like
    wn_min, wn_max : float
        Integration bounds (inclusive).

    Returns
    -------
    float
        Area under the curve.
    """
    wavenumbers = np.asarray(wavenumbers)
    intensities = np.asarray(intensities)
    mask = (wavenumbers >= wn_min) & (wavenumbers <= wn_max)
    return np.trapz(intensities[mask], wavenumbers[mask])


# ── Plotting utility ───────────────────────────────────────────────────────

def fix_ax_probs(ax, x_label, y_label):
    """
    Apply consistent formatting to a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x_label, y_label : str

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    ax.set_ylabel(y_label, size=6)
    ax.set_xlabel(x_label, size=6)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='both', which='minor', labelsize=5)
    ax.grid(False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    return ax
