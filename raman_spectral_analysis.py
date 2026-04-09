"""
raman_spectral_analysis.py
---------------------------
Spectral analysis routines for Raman data:
  - PCA
  - Gaussian peak fitting (Amide I band and general)
  - Weighted linear and quadratic regression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


# ── Principal component analysis ───────────────────────────────────────────

def run_pca(data, n_components=2, scale=True, plot=True,
            labels=None, plot_loadings=False, feature_names=None):
    """
    Run PCA on a spectral dataset.

    Parameters
    ----------
    data : array_like, shape (n_samples, n_features)
        Input data matrix.
    n_components : int
        Number of principal components to retain.
    scale : bool
        If True, standardise (mean=0, std=1) before PCA.
    plot : bool
        Plot the first two score components when True.
    labels : array_like, optional
        Sample labels for colour-coding the score plot.
    plot_loadings : bool
        If True, also plot the PC loadings.
    feature_names : array_like, optional
        Feature names (e.g. wavenumbers) for the loading plot x-axis.

    Returns
    -------
    pca_result : ndarray, shape (n_samples, n_components)
        Projected scores.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA object (contains explained variance, components, etc.).
    """
    data = np.asarray(data)

    if scale:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)

    if plot and n_components >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        if labels is not None:
            for label in np.unique(labels):
                idx = np.asarray(labels) == label
                ax.scatter(pca_result[idx, 0], pca_result[idx, 1],
                           label=str(label), alpha=0.7)
            ax.legend()
        else:
            ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
        ax.set_title("PCA – first two principal components")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    if plot_loadings:
        fig, ax = plt.subplots(figsize=(10, 5))
        x_axis = feature_names if feature_names is not None else np.arange(data.shape[1])
        for i in range(n_components):
            ax.plot(x_axis, pca.components_[i],
                    label=f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f} %)")
        ax.set_xlabel("Feature" if feature_names is None else "Wavenumber (cm⁻¹)")
        ax.set_ylabel("Loading")
        ax.set_title("PCA loadings")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return pca_result, pca


# ── Gaussian peak fitting ──────────────────────────────────────────────────

def gaussian_fitting_amide_band(x_data, y_data, fixed_centers,
                                initial_guess, bounds, plot=True):
    """
    Fit 2–4 Gaussians with fixed centers to the Raman Amide I band.

    Parameters
    ----------
    x_data : array_like
        Wavenumber values.
    y_data : array_like
        Normalised intensity values.
    fixed_centers : list of float
        Fixed center positions for each Gaussian component.
    initial_guess : list of float
        Initial guesses [amp1, width1, amp2, width2, …].
    bounds : tuple
        (lower_bounds, upper_bounds) passed to scipy.optimize.curve_fit.
    plot : bool
        Display the fit plot.

    Returns
    -------
    popt : ndarray
        Optimised parameters [amp1, width1, amp2, width2, …].
    r_squared : float
        Coefficient of determination of the combined fit.
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    def _model(x, *params):
        y = np.zeros_like(x, dtype=float)
        for i, cen in enumerate(fixed_centers):
            amp = params[i * 2]
            wid = params[i * 2 + 1]
            y += amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))
        return y

    popt, _ = curve_fit(_model, x_data, y_data, p0=initial_guess, bounds=bounds)
    y_fit = _model(x_data, *popt)
    r_squared = r2_score(y_data, y_fit)

    if plot:
        fig, ax = plt.subplots(figsize=(8.25 / 2.54, 6.25 / 2.54))
        ax.scatter(x_data, y_data, s=20, facecolors='none',
                   edgecolors='black', label='Raw data')
        ax.plot(x_data, y_fit, 'r--', lw=2, color='black', label='Combined fit')
        colors = ['purple', 'darkred', 'steelblue', 'green']
        for i, cen in enumerate(fixed_centers):
            amp = popt[i * 2]
            wid = popt[i * 2 + 1]
            single = amp * np.exp(-(x_data - cen) ** 2 / (2 * wid ** 2))
            ax.plot(x_data, single, '-', color=colors[i % len(colors)],
                    label=f'Gaussian {i+1}')
        ax.hlines(0, x_data.min(), x_data.max(), ls='--', color='black', alpha=0.6)
        ax.set_xlim(x_data.min(), x_data.max())
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Normalised Raman intensity')
        ax.legend()
        plt.tight_layout()
        plt.show()

    for i, cen in enumerate(fixed_centers):
        print(f"Gaussian {i+1}: amplitude = {popt[i*2]:.2f}, "
              f"center = {cen} cm⁻¹, width = {popt[i*2+1]:.2f} cm⁻¹")
    print(f"R² = {r_squared:.4f}")
    return popt, r_squared


# ── Weighted regression ────────────────────────────────────────────────────

def weighted_linear_fit(x, y, yerr=None):
    """
    Weighted linear regression (y = slope·x + intercept).

    Parameters
    ----------
    x, y : array_like
    yerr : array_like, optional
        Standard deviations in y used as weights (w = 1/σ²).

    Returns
    -------
    slope, intercept, slope_std_err, intercept_std_err, R_squared, y_fit
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = 1.0 / np.asarray(yerr, float) ** 2 if yerr is not None else np.ones_like(y)

    x_mean = np.average(x, weights=w)
    coeffs  = np.polyfit(x, y, deg=1, w=w)
    slope, intercept = coeffs
    y_fit = slope * x + intercept
    residuals = y - y_fit

    dof = len(x) - 2
    res_var = np.sum(w * residuals ** 2) / dof
    S_xx = np.sum(w * (x - x_mean) ** 2)

    slope_std_err     = np.sqrt(res_var / S_xx)
    intercept_std_err = np.sqrt(res_var * (1 / np.sum(w) + x_mean ** 2 / S_xx))

    SS_res = np.sum(residuals ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    R_squared = 1 - SS_res / SS_tot

    return slope, intercept, slope_std_err, intercept_std_err, R_squared, y_fit


def weighted_quadratic_fit(x, y, yerr=None):
    """
    Weighted quadratic regression (y = a·x² + b·x + c).

    Parameters
    ----------
    x, y : array_like
    yerr : array_like, optional
        Standard deviations in y used as weights (w = 1/σ²).

    Returns
    -------
    a, b, c, a_std_err, b_std_err, c_std_err, R_squared, y_fit
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = 1.0 / np.asarray(yerr, float) ** 2 if yerr is not None else np.ones_like(y, float)

    A = np.vstack([x ** 2, x, np.ones_like(x)]).T
    sqrt_w = np.sqrt(w)
    A_w = A * sqrt_w[:, None]
    y_w = y * sqrt_w

    ATA = A_w.T @ A_w
    try:
        ATA_inv = np.linalg.inv(ATA)
    except np.linalg.LinAlgError:
        nan8 = (np.nan,) * 7 + (np.full_like(y, np.nan),)
        return nan8

    params = ATA_inv @ (A_w.T @ y_w)
    a, b, c = params
    y_fit = a * x ** 2 + b * x + c
    residuals = y - y_fit

    dof = len(x) - 3
    if dof > 0:
        res_var = np.sum(w * residuals ** 2) / dof
        param_cov = res_var * ATA_inv
    else:
        res_var = np.nan
        param_cov = np.full_like(ATA_inv, np.nan)

    a_std_err, b_std_err, c_std_err = np.sqrt(np.diag(param_cov))

    SS_res = np.sum(residuals ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    R_squared = 1.0 - SS_res / SS_tot if SS_tot != 0 else 1.0

    return a, b, c, a_std_err, b_std_err, c_std_err, R_squared, y_fit
