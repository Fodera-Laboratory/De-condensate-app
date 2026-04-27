"""
analysis.py — Top-level pipeline for the PEARL app.

Re-exports all public functions from the sub-modules so that app.py can
continue to use `import analysis as an` without changes.

Sub-modules
-----------
preprocessing  : file loading, spectral preprocessing, scan geometry
decomposition  : PLS calibration (single & dual) and MCR-ALS
"""

import io

import numpy as np
import pandas as pd

from preprocessing import (          # noqa: F401 (re-exported)
    load_linescan_bytes,
    build_mask,
    preprocess_matrix,
    detect_scan_mode,
    compute_cumulative_distance,
)

from decomposition import (          # noqa: F401 (re-exported)
    build_pls_model,
    build_dual_pls_model,
    build_triple_pls_model,
    run_mcr,
)


# ─────────────────────────────────────────────────────────────────────────────
# Full linescan pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_linescan(
    wn_original:      np.ndarray,
    X:                np.ndarray,
    positions:        dict,
    scan_mode:        str,
    pls_protein_info: dict,    # may be None
    pls_salt_info:    dict,    # may be None
    ST_init:          np.ndarray,  # may be None (skips MCR)
    n_components:     int,
    settings:         dict,
    mcr_params:       dict = None,
    pls_settings:     dict = None,  # if None, falls back to settings
) -> dict:
    """
    Full pipeline for one linescan: preprocess → MCR → PLS.

    Returns a results dict with all outputs.  Any of MCR, protein PLS, and
    salt PLS may be absent if the corresponding inputs are None.
    """
    if mcr_params is None:
        mcr_params = {}
    if pls_settings is None:
        pls_settings = settings

    distance = compute_cumulative_distance(positions, scan_mode)
    X_proc, wn_proc, _ = preprocess_matrix(X, wn_original, settings)

    # MCR (optional)
    C_mcr = ST_mcr = mcr_ratio = mcr_n_iter = None
    if ST_init is not None:
        C_mcr, ST_mcr, mcr_n_iter = run_mcr(
            X_proc, ST_init, n_components, **mcr_params
        )
        if C_mcr.shape[1] >= 2:
            mcr_ratio = np.divide(
                C_mcr[:, 0], C_mcr[:, 1],
                out=np.zeros_like(C_mcr[:, 0]),
                where=C_mcr[:, 1] != 0,
            )

    # Protein (and optionally molecular crowder) PLS (optional)
    # Preprocess with pls_settings to match the normalization used during training
    pls_protein = pls_protein2 = pls_peg = None
    X_pls = wn_pls = None
    if pls_protein_info is not None:
        X_pls, wn_pls, _ = preprocess_matrix(X, wn_original, pls_settings)
        if pls_protein_info.get("triple"):
            Y_triple     = pls_protein_info["model"].predict(
                X_pls[:, pls_protein_info["valid_features"]]
            )
            pls_protein  = Y_triple[:, 0].flatten()
            pls_protein2 = Y_triple[:, 1].flatten()
            pls_peg      = Y_triple[:, 2].flatten()
        elif pls_protein_info.get("dual"):
            Y_dual      = pls_protein_info["model"].predict(
                X_pls[:, pls_protein_info["valid_features"]]
            )
            pls_protein = Y_dual[:, 0].flatten()
            pls_peg     = Y_dual[:, 1].flatten()
        else:
            pls_protein = pls_protein_info["model"].predict(
                X_pls[:, pls_protein_info["valid_features"]]
            ).flatten()

    # Salt PLS (optional, separate model on fingerprint region)
    pls_salt = None
    if pls_salt_info is not None:
        X_salt, _, _ = preprocess_matrix(X, wn_original, pls_settings, is_salt=True)
        pls_salt = pls_salt_info["model"].predict(
            X_salt[:, pls_salt_info["valid_features"]]
        ).flatten()

    return {
        "distance":     distance,
        "positions":    positions,          # raw stage x/y/z per spectrum (µm)
        "X_proc":       X_proc,
        "wn_proc":      wn_proc,
        "C_mcr":        C_mcr,
        "ST_mcr":       ST_mcr,
        "mcr_ratio":    mcr_ratio,
        "mcr_n_iter":   mcr_n_iter,
        "pls_protein":  pls_protein,
        "pls_protein2": pls_protein2,
        "pls_peg":      pls_peg,
        "pls_salt":     pls_salt,
        "X_pls":        X_pls,
        "wn_pls":       wn_pls,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Classical Least Squares (CLS)
# ─────────────────────────────────────────────────────────────────────────────

def run_cls(X_proc, wn_proc, ST_ref_raw, wn_ref, settings):
    """
    Classical Least Squares unmixing via per-spectrum NNLS.

    Parameters
    ----------
    X_proc     : ndarray (n_spectra, n_wn)   already-preprocessed linescan matrix
    wn_proc    : ndarray (n_wn,)             wavenumber axis of X_proc
    ST_ref_raw : ndarray (n_comp, n_wn_ref)  raw reference spectra (unprocessed)
    wn_ref     : ndarray (n_wn_ref,)         wavenumber axis of ST_ref_raw
    settings   : dict                        preprocessing settings (same as linescan)

    Returns
    -------
    dict with keys
        C        : (n_spectra, n_comp)        NNLS concentrations
        ST       : (n_comp, n_wn)             preprocessed references on wn_proc axis
        R2       : (n_spectra,)               per-spectrum R² of reconstruction
        RMSE     : (n_spectra,)               per-spectrum RMSE
        residual : (n_spectra, n_wn)          measured − reconstructed
    """
    from scipy.optimize import nnls as _nnls
    from preprocessing import preprocess_matrix

    wn_proc = np.asarray(wn_proc)
    wn_ref  = np.asarray(wn_ref)

    ST_aligned = np.vstack([np.interp(wn_proc, wn_ref, row) for row in ST_ref_raw])
    ST_proc, _, _ = preprocess_matrix(ST_aligned, wn_proc, settings)

    n_spectra = X_proc.shape[0]
    n_comp    = ST_proc.shape[0]
    C = np.zeros((n_spectra, n_comp))
    for i in range(n_spectra):
        C[i], _ = _nnls(ST_proc.T, X_proc[i])

    X_rec    = C @ ST_proc
    residual = X_proc - X_rec
    ss_res   = (residual ** 2).sum(axis=1)
    ss_tot   = ((X_proc - X_proc.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    R2   = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    RMSE = np.sqrt((residual ** 2).mean(axis=1))

    return dict(C=C, ST=ST_proc, R2=R2, RMSE=RMSE, residual=residual)


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def results_to_excel_bytes(
    results:      dict,
    scan_mode:    str,
    protein_unit: str,
    salt_unit:    str,
) -> bytes:
    """Serialise a results dict to an Excel file returned as bytes."""
    dist_col = "Depth_um" if scan_mode == "z" else "Distance_um"
    data = {dist_col: results["distance"].astype(float)}

    if results["C_mcr"] is not None:
        n_comp = results["C_mcr"].shape[1]
        for k in range(n_comp):
            data[f"MCR_Comp{k + 1}"] = results["C_mcr"][:, k].astype(float)
        if results["mcr_ratio"] is not None:
            data["MCR_Ratio_Comp1_Comp2"] = results["mcr_ratio"].astype(float)

    if results.get("pls_protein") is not None:
        prot_col = f"PLS_Protein1_{protein_unit.replace('/', '_per_')}"
        data[prot_col] = results["pls_protein"].astype(float)

    if results.get("pls_protein2") is not None:
        data[f"PLS_Protein2_{protein_unit.replace('/', '_per_')}"] = results["pls_protein2"].astype(float)

    if results.get("pls_peg") is not None:
        data["PLS_Crowder_wt_percent"] = results["pls_peg"].astype(float)

    if results.get("pls_salt") is not None:
        data[f"PLS_Salt_{salt_unit}"] = results["pls_salt"].astype(float)

    df  = pd.DataFrame(data).replace([np.inf, -np.inf], np.nan)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()
