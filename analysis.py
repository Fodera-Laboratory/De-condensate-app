"""
analysis.py — Top-level pipeline for the De-condensate app.

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
    wn_original:       np.ndarray,
    X:                 np.ndarray,
    positions:         dict,
    scan_mode:         str,
    pls_protein_info:  dict,    # may be None
    pls_salt_info:     dict,    # may be None
    ST_init:           np.ndarray,  # may be None (skips MCR)
    n_components:      int,
    settings:          dict,
    mcr_params:        dict = None,
    pls_settings:      dict = None,  # if None, falls back to settings
    pls_crowder_info:  dict = None,  # independent crowder PLS1, may be None
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

    # Protein PLS1 (single output, independent)
    pls_protein = pls_protein2 = pls_peg = None
    if pls_protein_info is not None:
        X_pls, _, _ = preprocess_matrix(X, wn_original, pls_settings)
        pls_protein = pls_protein_info["model"].predict(
            X_pls[:, pls_protein_info["valid_features"]]
        ).flatten()

    # Crowder PLS1 (independent model, optional)
    if pls_crowder_info is not None:
        X_pls_c, _, _ = preprocess_matrix(X, wn_original, pls_settings)
        pls_peg = pls_crowder_info["model"].predict(
            X_pls_c[:, pls_crowder_info["valid_features"]]
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
    }


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
