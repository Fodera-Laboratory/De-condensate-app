"""
decomposition.py — Spectral decomposition for the De-condensate app.

Covers:
  - PLS calibration (single-output and dual protein+molecular crowder)
    with automatic component selection via the 1-standard-error rule
  - MCR-ALS via pymcr
"""

import numpy as np
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg, ConstraintNorm
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


# ─────────────────────────────────────────────────────────────────────────────
# PLS — component selection
# ─────────────────────────────────────────────────────────────────────────────

def _find_optimal_components(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int = 20,
    cv_folds: int = 5,
):
    """
    Cross-validate PLSRegression over 1..max_components and return the optimal
    count using the 1-standard-error rule (simplest model within 1 SE of the
    minimum CV RMSE).

    Returns
    -------
    optimal_n   : int
    rmse_cv     : list[float]  mean CV RMSE per component count
    rmse_train  : list[float]  training RMSE per component count
    """
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rmse_cv, rmse_train = [], []
    fold_errors_all = []
    max_comp = min(max_components, X.shape[0] - 1, X.shape[1])

    for n_comp in range(1, max_comp + 1):
        pls    = PLSRegression(n_components=n_comp, scale=False)
        errors = []
        for train_idx, val_idx in kfold.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            feat_std    = np.std(X_tr, axis=0)
            valid       = feat_std > 1e-10
            if np.sum(valid) < n_comp:
                continue
            pls.fit(X_tr[:, valid], y[train_idx])
            errors.append(
                float(np.sqrt(mean_squared_error(
                    y[val_idx], pls.predict(X_val[:, valid])
                )))
            )
        fold_errors_all.append(errors)
        rmse_cv.append(float(np.mean(errors)) if errors else float("inf"))

        feat_std = np.std(X, axis=0)
        valid    = feat_std > 1e-10
        pls.fit(X[:, valid], y)
        rmse_train.append(
            float(np.sqrt(mean_squared_error(y, pls.predict(X[:, valid]))))
        )

    # 1-SE rule: prefer the simplest model within 1 SE of the best CV RMSE.
    best_idx   = int(np.argmin(rmse_cv))
    best_errs  = fold_errors_all[best_idx]
    se_best    = (float(np.std(best_errs, ddof=1)) / np.sqrt(len(best_errs))
                  if len(best_errs) > 1 else 0.0)
    threshold  = rmse_cv[best_idx] + se_best
    optimal_n  = next(
        (i + 1 for i, v in enumerate(rmse_cv) if v <= threshold),
        best_idx + 1,
    )
    return optimal_n, rmse_cv, rmse_train


# ─────────────────────────────────────────────────────────────────────────────
# PLS — model building
# ─────────────────────────────────────────────────────────────────────────────

def build_pls_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_components: int = 20,
    cv_folds: int = 5,
) -> dict:
    """
    Train a single-output PLS model with automatic component selection.

    Returns a dict with the fitted model, feature mask, metrics, and training
    predictions — ready to be augmented with wn/X_train_proc by the caller.
    """
    n_opt, rmse_cv, rmse_train = _find_optimal_components(
        X_train, y_train,
        max_components=max_components,
        cv_folds=cv_folds,
    )

    feat_std       = np.std(X_train, axis=0)
    valid_features = feat_std > 1e-10

    model  = PLSRegression(n_components=n_opt, scale=False)
    model.fit(X_train[:, valid_features], y_train)
    y_pred = model.predict(X_train[:, valid_features]).flatten()

    return {
        "model":          model,
        "valid_features": valid_features,
        "n_components":   n_opt,
        "rmse_train":     float(np.sqrt(mean_squared_error(y_train, y_pred))),
        "r2_train":       float(r2_score(y_train, y_pred)),
        "cv_rmse":        float(rmse_cv[n_opt - 1]),
        "rmse_cv_all":    rmse_cv,
        "rmse_train_all": rmse_train,
        "y_train":        y_train,
        "y_pred_train":   y_pred,
    }


def build_dual_pls_model(
    X_protein: np.ndarray, y_protein: np.ndarray,
    X_peg:     np.ndarray, y_peg:     np.ndarray,
    max_components: int = 20,
    cv_folds: int = 5,
) -> dict:
    """
    Train a multi-output PLS2 model predicting protein and molecular crowder simultaneously.

    The training matrix stacks protein standards (crowder = 0) with molecular crowder standards
    (protein = 0), following the approach in the original batch script.

    Returns a dict with 'dual': True so process_linescan dispatches correctly.
    """
    X_train         = np.vstack([X_protein, X_peg])
    y_train_protein = np.concatenate([y_protein,              np.zeros(len(y_peg))])
    y_train_peg     = np.concatenate([np.zeros(len(y_protein)), y_peg])
    Y_train         = np.column_stack([y_train_protein, y_train_peg])

    n_opt, rmse_cv, rmse_train = _find_optimal_components(
        X_train, Y_train,
        max_components=max_components,
        cv_folds=cv_folds,
    )

    feat_std       = np.std(X_train, axis=0)
    valid_features = feat_std > 1e-10

    model  = PLSRegression(n_components=n_opt, scale=False)
    model.fit(X_train[:, valid_features], Y_train)
    Y_pred = model.predict(X_train[:, valid_features])

    rmse_protein = float(np.sqrt(mean_squared_error(y_train_protein, Y_pred[:, 0])))
    r2_protein   = float(r2_score(y_train_protein, Y_pred[:, 0]))
    rmse_peg     = float(np.sqrt(mean_squared_error(y_train_peg,     Y_pred[:, 1])))
    r2_peg       = float(r2_score(y_train_peg,     Y_pred[:, 1]))

    # Per-output CV RMSE for error bands in results plots
    kfold             = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    prot_cv_errors, peg_cv_errors = [], []
    X_f = X_train[:, valid_features]
    for train_idx, val_idx in kfold.split(X_f):
        X_tr_f       = X_f[train_idx]
        valid_inner  = np.std(X_tr_f, axis=0) > 1e-10
        pls_tmp      = PLSRegression(n_components=n_opt, scale=False)
        pls_tmp.fit(X_tr_f[:, valid_inner], Y_train[train_idx])
        Y_val_pred   = pls_tmp.predict(X_f[val_idx][:, valid_inner])
        prot_cv_errors.append(float(np.sqrt(mean_squared_error(
            Y_train[val_idx, 0], Y_val_pred[:, 0]
        ))))
        peg_cv_errors.append(float(np.sqrt(mean_squared_error(
            Y_train[val_idx, 1], Y_val_pred[:, 1]
        ))))

    return {
        "dual":                  True,
        "model":                 model,
        "valid_features":        valid_features,
        "n_components":          n_opt,
        "rmse_protein_train":    rmse_protein,
        "r2_protein_train":      r2_protein,
        "rmse_peg_train":        rmse_peg,
        "r2_peg_train":          r2_peg,
        "cv_rmse":               float(rmse_cv[n_opt - 1]),
        "cv_rmse_protein":       float(np.mean(prot_cv_errors)),
        "cv_rmse_peg":           float(np.mean(peg_cv_errors)),
        "rmse_cv_all":           rmse_cv,
        "rmse_train_all":        rmse_train,
        "y_protein_train":       y_train_protein,
        "y_peg_train":           y_train_peg,
        "y_pred_protein_train":  Y_pred[:, 0],
        "y_pred_peg_train":      Y_pred[:, 1],
        "X_train_proc":          X_train,
        "wn":                    None,   # filled by app.py after training
    }


def build_triple_pls_model(
    X_p1: np.ndarray, y_p1: np.ndarray,
    X_p2: np.ndarray, y_p2: np.ndarray,
    X_peg: np.ndarray, y_peg: np.ndarray,
    max_components: int = 20,
    cv_folds: int = 5,
) -> dict:
    """
    Train a multi-output PLS2 model predicting two proteins and a molecular crowder
    simultaneously.  Training stacks all three standard sets with the other outputs
    zeroed out, identical to the dual approach.

    Returns a dict with 'triple': True and 'dual': True so downstream code that
    only checks 'dual' still dispatches correctly.
    """
    n1, n2, np_ = len(y_p1), len(y_p2), len(y_peg)
    X_train  = np.vstack([X_p1, X_p2, X_peg])
    y_t_p1   = np.concatenate([y_p1,           np.zeros(n2),  np.zeros(np_)])
    y_t_p2   = np.concatenate([np.zeros(n1),   y_p2,          np.zeros(np_)])
    y_t_peg  = np.concatenate([np.zeros(n1),   np.zeros(n2),  y_peg        ])
    Y_train  = np.column_stack([y_t_p1, y_t_p2, y_t_peg])

    n_opt, rmse_cv, rmse_train = _find_optimal_components(
        X_train, Y_train,
        max_components=max_components,
        cv_folds=cv_folds,
    )

    feat_std       = np.std(X_train, axis=0)
    valid_features = feat_std > 1e-10

    model  = PLSRegression(n_components=n_opt, scale=False)
    model.fit(X_train[:, valid_features], Y_train)
    Y_pred = model.predict(X_train[:, valid_features])

    rmse_p1  = float(np.sqrt(mean_squared_error(y_t_p1,  Y_pred[:, 0])))
    r2_p1    = float(r2_score(y_t_p1,  Y_pred[:, 0]))
    rmse_p2  = float(np.sqrt(mean_squared_error(y_t_p2,  Y_pred[:, 1])))
    r2_p2    = float(r2_score(y_t_p2,  Y_pred[:, 1]))
    rmse_peg = float(np.sqrt(mean_squared_error(y_t_peg, Y_pred[:, 2])))
    r2_peg   = float(r2_score(y_t_peg, Y_pred[:, 2]))

    # Per-output CV RMSE for error bands in results plots
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    p1_cv_errors, p2_cv_errors, peg_cv_errors = [], [], []
    X_f = X_train[:, valid_features]
    for train_idx, val_idx in kfold.split(X_f):
        X_tr_f      = X_f[train_idx]
        valid_inner = np.std(X_tr_f, axis=0) > 1e-10
        pls_tmp     = PLSRegression(n_components=n_opt, scale=False)
        pls_tmp.fit(X_tr_f[:, valid_inner], Y_train[train_idx])
        Y_vp = pls_tmp.predict(X_f[val_idx][:, valid_inner])
        p1_cv_errors.append(float(np.sqrt(mean_squared_error(Y_train[val_idx, 0], Y_vp[:, 0]))))
        p2_cv_errors.append(float(np.sqrt(mean_squared_error(Y_train[val_idx, 1], Y_vp[:, 1]))))
        peg_cv_errors.append(float(np.sqrt(mean_squared_error(Y_train[val_idx, 2], Y_vp[:, 2]))))

    return {
        "triple":               True,
        "dual":                 True,   # superset — keeps dual dispatch paths working
        "model":                model,
        "valid_features":       valid_features,
        "n_components":         n_opt,
        "rmse_p1_train":        rmse_p1,
        "r2_p1_train":          r2_p1,
        "rmse_p2_train":        rmse_p2,
        "r2_p2_train":          r2_p2,
        "rmse_peg_train":       rmse_peg,
        "r2_peg_train":         r2_peg,
        "cv_rmse":              float(rmse_cv[n_opt - 1]),
        "cv_rmse_p1":           float(np.mean(p1_cv_errors)),
        "cv_rmse_p2":           float(np.mean(p2_cv_errors)),
        "cv_rmse_peg":          float(np.mean(peg_cv_errors)),
        # keep old key names so non-triple dual code doesn't KeyError
        "cv_rmse_protein":      float(np.mean(p1_cv_errors)),
        "rmse_cv_all":          rmse_cv,
        "rmse_train_all":       rmse_train,
        "y_p1_train":           y_t_p1,
        "y_p2_train":           y_t_p2,
        "y_peg_train":          y_t_peg,
        "y_pred_p1_train":      Y_pred[:, 0],
        "y_pred_p2_train":      Y_pred[:, 1],
        "y_pred_peg_train":     Y_pred[:, 2],
        "X_train_proc":         X_train,
        "wn":                   None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MCR-ALS
# ─────────────────────────────────────────────────────────────────────────────

def run_mcr(
    X_data:       np.ndarray,
    ST_init:      np.ndarray,
    n_components: int,
    max_iter:     int   = 2000,
    tol_increase: float = 1e-2,
    c_regr:       str   = "OLS",
    st_regr:      str   = "NNLS",
) -> tuple:
    """
    Run MCR-ALS on X_data initialised from ST_init.

    Returns
    -------
    C_mcr  : ndarray  (n_spectra × n_components)  concentration profiles
    ST_mcr : ndarray  (n_components × n_wn)        recovered pure spectra
    n_iter : int                                    iterations to convergence
    """
    mcr = McrAR(
        max_iter=max_iter,
        st_regr=st_regr,
        c_regr=c_regr,
        c_constraints=[ConstraintNonneg(), ConstraintNorm()],
        st_constraints=[ConstraintNonneg()],
        tol_increase=tol_increase,
    )
    mcr.fit(X_data, ST=ST_init[:n_components])
    return mcr.C_, mcr.ST_, mcr.n_iter
