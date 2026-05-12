"""
decomposition.py — Spectral decomposition for the PEARL app.

Covers:
  - PLS calibration (single-output and dual protein+molecular crowder)
    with automatic component selection via the 1-standard-error rule
  - MCR-ALS via pymcr
"""

import numpy as np
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg, ConstraintNorm, Constraint
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split


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
    kfold = RepeatedKFold(n_splits=min(cv_folds, X.shape[0]), n_repeats=3, random_state=42)
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
    test_size: float = 0.2,
) -> dict:
    """
    Train a single-output PLS model with automatic component selection.

    The data are split 80/20 (train/test).  Repeated k-fold CV on the 80 %
    training set selects the optimal number of components; the final model is
    trained on the 80 % and evaluated on the held-out 20 % (RMSEP / R²_test).

    Returns a dict with the fitted model, feature mask, metrics, and
    predictions — ready to be augmented with wn/X_train_proc by the caller.
    """
    if test_size > 0 and len(y_train) >= max(5, int(1 / test_size) + 1):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42
        )
    else:
        X_tr, y_tr = X_train, y_train
        X_te, y_te = X_train[:0], y_train[:0]

    n_opt, rmse_cv, rmse_train = _find_optimal_components(
        X_tr, y_tr,
        max_components=max_components,
        cv_folds=cv_folds,
    )

    feat_std       = np.std(X_tr, axis=0)
    valid_features = feat_std > 1e-10

    X_fit  = X_tr[:, valid_features]
    x_mean = X_fit.mean(axis=0)
    y_mean = np.atleast_1d(np.mean(y_tr, axis=0))

    model  = PLSRegression(n_components=n_opt, scale=False)
    model.fit(X_fit, y_tr)
    y_pred_train = model.predict(X_fit).flatten()

    if len(y_te) >= 2:
        y_pred_test = model.predict(X_te[:, valid_features]).flatten()
        rmse_test   = float(np.sqrt(mean_squared_error(y_te, y_pred_test)))
        r2_test     = float(r2_score(y_te, y_pred_test))
    else:
        y_pred_test = np.array([])
        rmse_test   = float("nan")
        r2_test     = float("nan")

    return {
        "model":          model,
        "valid_features": valid_features,
        "n_components":   n_opt,
        "rmse_train":     float(np.sqrt(mean_squared_error(y_tr, y_pred_train))),
        "r2_train":       float(r2_score(y_tr, y_pred_train)),
        "cv_rmse":        float(rmse_cv[n_opt - 1]),
        "rmse_test":      rmse_test,
        "r2_test":        r2_test,
        "rmse_cv_all":    rmse_cv,
        "rmse_train_all": rmse_train,
        "y_train":        y_tr,
        "y_pred_train":   y_pred_train,
        "y_test":         y_te,
        "y_pred_test":    y_pred_test,
        "x_mean":         x_mean,
        "y_mean":         y_mean,
        "x_loadings":     model.x_loadings_.copy(),
        "y_loadings":     model.y_loadings_.copy(),
    }


def build_dual_pls_model(
    X_protein: np.ndarray, y_protein: np.ndarray,
    X_peg:     np.ndarray, y_peg:     np.ndarray,
    max_components: int = 20,
    cv_folds: int = 5,
    test_size: float = 0.2,
) -> dict:
    """
    Train a multi-output PLS2 model predicting protein and molecular crowder simultaneously.

    Each component's standards are split 80/20 independently.  Repeated k-fold CV
    on the combined 80 % training set selects the optimal number of components;
    per-output RMSEP is computed on the held-out 20 % of each component's standards.

    Returns a dict with 'dual': True so process_linescan dispatches correctly.
    """
    def _split(X, y):
        if test_size > 0 and len(y) >= max(5, int(1 / test_size) + 1):
            return train_test_split(X, y, test_size=test_size, random_state=42)
        return X, X[:0], y, y[:0]

    X_prot_tr, X_prot_te, y_prot_tr, y_prot_te = _split(X_protein, y_protein)
    X_peg_tr,  X_peg_te,  y_peg_tr,  y_peg_te  = _split(X_peg,     y_peg)

    X_train         = np.vstack([X_prot_tr, X_peg_tr])
    y_train_protein = np.concatenate([y_prot_tr,              np.zeros(len(y_peg_tr))])
    y_train_peg     = np.concatenate([np.zeros(len(y_prot_tr)), y_peg_tr])
    Y_train         = np.column_stack([y_train_protein, y_train_peg])

    n_opt, rmse_cv, rmse_train = _find_optimal_components(
        X_train, Y_train,
        max_components=max_components,
        cv_folds=cv_folds,
    )

    feat_std       = np.std(X_train, axis=0)
    valid_features = feat_std > 1e-10

    X_fit  = X_train[:, valid_features]
    x_mean = X_fit.mean(axis=0)
    y_mean = np.atleast_1d(np.mean(Y_train, axis=0))

    model  = PLSRegression(n_components=n_opt, scale=False)
    model.fit(X_fit, Y_train)
    Y_pred = model.predict(X_fit)

    rmse_protein = float(np.sqrt(mean_squared_error(y_train_protein, Y_pred[:, 0])))
    r2_protein   = float(r2_score(y_train_protein, Y_pred[:, 0]))
    rmse_peg     = float(np.sqrt(mean_squared_error(y_train_peg,     Y_pred[:, 1])))
    r2_peg       = float(r2_score(y_train_peg,     Y_pred[:, 1]))

    # Per-output RMSEP on the held-out test sets
    def _rmsep(X_te, y_te, col):
        if len(y_te) < 1:
            return float("nan"), float("nan"), np.array([]), np.array([])
        yp = model.predict(X_te[:, valid_features])[:, col]
        rmsep = float(np.sqrt(mean_squared_error(y_te, yp)))
        r2    = float(r2_score(y_te, yp)) if len(y_te) >= 2 else float("nan")
        return rmsep, r2, y_te, yp

    rmsep_protein, r2_test_protein, y_test_protein, y_pred_test_protein = _rmsep(X_prot_te, y_prot_te, 0)
    rmsep_peg,     r2_test_peg,     y_test_peg,     y_pred_test_peg     = _rmsep(X_peg_te,  y_peg_te,  1)

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
        "cv_rmse_protein":       rmsep_protein,   # test RMSEP for protein
        "cv_rmse_peg":           rmsep_peg,        # test RMSEP for crowder
        "rmse_cv_all":           rmse_cv,
        "rmse_train_all":        rmse_train,
        "y_protein_train":       y_train_protein,
        "y_peg_train":           y_train_peg,
        "y_pred_protein_train":  Y_pred[:, 0],
        "y_pred_peg_train":      Y_pred[:, 1],
        "y_test_protein":        y_test_protein,
        "y_pred_test_protein":   y_pred_test_protein,
        "y_test_peg":            y_test_peg,
        "y_pred_test_peg":       y_pred_test_peg,
        "X_train_proc":          X_train,
        "wn":                    None,   # filled by app.py after training
        "x_mean":                x_mean,
        "y_mean":                y_mean,
        "x_loadings":            model.x_loadings_.copy(),
        "y_loadings":            model.y_loadings_.copy(),
    }


def build_triple_pls_model(
    X_p1: np.ndarray, y_p1: np.ndarray,
    X_p2: np.ndarray, y_p2: np.ndarray,
    X_peg: np.ndarray, y_peg: np.ndarray,
    max_components: int = 20,
    cv_folds: int = 5,
    test_size: float = 0.2,
) -> dict:
    """
    Train a multi-output PLS2 model predicting two proteins and a molecular crowder
    simultaneously.  Each component's standards are split 80/20 independently.
    Repeated k-fold CV on the combined 80 % training set selects the optimal number
    of components; per-output RMSEP is computed on the held-out 20 %.

    Returns a dict with 'triple': True and 'dual': True so downstream code that
    only checks 'dual' still dispatches correctly.
    """
    def _split(X, y):
        if test_size > 0 and len(y) >= max(5, int(1 / test_size) + 1):
            return train_test_split(X, y, test_size=test_size, random_state=42)
        return X, X[:0], y, y[:0]

    X_p1_tr, X_p1_te, y_p1_tr, y_p1_te = _split(X_p1,  y_p1)
    X_p2_tr, X_p2_te, y_p2_tr, y_p2_te = _split(X_p2,  y_p2)
    X_peg_tr, X_peg_te, y_peg_tr, y_peg_te = _split(X_peg, y_peg)

    n1, n2, np_ = len(y_p1_tr), len(y_p2_tr), len(y_peg_tr)
    X_train  = np.vstack([X_p1_tr, X_p2_tr, X_peg_tr])
    y_t_p1   = np.concatenate([y_p1_tr,          np.zeros(n2),   np.zeros(np_)])
    y_t_p2   = np.concatenate([np.zeros(n1),      y_p2_tr,        np.zeros(np_)])
    y_t_peg  = np.concatenate([np.zeros(n1),      np.zeros(n2),   y_peg_tr     ])
    Y_train  = np.column_stack([y_t_p1, y_t_p2, y_t_peg])

    n_opt, rmse_cv, rmse_train = _find_optimal_components(
        X_train, Y_train,
        max_components=max_components,
        cv_folds=cv_folds,
    )

    feat_std       = np.std(X_train, axis=0)
    valid_features = feat_std > 1e-10

    X_fit  = X_train[:, valid_features]
    x_mean = X_fit.mean(axis=0)
    y_mean = np.atleast_1d(np.mean(Y_train, axis=0))

    model  = PLSRegression(n_components=n_opt, scale=False)
    model.fit(X_fit, Y_train)
    Y_pred = model.predict(X_fit)

    rmse_p1  = float(np.sqrt(mean_squared_error(y_t_p1,  Y_pred[:, 0])))
    r2_p1    = float(r2_score(y_t_p1,  Y_pred[:, 0]))
    rmse_p2  = float(np.sqrt(mean_squared_error(y_t_p2,  Y_pred[:, 1])))
    r2_p2    = float(r2_score(y_t_p2,  Y_pred[:, 1]))
    rmse_peg = float(np.sqrt(mean_squared_error(y_t_peg, Y_pred[:, 2])))
    r2_peg   = float(r2_score(y_t_peg, Y_pred[:, 2]))

    # Per-output RMSEP on the held-out test sets
    def _rmsep(X_te, y_te, col):
        if len(y_te) < 1:
            return float("nan"), float("nan"), np.array([]), np.array([])
        yp = model.predict(X_te[:, valid_features])[:, col]
        rmsep = float(np.sqrt(mean_squared_error(y_te, yp)))
        r2    = float(r2_score(y_te, yp)) if len(y_te) >= 2 else float("nan")
        return rmsep, r2, y_te, yp

    rmsep_p1,  r2_test_p1,  y_test_p1,  y_pred_test_p1  = _rmsep(X_p1_te,  y_p1_te,  0)
    rmsep_p2,  r2_test_p2,  y_test_p2,  y_pred_test_p2  = _rmsep(X_p2_te,  y_p2_te,  1)
    rmsep_peg, r2_test_peg, y_test_peg, y_pred_test_peg  = _rmsep(X_peg_te, y_peg_te, 2)

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
        "cv_rmse_p1":           rmsep_p1,    # test RMSEP for protein 1
        "cv_rmse_p2":           rmsep_p2,    # test RMSEP for protein 2
        "cv_rmse_peg":          rmsep_peg,   # test RMSEP for crowder
        # backward-compat aliases so dual code paths don't KeyError on triple
        "cv_rmse_protein":      rmsep_p1,
        "r2_protein_train":     r2_p1,
        "rmse_protein_train":   rmse_p1,
        "rmse_cv_all":          rmse_cv,
        "rmse_train_all":       rmse_train,
        "y_p1_train":           y_t_p1,
        "y_p2_train":           y_t_p2,
        "y_peg_train":          y_t_peg,
        "y_pred_p1_train":      Y_pred[:, 0],
        "y_pred_p2_train":      Y_pred[:, 1],
        "y_pred_peg_train":     Y_pred[:, 2],
        "y_test_p1":            y_test_p1,
        "y_pred_test_p1":       y_pred_test_p1,
        "y_test_p2":            y_test_p2,
        "y_pred_test_p2":       y_pred_test_p2,
        "y_test_peg":           y_test_peg,
        "y_pred_test_peg":      y_pred_test_peg,
        "X_train_proc":         X_train,
        "wn":                   None,
        "x_mean":               x_mean,
        "y_mean":               y_mean,
        "x_loadings":           model.x_loadings_.copy(),
        "y_loadings":           model.y_loadings_.copy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MCR-ALS
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintFixedRows(Constraint):
    """
    Fixed-row (hard equality) constraint for ST spectra.

    After each ST update step in MCR-ALS, the specified rows of the spectral
    matrix are reset to their reference values. This prevents selected pure-
    component spectra from drifting during alternating least-squares iteration,
    which is useful when one or more components are known a priori (e.g. a
    pure-solvent spectrum) and should not be modified by the optimiser.

    Bases: :class:`pymcr.constraints.Constraint`

    Parameters
    ----------
    fixed_indices : list of int
        Zero-based row indices of the ST matrix to hold fixed. Each index
        must be in the range ``[0, n_components)``.
    fixed_spectra : array-like, shape (n_fixed, n_wavenumbers)
        Reference spectral intensities to restore at each iteration. Row *i*
        corresponds to ``fixed_indices[i]``. Values are not required to be
        non-negative; however, combining this constraint with
        :class:`pymcr.constraints.ConstraintNonneg` (applied first) is
        recommended to keep the overall ST matrix physically meaningful.

    Attributes
    ----------
    fixed_indices : list of int
        Stored copy of the row indices provided at initialisation.
    fixed_spectra : ndarray, shape (n_fixed, n_wavenumbers)
        Stored copy of the reference spectra provided at initialisation.

    Notes
    -----
    * The constraint is applied as the *last* element of ``st_constraints``
      in :class:`pymcr.mcr.McrAR`. Placing it after
      :class:`~pymcr.constraints.ConstraintNonneg` ensures that the free
      (non-fixed) rows are clipped to non-negative values first, while the
      fixed rows are then restored exactly to their reference values,
      overriding any clipping that may have been applied to those rows.
    * Only the ST (spectral) matrix is affected. The concentration matrix C
      is updated freely by the C-regression step and its own constraints.
    * Setting ``fixed_indices`` to an empty list is equivalent to applying
      no constraint; the ``transform`` method returns a copy of the input
      unchanged.

    Examples
    --------
    Pin the first row of ST to a known pure-water spectrum:

    >>> import numpy as np
    >>> from pymcr.mcr import McrAR
    >>> from pymcr.constraints import ConstraintNonneg, ConstraintNorm
    >>> water_ref = np.load("water_reference.npy")   # shape (n_wn,)
    >>> constraint = ConstraintFixedRows(
    ...     fixed_indices=[0],
    ...     fixed_spectra=water_ref[np.newaxis, :],
    ... )
    >>> mcr = McrAR(
    ...     st_constraints=[ConstraintNonneg(), constraint],
    ...     c_constraints=[ConstraintNonneg(), ConstraintNorm()],
    ... )
    >>> mcr.fit(X, ST=ST_init)
    """

    def __init__(self, fixed_indices, fixed_spectra):
        super().__init__(copy=True)
        self.fixed_indices = list(fixed_indices)
        self.fixed_spectra = np.array(fixed_spectra)

    def transform(self, A):
        """
        Reset fixed rows of A to their reference spectra.

        Parameters
        ----------
        A : ndarray, shape (n_components, n_wavenumbers)
            Spectral matrix ST as updated by the ST-regression step.

        Returns
        -------
        A : ndarray, shape (n_components, n_wavenumbers)
            Copy of the input with the fixed rows overwritten by their
            reference values.
        """
        A = A.copy()
        for i, idx in enumerate(self.fixed_indices):
            A[idx] = self.fixed_spectra[i]
        return A


def run_mcr(
    X_data:        np.ndarray,
    ST_init:       np.ndarray,
    n_components:  int,
    max_iter:      int        = 2000,
    tol_increase:  float      = 1e-2,
    c_regr:        str        = "OLS",
    st_regr:       str        = "NNLS",
    fixed_st_idx:  list       = None,
    fixed_st_vals: np.ndarray = None,
    norm_c:        str        = "sum to 1 (default)",
) -> tuple:
    """
    Run MCR-ALS on X_data initialised from ST_init.

    Parameters
    ----------
    fixed_st_idx  : list of int, optional
        0-based row indices in ST to hold fixed during every iteration.
    fixed_st_vals : ndarray (n_fixed × n_wn), optional
        Reference spectral values for the fixed rows.

    Returns
    -------
    C_mcr  : ndarray  (n_spectra × n_components)  concentration profiles
    ST_mcr : ndarray  (n_components × n_wn)        recovered pure spectra
    n_iter : int                                    iterations to convergence
    """
    # ── CLS mode: all components are fixed → direct NNLS unmixing ────────────
    # When every spectral component is pinned, MCR-ALS is degenerate: it can
    # drive the fixed component's concentration to zero and let the free
    # components absorb its variance.  If all n_components rows are fixed,
    # bypass ALS entirely and solve one NNLS problem per spectrum instead.
    if fixed_st_idx and len(fixed_st_idx) >= n_components:
        from scipy.optimize import nnls as _nnls
        ST_fixed = fixed_st_vals[:n_components]          # (n_comp, n_wn)
        C_cls = np.zeros((X_data.shape[0], n_components))
        for i in range(X_data.shape[0]):
            C_cls[i], _ = _nnls(ST_fixed.T, X_data[i])
        return C_cls, ST_fixed, 0                        # 0 ALS iterations

    # ── Partial-fix MCR mode ──────────────────────────────────────────────────
    st_constraints = [ConstraintNonneg()]
    _tol_n_increase  = 10
    _tol_n_above_min = 10
    if fixed_st_idx:
        st_constraints.append(ConstraintFixedRows(fixed_st_idx, fixed_st_vals))
        # Fixed rows cause the residual to bounce; disable all tolerance-based
        # stopping so only max_iter controls when the loop ends.
        tol_increase     = None
        _tol_n_increase  = None
        _tol_n_above_min = None

    if norm_c == "off":
        _c_constraints = [ConstraintNonneg()]
    else:  # "sum to 1 (default)"
        _c_constraints = [ConstraintNonneg(), ConstraintNorm()]

    mcr = McrAR(
        max_iter=max_iter,
        st_regr=st_regr,
        c_regr=c_regr,
        c_constraints=_c_constraints,
        st_constraints=st_constraints,
        tol_increase=tol_increase,
        tol_n_increase=_tol_n_increase,
        tol_n_above_min=_tol_n_above_min,
    )
    mcr.fit(X_data, ST=ST_init[:n_components])
    return mcr.C_, mcr.ST_, mcr.n_iter
