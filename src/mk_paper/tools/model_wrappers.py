"""Wrappers deterministas de modelos estadísticos / ML (sin código libre del LLM).

Cada función es tipada, encapsula scikit-learn / statsmodels y captura errores
con traceback para el bucle de auto-corrección del Quantitative Analyst.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge


@dataclass
class WrapperResult:
    """Resultado uniforme de un wrapper matemático."""

    status: str  # ok | error
    model_id: str
    estimator: Any = None
    y_train_pred: np.ndarray | None = None
    y_test_pred: np.ndarray | None = None
    coefficients: dict[str, float] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    message: str | None = None
    traceback: str | None = None


def _fail(model_id: str, exc: BaseException) -> WrapperResult:
    return WrapperResult(
        status="error",
        model_id=model_id,
        error_type=type(exc).__name__,
        message=str(exc),
        traceback=traceback.format_exc(),
    )


def _coef_map(estimator: Any, columns: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim > 1:
            coef = coef[0]
        out = {c: float(v) for c, v in zip(columns, coef, strict=False)}
        if hasattr(estimator, "intercept_"):
            intercept = estimator.intercept_
            if np.ndim(intercept) == 0:
                out["intercept"] = float(intercept)
    return out


def _importance_map(estimator: Any, columns: list[str]) -> dict[str, float]:
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        return {c: float(v) for c, v in zip(columns, values, strict=False)}
    return {}


def run_pca(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    *,
    n_components: int | float | None = 0.95,
    random_state: int = 42,
) -> WrapperResult:
    """PCA tipado sobre features (preprocesado / exploración)."""
    model_id = "pca"
    try:
        n_features = int(x_train.shape[1])
        if n_features < 1:
            raise ValueError("PCA requires at least one feature column")
        if n_components is None:
            n_components = min(n_features, 3)
        pca = PCA(n_components=n_components, random_state=random_state)
        z_train = pca.fit_transform(x_train)
        z_test = pca.transform(x_test)
        cols = [f"PC{i + 1}" for i in range(z_train.shape[1])]
        return WrapperResult(
            status="ok",
            model_id=model_id,
            estimator=pca,
            extras={
                "n_components": int(z_train.shape[1]),
                "explained_variance_ratio": [
                    float(v) for v in pca.explained_variance_ratio_
                ],
                "cumulative_explained_variance": float(
                    np.cumsum(pca.explained_variance_ratio_)[-1]
                ),
                "loadings": {
                    cols[j]: {
                        str(x_train.columns[i]): float(pca.components_[j, i])
                        for i in range(n_features)
                    }
                    for j in range(z_train.shape[1])
                },
                "x_train_pca": pd.DataFrame(z_train, index=x_train.index, columns=cols),
                "x_test_pca": pd.DataFrame(z_test, index=x_test.index, columns=cols),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(model_id, exc)


def run_ols_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> WrapperResult:
    """Regresión lineal OLS (statsmodels)."""
    model_id = "ols"
    try:
        x_tr = sm.add_constant(x_train, has_constant="add")
        x_te = sm.add_constant(x_test, has_constant="add")
        model = sm.OLS(y_train, x_tr).fit()
        y_tr_hat = np.asarray(model.predict(x_tr))
        y_te_hat = np.asarray(model.predict(x_te))
        coefs = {str(k): float(v) for k, v in model.params.items()}
        return WrapperResult(
            status="ok",
            model_id=model_id,
            estimator=model,
            y_train_pred=y_tr_hat,
            y_test_pred=y_te_hat,
            coefficients=coefs,
            extras={"aic": float(model.aic), "bic": float(model.bic)},
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(model_id, exc)


def run_ridge_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    alpha: float = 1.0,
    random_state: int = 42,
) -> WrapperResult:
    """Ridge regression (scikit-learn)."""
    model_id = "ridge"
    try:
        est = Ridge(alpha=float(alpha), random_state=random_state)
        est.fit(x_train, y_train)
        cols = [str(c) for c in x_train.columns]
        return WrapperResult(
            status="ok",
            model_id=model_id,
            estimator=est,
            y_train_pred=np.asarray(est.predict(x_train)),
            y_test_pred=np.asarray(est.predict(x_test)),
            coefficients=_coef_map(est, cols),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(model_id, exc)


def run_lasso_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    alpha: float = 0.01,
    max_iter: int = 5000,
    random_state: int = 42,
) -> WrapperResult:
    """Lasso regression (scikit-learn)."""
    model_id = "lasso"
    try:
        est = Lasso(
            alpha=float(alpha), max_iter=int(max_iter), random_state=random_state
        )
        est.fit(x_train, y_train)
        cols = [str(c) for c in x_train.columns]
        return WrapperResult(
            status="ok",
            model_id=model_id,
            estimator=est,
            y_train_pred=np.asarray(est.predict(x_train)),
            y_test_pred=np.asarray(est.predict(x_test)),
            coefficients=_coef_map(est, cols),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(model_id, exc)


def run_elasticnet_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
    random_state: int = 42,
) -> WrapperResult:
    """ElasticNet regression (scikit-learn)."""
    model_id = "elasticnet"
    try:
        est = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            max_iter=int(max_iter),
            random_state=random_state,
        )
        est.fit(x_train, y_train)
        cols = [str(c) for c in x_train.columns]
        return WrapperResult(
            status="ok",
            model_id=model_id,
            estimator=est,
            y_train_pred=np.asarray(est.predict(x_train)),
            y_test_pred=np.asarray(est.predict(x_test)),
            coefficients=_coef_map(est, cols),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(model_id, exc)


def run_random_forest_regressor(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> WrapperResult:
    """Random Forest regressor (scikit-learn)."""
    model_id = "random_forest_reg"
    try:
        est = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=random_state,
            n_jobs=1,
        )
        est.fit(x_train, y_train)
        cols = [str(c) for c in x_train.columns]
        return WrapperResult(
            status="ok",
            model_id=model_id,
            estimator=est,
            y_train_pred=np.asarray(est.predict(x_train)),
            y_test_pred=np.asarray(est.predict(x_test)),
            feature_importances=_importance_map(est, cols),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(model_id, exc)


def run_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> WrapperResult:
    """Logistic regression classifier (scikit-learn)."""
    model_id = "logistic"
    try:
        est = LogisticRegression(
            C=float(C),
            max_iter=int(max_iter),
            random_state=random_state,
            n_jobs=1,
        )
        est.fit(x_train, y_train)
        cols = [str(c) for c in x_train.columns]
        return WrapperResult(
            status="ok",
            model_id=model_id,
            estimator=est,
            y_train_pred=np.asarray(est.predict(x_train)),
            y_test_pred=np.asarray(est.predict(x_test)),
            coefficients=_coef_map(est, cols),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(model_id, exc)


def run_random_forest_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> WrapperResult:
    """Random Forest classifier (scikit-learn)."""
    model_id = "random_forest_clf"
    try:
        est = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=random_state,
            n_jobs=1,
        )
        est.fit(x_train, y_train)
        cols = [str(c) for c in x_train.columns]
        return WrapperResult(
            status="ok",
            model_id=model_id,
            estimator=est,
            y_train_pred=np.asarray(est.predict(x_train)),
            y_test_pred=np.asarray(est.predict(x_test)),
            feature_importances=_importance_map(est, cols),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(model_id, exc)


def run_gradient_boosting_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 42,
) -> WrapperResult:
    """Gradient Boosting classifier (scikit-learn)."""
    model_id = "gradient_boosting_clf"
    try:
        est = GradientBoostingClassifier(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            random_state=random_state,
        )
        est.fit(x_train, y_train)
        cols = [str(c) for c in x_train.columns]
        return WrapperResult(
            status="ok",
            model_id=model_id,
            estimator=est,
            y_train_pred=np.asarray(est.predict(x_train)),
            y_test_pred=np.asarray(est.predict(x_test)),
            feature_importances=_importance_map(est, cols),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(model_id, exc)


def dispatch_supervised_wrapper(
    model_id: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> WrapperResult:
    """Enruta model_id → wrapper tipado (único punto de extensión)."""
    p = dict(params or {})
    mid = model_id.strip().lower()
    try:
        if mid == "ols":
            return run_ols_regression(x_train, y_train, x_test)
        if mid == "ridge":
            return run_ridge_regression(
                x_train,
                y_train,
                x_test,
                alpha=float(p.get("alpha", 1.0)),
                random_state=random_state,
            )
        if mid == "lasso":
            return run_lasso_regression(
                x_train,
                y_train,
                x_test,
                alpha=float(p.get("alpha", 0.01)),
                max_iter=int(p.get("max_iter", 5000)),
                random_state=random_state,
            )
        if mid == "elasticnet":
            return run_elasticnet_regression(
                x_train,
                y_train,
                x_test,
                alpha=float(p.get("alpha", 0.01)),
                l1_ratio=float(p.get("l1_ratio", 0.5)),
                max_iter=int(p.get("max_iter", 5000)),
                random_state=random_state,
            )
        if mid == "random_forest_reg":
            return run_random_forest_regressor(
                x_train,
                y_train,
                x_test,
                n_estimators=int(p.get("n_estimators", 100)),
                max_depth=p.get("max_depth"),
                min_samples_leaf=int(p.get("min_samples_leaf", 1)),
                random_state=random_state,
            )
        if mid == "logistic":
            return run_logistic_regression(
                x_train,
                y_train,
                x_test,
                C=float(p.get("C", 1.0)),
                max_iter=int(p.get("max_iter", 2000)),
                random_state=random_state,
            )
        if mid == "random_forest_clf":
            return run_random_forest_classifier(
                x_train,
                y_train,
                x_test,
                n_estimators=int(p.get("n_estimators", 100)),
                max_depth=p.get("max_depth"),
                min_samples_leaf=int(p.get("min_samples_leaf", 1)),
                random_state=random_state,
            )
        if mid == "gradient_boosting_clf":
            return run_gradient_boosting_classifier(
                x_train,
                y_train,
                x_test,
                n_estimators=int(p.get("n_estimators", 100)),
                learning_rate=float(p.get("learning_rate", 0.1)),
                max_depth=int(p.get("max_depth", 3)),
                random_state=random_state,
            )
        raise ValueError(f"No deterministic wrapper registered for model_id={mid!r}")
    except Exception as exc:  # noqa: BLE001
        return _fail(mid, exc)
