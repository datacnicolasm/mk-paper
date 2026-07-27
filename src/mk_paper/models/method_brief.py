"""Modelos del embudo de análisis cuantitativo determinista."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


TaskType = Literal["regression", "classification"]

ALLOWED_REGRESSION_MODELS = (
    "ols",
    "ridge",
    "lasso",
    "elasticnet",
    "random_forest_reg",
)
ALLOWED_CLASSIFICATION_MODELS = (
    "logistic",
    "random_forest_clf",
    "gradient_boosting_clf",
)
ALLOWED_UNSUPERVISED_MODELS = ("pca",)
ALLOWED_MODELS = (
    ALLOWED_REGRESSION_MODELS
    + ALLOWED_CLASSIFICATION_MODELS
    + ALLOWED_UNSUPERVISED_MODELS
)

MetricDirection = Literal["maximize", "minimize"]

PRIMARY_METRIC_DIRECTION: dict[str, MetricDirection] = {
    "rmse": "minimize",
    "mae": "minimize",
    "mse": "minimize",
    "r2": "maximize",
    "accuracy": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "f1": "maximize",
    "f1_macro": "maximize",
    "f1_weighted": "maximize",
}


class RobustnessFlags(BaseModel):
    """Pruebas de robustez opcionales post-mejor-modelo."""

    vif: bool = True
    residual_diagnostics: bool = True  # normalidad / heteroscedasticidad (regresión)
    confusion_matrix: bool = True
    classification_report: bool = True


class LiteratureBenchmark(BaseModel):
    """Umbral/métrica reportada en la literatura para contrastar resultados."""

    metric: str
    reference_value: float
    higher_is_better: bool | None = None  # None → inferir de PRIMARY_METRIC_DIRECTION
    source: str = ""  # DOI, título corto o etiqueta
    note: str = ""


class MethodBrief(BaseModel):
    """Brief metodológico estructurado (entrada del Quantitative Analyst).

    El agente no inventa metodología: solo ejecuta lo declarado aquí sobre
    un archivo local CSV/XLSX, opcionalmente contrastado con literatura local.
    """

    title: str
    task_type: TaskType
    target: str
    features: list[str] = Field(min_length=1)
    controls: list[str] = Field(default_factory=list)
    dataset_path: str
    sheet_name: str | None = None
    test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    cv_folds: int = Field(default=0, ge=0, le=10)
    random_state: int = 42
    primary_metric: str = "rmse"
    models: list[str] = Field(min_length=1)
    hyperparameter_grids: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)
    max_grid_combinations: int = Field(default=50, ge=1, le=500)
    robustness: RobustnessFlags = Field(default_factory=RobustnessFlags)
    average: Literal["binary", "macro", "weighted"] = "weighted"
    dropna: bool = True
    impute_strategy: Literal["drop", "mean", "median"] = "drop"
    run_pca: bool = False
    pca_n_components: int | float | None = 0.95
    # Contexto del Literature Reviewer (solo archivos locales / JSON inline).
    literature_review_path: str | None = None
    literature_context: dict[str, Any] | str | None = None
    literature_benchmarks: list[LiteratureBenchmark] = Field(default_factory=list)
    enrich_discussion_with_llm: bool = True

    @field_validator("models")
    @classmethod
    def _validate_models(cls, value: list[str]) -> list[str]:
        cleaned = [str(m).strip().lower() for m in value if str(m).strip()]
        if not cleaned:
            raise ValueError("models must contain at least one model id")
        unknown = [m for m in cleaned if m not in ALLOWED_MODELS]
        if unknown:
            raise ValueError(
                f"Unsupported models: {unknown}. Allowed: {list(ALLOWED_MODELS)}"
            )
        return cleaned

    @field_validator("primary_metric")
    @classmethod
    def _normalize_metric(cls, value: str) -> str:
        metric = (value or "").strip().lower()
        if metric not in PRIMARY_METRIC_DIRECTION:
            raise ValueError(
                f"Unsupported primary_metric={value!r}. "
                f"Allowed: {sorted(PRIMARY_METRIC_DIRECTION)}"
            )
        return metric

    def predictor_columns(self) -> list[str]:
        """Features + controles, sin duplicados, orden estable."""
        seen: set[str] = set()
        cols: list[str] = []
        for name in [*self.features, *self.controls]:
            key = str(name).strip()
            if key and key not in seen:
                seen.add(key)
                cols.append(key)
        return cols

    def metric_direction(self) -> MetricDirection:
        return PRIMARY_METRIC_DIRECTION[self.primary_metric]


class DatasetSchema(BaseModel):
    """Esquema resumido del dataset usado."""

    n_rows: int = 0
    n_cols: int = 0
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    missing_counts: dict[str, int] = Field(default_factory=dict)
    rows_after_dropna: int = 0


class ModelResult(BaseModel):
    """Resultado de un modelo (mejor configuración encontrada)."""

    model_id: str
    best_params: dict[str, Any] = Field(default_factory=dict)
    metrics_train: dict[str, float] = Field(default_factory=dict)
    metrics_test: dict[str, float] = Field(default_factory=dict)
    cv_mean: float | None = None
    cv_std: float | None = None
    coefficients: dict[str, float] = Field(default_factory=dict)
    feature_importances: dict[str, float] = Field(default_factory=dict)
    n_iterations: int = 0
    notes: list[str] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    """Salida Writer-ready del análisis cuantitativo."""

    brief_title: str
    task_type: TaskType
    dataset_path: str
    dataset_schema: DatasetSchema = Field(default_factory=DatasetSchema)
    primary_metric: str = "rmse"
    best_model: str | None = None
    best_score: float | None = None
    model_results: list[ModelResult] = Field(default_factory=list)
    descriptive_tables: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    robustness_tests: dict[str, Any] = Field(default_factory=dict)
    iteration_log: list[dict[str, Any]] = Field(default_factory=list)
    tables: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    literature_snapshot: dict[str, Any] = Field(default_factory=dict)
    literature_benchmarking: list[dict[str, Any]] = Field(default_factory=list)
    analytical_discussion: str = ""
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict JSON-compatible."""
        return self.model_dump(mode="json")
