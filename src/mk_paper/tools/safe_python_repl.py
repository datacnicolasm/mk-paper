"""REPL científico seguro para ejecución dinámica de código del Analista.

El LLM puede escribir Python a la medida del MethodBrief; el código corre
solo sobre datasets locales (sandbox) con librerías científicas preinstaladas.
Errores devuelven traceback completo para auto-corrección.
"""

from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import io
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mk_paper.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_FORBIDDEN_MODULES = frozenset(
    {
        "socket",
        "subprocess",
        "ctypes",
        "multiprocessing",
        "pathlib",  # use provided df / sandboxed open
        "shutil",
        "http",
        "httpx",
        "urllib",
        "requests",
        "aiohttp",
        "ftplib",
        "telnetlib",
        "pickle",
        "dill",
        "cloudpickle",
    }
)

_FORBIDDEN_NAMES = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "breakpoint",
        "exit",
        "quit",
        "input",
    }
)


class UnsafeCodeError(ValueError):
    """Código rechazado por la política de seguridad del REPL."""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sandbox_roots(settings: Settings) -> list[Path]:
    return [
        Path(settings.data_dir).resolve(),
        Path(settings.workspace_dir).resolve(),
        Path(settings.output_dir).resolve(),
    ]


def assert_path_in_sandbox(path: Path, settings: Settings) -> Path:
    """Exige que path resuelto esté bajo data/workspace/output."""
    resolved = path.expanduser().resolve()
    if not any(_is_relative_to(resolved, root) for root in _sandbox_roots(settings)):
        raise PermissionError(
            f"Path outside sandbox (data/workspace/output): {resolved}"
        )
    return resolved


def validate_python_code(code: str) -> None:
    """Analiza AST y rechaza imports/nombres peligrosos (sin red / shell)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCodeError(f"SyntaxError while parsing code: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_MODULES or root not in _ALLOWED_IMPORT_ROOTS:
                    raise UnsafeCodeError(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in _FORBIDDEN_MODULES or (
                mod and mod not in _ALLOWED_IMPORT_ROOTS
            ):
                raise UnsafeCodeError(f"Forbidden import from: {node.module}")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in {
            "system",
            "popen",
            "fork",
            "remove",
            "unlink",
            "rmtree",
        }:
            raise UnsafeCodeError(f"Forbidden attribute access: .{node.attr}")


_ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "pandas",
        "numpy",
        "sklearn",
        "scipy",
        "statsmodels",
        "xgboost",
        "math",
        "statistics",
        "collections",
        "itertools",
        "functools",
        "re",
        "datetime",
        "typing",
        "warnings",
        "copy",
        "json",
    }
)


def _restricted_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Import solo de paquetes científicos / stdlib segura (sin red/shell)."""
    root = name.split(".")[0]
    if root in _FORBIDDEN_MODULES or root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(
            f"Import blocked by analysis REPL policy: {name}. "
            f"Allowed roots: {sorted(_ALLOWED_IMPORT_ROOTS)}"
        )
    return __import__(name, globals, locals, fromlist, level)


def _safe_builtins(settings: Settings) -> dict[str, Any]:
    """Builtins reducidos + open restringido al sandbox."""
    import builtins as _builtins

    allowed = (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "map",
        "filter",
        "isinstance",
        "type",
        "hasattr",
        "getattr",
        "setattr",
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "RuntimeError",
        "ImportError",
        "True",
        "False",
        "None",
    )
    ns: dict[str, Any] = {
        name: getattr(_builtins, name) for name in allowed if hasattr(_builtins, name)
    }

    def sandboxed_open(file: str, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        path = assert_path_in_sandbox(Path(file), settings)
        if any(flag in mode for flag in ("x", "a", "w", "+")):
            if not _is_relative_to(path, Path(settings.output_dir).resolve()):
                raise PermissionError(
                    "Writes are only allowed under OUTPUT_DIR sandbox."
                )
        return _builtins.open(path, mode, *args, **kwargs)

    ns["open"] = sandboxed_open
    ns["__import__"] = _restricted_import
    return ns


def _import_scientific_stack() -> dict[str, Any]:
    """Importa stack científico; scipy/xgboost opcionales si faltan en la imagen."""
    import sklearn
    import statsmodels
    import statsmodels.api as sm

    stack: dict[str, Any] = {
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
        "sklearn": sklearn,
        "statsmodels": statsmodels,
        "sm": sm,
    }
    try:
        import scipy

        stack["scipy"] = scipy
    except ImportError:  # pragma: no cover
        stack["scipy"] = None
        logger.warning("scipy not installed; available as None in REPL namespace")
    try:
        import xgboost as xgb

        stack["xgboost"] = xgb
        stack["xgb"] = xgb
    except ImportError:  # pragma: no cover
        stack["xgboost"] = None
        stack["xgb"] = None
        logger.warning("xgboost not installed; available as None in REPL namespace")
    return stack


def execute_analysis_code(
    *,
    df: pd.DataFrame,
    code: str,
    dataset_path: str,
    settings: Settings | None = None,
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    """Ejecuta ``code`` con ``df`` y librerías científicas en el namespace.

    El script debe asignar un dict serializable a ``results`` (métricas/tablas).
    """
    cfg = settings or get_settings()
    try:
        validate_python_code(code)
    except UnsafeCodeError as exc:
        return {
            "status": "error",
            "error_type": "UnsafeCodeError",
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "stdout": "",
            "results": {},
            "retry_hint": (
                "SELF-CORRECTION: remove forbidden imports/calls "
                "(network, shell, pickle, eval/exec). Use df and scientific "
                "libs already in the namespace, then re-run."
            ),
        }

    stdout_buf = io.StringIO()
    namespace: dict[str, Any] = {
        "__name__": "mk_paper_analysis_repl",
        "__builtins__": _safe_builtins(cfg),
        "df": df.copy(),
        "dataset_path": dataset_path,
        "results": {},
        **_import_scientific_stack(),
    }

    def _run() -> None:
        with contextlib.redirect_stdout(stdout_buf):
            exec(compile(code, "<analysis_repl>", "exec"), namespace, namespace)  # noqa: S102

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            future.result(timeout=max(5.0, float(timeout_seconds)))
    except concurrent.futures.TimeoutError:
        return {
            "status": "error",
            "error_type": "TimeoutError",
            "message": f"Code execution exceeded {timeout_seconds}s",
            "traceback": (
                f"TimeoutError: analysis REPL exceeded {timeout_seconds} seconds"
            ),
            "stdout": stdout_buf.getvalue()[-8000:],
            "results": {},
            "retry_hint": (
                "SELF-CORRECTION: simplify the script, reduce grid search size, "
                "or sample the DataFrame, then re-run Execute Python Analysis."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "stdout": stdout_buf.getvalue()[-8000:],
            "results": {},
            "retry_hint": (
                "SELF-CORRECTION: read the traceback, fix the Python script "
                "(columns, dtypes, shapes, imports), and call "
                "Execute Python Analysis again. Do not use the network."
            ),
        }

    raw_results = namespace.get("results", {})
    try:
        # Garantiza JSON-serializable
        results = json.loads(json.dumps(raw_results, default=str))
    except Exception:  # noqa: BLE001
        results = {"raw": str(raw_results)}

    return {
        "status": "ok",
        "dataset_path": dataset_path,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "stdout": stdout_buf.getvalue()[-12000:],
        "results": results,
        "retry_hint": "",
    }
