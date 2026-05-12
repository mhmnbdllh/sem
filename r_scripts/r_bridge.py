"""
r_bridge.py
===========
Python-R bridge for SEM Studio.
Connects Streamlit (Python) to lavaan (R) via rpy2.

All SEM computations are delegated to R/lavaan.
Results are returned as Python dicts for use in Streamlit.
"""

import os
import numpy as np
import pandas as pd

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, r
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False


# ── Load R script once ───────────────────────────────────────────

_R_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "sem_analysis.R")
_R_LOADED = False


def load_r_script():
    """Load the R analysis script into the R environment."""
    global _R_LOADED
    if not _R_LOADED and RPY2_AVAILABLE:
        try:
            r.source(_R_SCRIPT_PATH)
            _R_LOADED = True
        except Exception as e:
            raise RuntimeError(f"Failed to load R script: {e}")


def check_r_available() -> dict:
    """Check if R and required packages are available."""
    if not RPY2_AVAILABLE:
        return {
            "available": False,
            "message": "rpy2 is not installed. Please add it to requirements.txt."
        }
    try:
        load_r_script()
        return {"available": True, "message": "R and lavaan are ready."}
    except Exception as e:
        return {"available": False, "message": str(e)}


# ── R object converters ──────────────────────────────────────────

def r_to_py(obj):
    """
    Recursively convert rpy2 R objects to Python native types.
    Handles lists, vectors, dataframes, NULL, NA.
    """
    if obj is None or obj == ro.NULL:
        return None

    # Named list → dict
    if hasattr(obj, 'names') and obj.names != ro.NULL:
        names = list(obj.names)
        result = {}
        for i, name in enumerate(names):
            try:
                result[name] = r_to_py(obj[i])
            except Exception:
                result[name] = None
        return result

    # DataFrame
    if isinstance(obj, ro.DataFrame):
        with localconverter(ro.default_converter + pandas2ri.converter):
            return ro.conversion.rpy2py(obj)

    # Numeric vector
    if isinstance(obj, ro.FloatVector):
        arr = list(obj)
        # Handle NA
        arr = [None if (isinstance(v, float) and np.isnan(v)) else v for v in arr]
        if hasattr(obj, 'names') and obj.names != ro.NULL:
            return dict(zip(list(obj.names), arr))
        return arr[0] if len(arr) == 1 else arr

    # Integer vector
    if isinstance(obj, ro.IntVector):
        arr = list(obj)
        if hasattr(obj, 'names') and obj.names != ro.NULL:
            return dict(zip(list(obj.names), arr))
        return arr[0] if len(arr) == 1 else arr

    # Character vector
    if isinstance(obj, ro.StrVector):
        arr = list(obj)
        if hasattr(obj, 'names') and obj.names != ro.NULL:
            return dict(zip(list(obj.names), arr))
        return arr[0] if len(arr) == 1 else arr

    # Logical vector
    if isinstance(obj, ro.BoolVector):
        arr = list(obj)
        return arr[0] if len(arr) == 1 else arr

    # Generic list
    if isinstance(obj, ro.ListVector):
        if hasattr(obj, 'names') and obj.names != ro.NULL:
            names = list(obj.names)
            return {name: r_to_py(obj[i]) for i, name in enumerate(names)}
        return [r_to_py(obj[i]) for i in range(len(obj))]

    # Matrix
    if isinstance(obj, ro.Matrix):
        mat = np.array(obj)
        row_names = list(obj.rownames) if obj.rownames != ro.NULL else None
        col_names = list(obj.colnames) if obj.colnames != ro.NULL else None
        df = pd.DataFrame(mat, index=row_names, columns=col_names)
        return df

    # Fallback
    try:
        return list(obj)[0] if len(obj) == 1 else list(obj)
    except Exception:
        return str(obj)


def df_to_r(df: pd.DataFrame):
    """Convert pandas DataFrame to R dataframe."""
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


def list_to_r_list(d: dict):
    """Convert Python dict to R named list."""
    items = []
    for k, v in d.items():
        if isinstance(v, list):
            items.append(ro.StrVector(v))
        elif isinstance(v, str):
            items.append(ro.StrVector([v]))
        else:
            items.append(ro.StrVector(list(v)))
    return ro.ListVector(dict(zip(d.keys(), items)))


# ── Public API ───────────────────────────────────────────────────

def run_descriptives(df: pd.DataFrame, indicator_cols: list) -> dict:
    """Run descriptive statistics via R/psych."""
    if not RPY2_AVAILABLE:
        return {"error": "R not available"}
    try:
        load_r_script()
        r_data = df_to_r(df[indicator_cols])
        result = r["run_descriptives"](r_data)
        return r_to_py(result)
    except Exception as e:
        return {"error": str(e)}


def run_mardia(df: pd.DataFrame, indicator_cols: list) -> dict:
    """Run Mardia's multivariate normality test via R/psych."""
    if not RPY2_AVAILABLE:
        return {"error": "R not available"}
    try:
        load_r_script()
        r_data = df_to_r(df[indicator_cols].dropna())
        result = r["run_mardia"](r_data)
        return r_to_py(result)
    except Exception as e:
        return {"error": str(e)}


def run_harman(df: pd.DataFrame, indicator_cols: list) -> dict:
    """Run Harman's single factor test via R/psych."""
    if not RPY2_AVAILABLE:
        return {"error": "R not available"}
    try:
        load_r_script()
        r_data = df_to_r(df[indicator_cols].dropna())
        result = r["run_harman"](r_data)
        return r_to_py(result)
    except Exception as e:
        return {"error": str(e)}


def run_efa(df: pd.DataFrame, indicator_cols: list,
            n_factors: int, rotation: str = "oblimin") -> dict:
    """Run EFA via R/psych with PAF extraction."""
    if not RPY2_AVAILABLE:
        return {"error": "R not available"}
    try:
        load_r_script()
        r_data     = df_to_r(df[indicator_cols].dropna())
        r_nfactors = ro.IntVector([n_factors])
        r_rotation = ro.StrVector([rotation])
        result     = r["run_efa"](r_data, r_nfactors, r_rotation)
        return r_to_py(result)
    except Exception as e:
        return {"error": str(e)}


def run_cfa(df: pd.DataFrame, indicator_cols: list,
            model_syntax: str, estimator: str = "MLR") -> dict:
    """Run CFA via R/lavaan."""
    if not RPY2_AVAILABLE:
        return {"error": "R not available"}
    try:
        load_r_script()
        r_data      = df_to_r(df[indicator_cols].dropna())
        r_syntax    = ro.StrVector([model_syntax])
        r_estimator = ro.StrVector([estimator])
        result      = r["run_cfa"](r_data, r_syntax, r_estimator)
        return r_to_py(result)
    except Exception as e:
        return {"error": str(e)}


def run_sem(df: pd.DataFrame, all_cols: list,
            model_syntax: str, estimator: str = "MLR") -> dict:
    """Run full SEM via R/lavaan."""
    if not RPY2_AVAILABLE:
        return {"error": "R not available"}
    try:
        load_r_script()
        r_data      = df_to_r(df[all_cols].dropna())
        r_syntax    = ro.StrVector([model_syntax])
        r_estimator = ro.StrVector([estimator])
        result      = r["run_sem"](r_data, r_syntax, r_estimator)
        return r_to_py(result)
    except Exception as e:
        return {"error": str(e)}


def run_mediation(df: pd.DataFrame, all_cols: list,
                  x_var: str, m_var: str, y_var: str,
                  constructs: dict,
                  n_boot: int = 5000,
                  estimator: str = "MLR") -> dict:
    """Run bootstrap mediation via R/lavaan."""
    if not RPY2_AVAILABLE:
        return {"error": "R not available"}
    try:
        load_r_script()
        r_data      = df_to_r(df[all_cols])
        r_x         = ro.StrVector([x_var])
        r_m         = ro.StrVector([m_var])
        r_y         = ro.StrVector([y_var])
        r_constructs= list_to_r_list(constructs)
        r_nboot     = ro.IntVector([n_boot])
        r_estimator = ro.StrVector([estimator])
        result      = r["run_mediation"](
            r_data, r_x, r_m, r_y,
            r_constructs, r_nboot, r_estimator
        )
        return r_to_py(result)
    except Exception as e:
        return {"error": str(e)}


def run_moderation(df: pd.DataFrame, all_cols: list,
                   x_var: str, w_var: str, y_var: str,
                   constructs: dict) -> dict:
    """Run moderation analysis via R."""
    if not RPY2_AVAILABLE:
        return {"error": "R not available"}
    try:
        load_r_script()
        r_data      = df_to_r(df[all_cols])
        r_x         = ro.StrVector([x_var])
        r_w         = ro.StrVector([w_var])
        r_y         = ro.StrVector([y_var])
        r_constructs= list_to_r_list(constructs)
        result      = r["run_moderation"](r_data, r_x, r_w, r_y, r_constructs)
        return r_to_py(result)
    except Exception as e:
        return {"error": str(e)}


def run_invariance(df: pd.DataFrame, indicator_cols: list,
                   model_syntax: str, group_var: str,
                   estimator: str = "MLR") -> dict:
    """Run measurement invariance via R/lavaan."""
    if not RPY2_AVAILABLE:
        return {"error": "R not available"}
    try:
        load_r_script()
        cols        = indicator_cols + [group_var]
        r_data      = df_to_r(df[cols].dropna())
        r_syntax    = ro.StrVector([model_syntax])
        r_group     = ro.StrVector([group_var])
        r_estimator = ro.StrVector([estimator])
        result      = r["run_invariance"](r_data, r_syntax, r_group, r_estimator)
        return r_to_py(result)
    except Exception as e:
        return {"error": str(e)}


def run_model_comparison(df: pd.DataFrame, all_cols: list,
                         models: dict,
                         estimator: str = "MLR") -> dict:
    """Run model comparison via R/lavaan."""
    if not RPY2_AVAILABLE:
        return {"error": "R not available"}
    try:
        load_r_script()
        r_data      = df_to_r(df[all_cols].dropna())
        r_models    = ro.ListVector(
            {k: ro.StrVector([v]) for k, v in models.items()}
        )
        r_estimator = ro.StrVector([estimator])
        result      = r["run_model_comparison"](r_data, r_models, r_estimator)
        return r_to_py(result)
    except Exception as e:
        return {"error": str(e)}
