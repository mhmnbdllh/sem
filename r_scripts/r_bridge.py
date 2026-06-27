"""
r_bridge.py
===========
Python-R bridge for SEM Studio.
Uses subprocess to call R — more robust than rpy2 on Streamlit Cloud.
"""

import os
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd

_R_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "sem_analysis.R")


def check_r_available() -> dict:
    """Check if R is available on the system."""
    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return {"available": True, "message": "R is ready."}
        return {"available": False, "message": "R not found."}
    except Exception as e:
        return {"available": False, "message": str(e)}


def _run_r(r_code: str) -> dict:
    """
    Execute R code and return JSON result.
    Writes JSON to a temp file to avoid stdout contamination from R messages.
    """
    # Create temp files
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".R", delete=False, encoding="utf-8"
    ) as f:
        f.write(r_code)
        tmp_r = f.name

    tmp_json = tmp_r.replace(".R", "_out.json")

    # Inject JSON output file path into R code
    r_code_with_output = r_code + f'''
tryCatch({{
  # Write result to JSON file
  writeLines(
    jsonlite::toJSON(.SEM_RESULT, auto_unbox=TRUE, na="null"),
    "{tmp_json.replace(chr(92), "/")}"
  )
}}, error = function(e) {{
  writeLines(
    jsonlite::toJSON(list(error=conditionMessage(e)), auto_unbox=TRUE),
    "{tmp_json.replace(chr(92), "/")}"
  )
}})
'''

    # Rewrite R file with output injection
    with open(tmp_r, "w", encoding="utf-8") as f:
        f.write(r_code_with_output)

    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", tmp_r],
            capture_output=True, text=True, timeout=300
        )

        # Read from JSON file (not stdout)
        if os.path.exists(tmp_json):
            with open(tmp_json, "r", encoding="utf-8") as f:
                json_str = f.read().strip()
            if json_str:
                return json.loads(json_str)

        # Fallback: check stderr for error message
        if result.returncode != 0 and result.stderr:
            return {"error": result.stderr[-1000:]}

        return {"error": "R produced no output"}

    except subprocess.TimeoutExpired:
        return {"error": "R script timed out (>300s)"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        for f in [tmp_r, tmp_json]:
            try:
                os.unlink(f)
            except Exception:
                pass


def _df_to_r_csv(df: pd.DataFrame) -> str:
    """Write DataFrame to temp CSV and return path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def _escape(s: str) -> str:
    """Escape backslashes for R string."""
    return s.replace("\\", "/")


# ── Public API ────────────────────────────────────────────────────

def run_descriptives(df: pd.DataFrame, indicator_cols: list) -> dict:
    csv_path = _df_to_r_csv(df[indicator_cols])
    r_code = f"""
source("{_escape(_R_SCRIPT_PATH)}")
data <- read.csv("{_escape(csv_path)}")
result <- run_descriptives(data)
.SEM_RESULT <- result
"""
    res = _run_r(r_code)
    try: os.unlink(csv_path)
    except: pass
    return res


def run_mardia(df: pd.DataFrame, indicator_cols: list) -> dict:
    data = df[indicator_cols].dropna()
    csv_path = _df_to_r_csv(data)
    r_code = f"""
source("{_escape(_R_SCRIPT_PATH)}")
data <- read.csv("{_escape(csv_path)}")
result <- run_mardia(data)
.SEM_RESULT <- result
"""
    res = _run_r(r_code)
    try: os.unlink(csv_path)
    except: pass
    return res


def run_harman(df: pd.DataFrame, indicator_cols: list) -> dict:
    data = df[indicator_cols].dropna()
    csv_path = _df_to_r_csv(data)
    r_code = f"""
source("{_escape(_R_SCRIPT_PATH)}")
data <- read.csv("{_escape(csv_path)}")
result <- run_harman(data)
.SEM_RESULT <- result
"""
    res = _run_r(r_code)
    try: os.unlink(csv_path)
    except: pass
    return res


def run_efa(df: pd.DataFrame, indicator_cols: list,
            n_factors: int, rotation: str = "oblimin") -> dict:
    data = df[indicator_cols].dropna()
    csv_path = _df_to_r_csv(data)
    r_code = f"""
source("{_escape(_R_SCRIPT_PATH)}")
data <- read.csv("{_escape(csv_path)}")
result <- run_efa(data, {n_factors}, "{rotation}")
.SEM_RESULT <- result
"""
    res = _run_r(r_code)
    try: os.unlink(csv_path)
    except: pass
    return res


def run_cfa(df: pd.DataFrame, indicator_cols: list,
            model_syntax: str, estimator: str = "MLR") -> dict:
    data = df[indicator_cols].dropna()
    csv_path = _df_to_r_csv(data)
    # Escape syntax for R string
    syntax_escaped = model_syntax.replace('"', '\\"').replace("\n", "\\n")
    r_code = f"""
source("{_escape(_R_SCRIPT_PATH)}")
data <- read.csv("{_escape(csv_path)}")
syntax <- "{syntax_escaped}"
result <- run_cfa(data, syntax, "{estimator}")
.SEM_RESULT <- result
"""
    res = _run_r(r_code)
    try: os.unlink(csv_path)
    except: pass
    return res


def run_sem(df: pd.DataFrame, all_cols: list,
            model_syntax: str, estimator: str = "MLR") -> dict:
    data = df[all_cols].dropna()
    csv_path = _df_to_r_csv(data)
    syntax_escaped = model_syntax.replace('"', '\\"').replace("\n", "\\n")
    r_code = f"""
source("{_escape(_R_SCRIPT_PATH)}")
data <- read.csv("{_escape(csv_path)}")
syntax <- "{syntax_escaped}"
result <- run_sem(data, syntax, "{estimator}")
.SEM_RESULT <- result
"""
    res = _run_r(r_code)
    try: os.unlink(csv_path)
    except: pass
    return res


def run_mediation(df: pd.DataFrame, all_cols: list,
                  x_var: str, m_var: str, y_var: str,
                  constructs: dict,
                  n_boot: int = 5000,
                  estimator: str = "MLR") -> dict:
    data = df[all_cols]
    csv_path = _df_to_r_csv(data)

    # Build R constructs list
    construct_r = "list(" + ", ".join(
        f'"{k}"=c({",".join(repr(v) for v in vals)})'
        for k, vals in constructs.items() if vals
    ) + ")"

    r_code = f"""
source("{_escape(_R_SCRIPT_PATH)}")
data <- read.csv("{_escape(csv_path)}")
constructs <- {construct_r}
result <- run_mediation(data, "{x_var}", "{m_var}", "{y_var}",
                        constructs, {n_boot}, "{estimator}")
.SEM_RESULT <- result
"""
    res = _run_r(r_code)
    try: os.unlink(csv_path)
    except: pass
    return res


def run_moderation(df: pd.DataFrame, all_cols: list,
                   x_var: str, w_var: str, y_var: str,
                   constructs: dict) -> dict:
    data = df[all_cols]
    csv_path = _df_to_r_csv(data)

    construct_r = "list(" + ", ".join(
        f'"{k}"=c({",".join(repr(v) for v in vals)})'
        for k, vals in constructs.items() if vals
    ) + ")"

    r_code = f"""
source("{_escape(_R_SCRIPT_PATH)}")
data <- read.csv("{_escape(csv_path)}")
constructs <- {construct_r}
result <- run_moderation(data, "{x_var}", "{w_var}", "{y_var}", constructs)
.SEM_RESULT <- result
"""
    res = _run_r(r_code)
    try: os.unlink(csv_path)
    except: pass
    return res


def run_invariance(df: pd.DataFrame, indicator_cols: list,
                   model_syntax: str, group_var: str,
                   estimator: str = "MLR") -> dict:
    cols = list(set(indicator_cols + [group_var]))
    data = df[[c for c in cols if c in df.columns]].dropna()
    csv_path = _df_to_r_csv(data)
    syntax_escaped = model_syntax.replace('"', '\\"').replace("\n", "\\n")
    r_code = f"""
source("{_escape(_R_SCRIPT_PATH)}")
data <- read.csv("{_escape(csv_path)}")
syntax <- "{syntax_escaped}"
result <- run_invariance(data, syntax, "{group_var}", "{estimator}")
.SEM_RESULT <- result
"""
    res = _run_r(r_code)
    try: os.unlink(csv_path)
    except: pass
    return res


def run_model_comparison(df: pd.DataFrame, all_cols: list,
                          models: dict,
                          estimator: str = "MLR") -> dict:
    data = df[all_cols].dropna()
    csv_path = _df_to_r_csv(data)

    # Write each model syntax to temp file and read in R
    import json as _json
    models_json_path = csv_path.replace(".csv", "_models.json")
    with open(models_json_path, "w", encoding="utf-8") as f:
        _json.dump(models, f)

    r_code = f"""
source("{_escape(_R_SCRIPT_PATH)}")
data <- read.csv("{_escape(csv_path)}")
models_raw <- jsonlite::fromJSON("{_escape(models_json_path)}")
models <- as.list(models_raw)
result <- run_model_comparison(data, models, "{estimator}")
.SEM_RESULT <- result
"""
    res = _run_r(r_code)
    for f in [csv_path, models_json_path]:
        try: os.unlink(f)
        except: pass
    return res


# =============================================================================
# PLS-SEM Bridge
# =============================================================================

_PLS_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "pls_analysis.R")


def run_plssem(df: pd.DataFrame,
               constructs: dict,
               paths: list,
               construct_types: dict = None,
               n_boot: int = 1000) -> dict:
    """
    Run PLS-SEM via seminr package.
    
    Parameters
    ----------
    df : DataFrame
    constructs : dict  {name: [items]}
    paths : list       [(from, to), ...]
    construct_types : dict  {name: "reflective"|"formative"}
    n_boot : int       Bootstrap resamples
    """
    all_items = [item for items in constructs.values() for item in items]
    data = df[[c for c in all_items if c in df.columns]]
    csv_path = _df_to_r_csv(data)

    # Build R construct list
    construct_r = "list(" + ", ".join(
        f'"{k}"=c({",".join(repr(v) for v in vals)})'
        for k, vals in constructs.items() if vals
    ) + ")"

    # Build R paths list
    paths_r = "list(" + ", ".join(
        f'c("{p[0]}", "{p[1]}")'
        for p in paths
    ) + ")"

    # Build R construct_types list
    if construct_types:
        ct_r = "list(" + ", ".join(
            f'"{k}"="{v}"' for k, v in construct_types.items()
        ) + ")"
    else:
        ct_r = "NULL"

    r_code = f"""
source("{_escape(_PLS_SCRIPT_PATH)}")
data       <- read.csv("{_escape(csv_path)}")
constructs <- {construct_r}
paths      <- {paths_r}
ctypes     <- {ct_r}
result     <- run_plssem(data, constructs, paths,
                         construct_types = ctypes,
                         n_boot = {n_boot}, seed = 42)
.SEM_RESULT <- result
"""
    res = _run_r(r_code)
    try:
        os.unlink(csv_path)
    except Exception:
        pass
    return res
