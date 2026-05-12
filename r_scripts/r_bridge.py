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
    Wraps result in tryCatch and outputs as JSON.
    """
    # Write R code to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".R", delete=False, encoding="utf-8"
    ) as f:
        f.write(r_code)
        tmp_r = f.name

    try:
        result = subprocess.run(
            ["Rscript", tmp_r],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            return {"error": result.stderr[-2000:] if result.stderr else "R script failed"}

        stdout = result.stdout.strip()
        if not stdout:
            return {"error": "R returned no output"}

        # Find JSON in output
        start = stdout.rfind("{")
        end   = stdout.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = stdout[start:end]
            return json.loads(json_str)
        return {"error": f"Could not parse R output: {stdout[:500]}"}

    except subprocess.TimeoutExpired:
        return {"error": "R script timed out (>300s)"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(tmp_r)
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE, na="null"))
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE, na="null"))
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE, na="null"))
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE, na="null"))
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE, na="null"))
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE, na="null"))
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE, na="null"))
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE, na="null"))
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE, na="null"))
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

    models_r = "list(" + ", ".join(
        f'"{k}"="{v.replace(chr(34), chr(39)).replace(chr(10), " ")}"'
        for k, v in models.items()
    ) + ")"

    r_code = f"""
source("{_escape(_R_SCRIPT_PATH)}")
data <- read.csv("{_escape(csv_path)}")
models <- {models_r}
result <- run_model_comparison(data, models, "{estimator}")
cat(jsonlite::toJSON(result, auto_unbox=TRUE, na="null"))
"""
    res = _run_r(r_code)
    try: os.unlink(csv_path)
    except: pass
    return res
