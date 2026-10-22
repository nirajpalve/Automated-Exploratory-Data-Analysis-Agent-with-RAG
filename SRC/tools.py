import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # save plots to files, never open windows
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PLOTS = Path("outputs/plots")


# ---------- small helpers ----------

def _r(x, nd=3):
    """Round to a plain float (JSON-safe); NaN becomes None."""
    return None if pd.isna(x) else round(float(x), nd)

def _safe(name):
    """Make a column name safe to use in a filename."""
    return re.sub(r"[^\w-]", "_", str(name))

def _save(name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return str(path)

def _numeric_cols(df):
    """Numeric columns, skipping ID-like ones (an integer that is unique on every row)."""
    cols = []
    for c in df.select_dtypes("number").columns:
        if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() == len(df):
            continue
        cols.append(c)
    return cols


# ---------- the tools ----------

def profile(df):
    """Basic shape, column types, duplicates and likely ID columns."""
    numeric = _numeric_cols(df)
    all_numeric = df.select_dtypes("number").columns.tolist()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "numeric_columns": numeric,
        "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
        "id_like_columns": [c for c in all_numeric if c not in numeric],
        "duplicate_rows": int(df.duplicated().sum()),
    }


def missing_values(df):
    """Percent missing per column, highest first, plus a bar chart."""
    pct = (df.isna().mean() * 100).round(2)
    pct = pct[pct > 0].sort_values(ascending=False)
    if pct.empty:
        return {"columns_with_missing": {}, "plot": None}
    plt.figure(figsize=(7, max(2, len(pct) * 0.4)))
    pct.sort_values().plot.barh(color="#d9534f")
    plt.xlabel("% missing")
    plt.title("Missing values by column")
    return {"columns_with_missing": {c: float(v) for c, v in pct.items()},
            "plot": _save("missing_values")}


def distributions(df):
    """Summary statistics and a histogram for each numeric column."""
    out = {}
    for c in _numeric_cols(df)[:20]:          # cap at 20 columns to keep it manageable
        s = df[c].dropna()
        if s.empty:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(s, kde=True)
        plt.title(f"Distribution of {c}")
        out[c] = {
            "mean": _r(s.mean()), "median": _r(s.median()), "std": _r(s.std()),
            "min": _r(s.min()), "max": _r(s.max()), "skew": _r(s.skew()),
            "plot": _save(f"dist_{_safe(c)}"),
        }
    return out


def outliers(df, method="iqr"):
    """Count outliers per numeric column using IQR (default) or z-score."""
    out = {}
    for c in _numeric_cols(df):
        s = df[c].dropna()
        if len(s) < 5:
            continue
        if method == "zscore":
            mask = ((s - s.mean()) / s.std()).abs() > 3
        else:
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (s < low) | (s > high)
        out[c] = {"method": method, "count": int(mask.sum()), "percent": _r(100 * mask.mean(), 2),
                  "example_values": [_r(v) for v in s[mask].head(5)]}
    return out


def categorical_summary(df, top_n=5):
    """Unique counts and most common values for each non-numeric column."""
    out = {}
    for c in df.select_dtypes(exclude="number").columns:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        top = (s.value_counts(normalize=True).head(top_n) * 100).round(2)
        lowered = s.str.strip().str.lower()
        out[c] = {
            "unique_values": int(s.nunique()),
            "high_cardinality": bool(s.nunique() > 50 or s.nunique() > 0.5 * len(s)),
            "top_values_percent": {k: float(v) for k, v in top.items()},
            "possible_inconsistent_spelling": bool(lowered.nunique() < s.nunique()),
        }
    return out


def correlations(df, method="spearman", threshold=0.7):
    """Correlation heatmap and the strongly correlated pairs."""
    cols = _numeric_cols(df)
    if len(cols) < 2:
        return {"note": "fewer than 2 numeric columns, correlation skipped"}
    corr = df[cols].corr(method=method)
    plt.figure(figsize=(max(5, len(cols) * 0.8), max(4, len(cols) * 0.6)))
    sns.heatmap(corr, annot=len(cols) <= 12, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(f"{method.capitalize()} correlation")
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    strong = {f"{a} ~ {b}": _r(v) for (a, b), v in upper.items() if abs(v) >= threshold}
    return {"method": method, "threshold": threshold,
            "strong_pairs": strong, "plot": _save("correlation_heatmap")}


# The agent may only call tools listed here
TOOLS = {
    "missing_values": missing_values,
    "distributions": distributions,
    "outliers": outliers,
    "categorical_summary": categorical_summary,
    "correlations": correlations,
}

TOOL_DESCRIPTIONS = {
    "missing_values": "Percent missing per column. No args.",
    "distributions": "Mean, median, std, min, max, skew and histogram per numeric column. No args.",
    "outliers": "Outlier counts per numeric column. Args: method = 'iqr' (default) or 'zscore'.",
    "categorical_summary": "Unique counts, top values, cardinality and spelling issues. Args: top_n (int).",
    "correlations": "Correlation heatmap and strong pairs. Args: method = 'spearman' or 'pearson', threshold (float).",
}


# ---------- test with a dataset that has known problems ----------

if __name__ == "__main__":
    import json

    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),                                  # ID column
        "age": rng.normal(40, 10, n).round(),
        "income": rng.lognormal(10, 1, n).round(2),                           # heavily skewed
        "city": rng.choice(["Dublin", "Cork", "Galway", "dublin"], n, p=[.5, .25, .2, .05]),  # spelling issue
    })
    df["spend"] = (df["income"] * 0.3 + rng.normal(0, 500, n)).round(2)     # correlated with income
    df.loc[rng.choice(n, 300, replace=False), "age"] = np.nan                # 30% missing
    df.loc[:9, "age"] = 250                                                  # impossible ages
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)                    # 5 duplicate rows

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/synthetic_test.csv", index=False)
    print("Saved data/synthetic_test.csv\n")

    print("PROFILE:", json.dumps(profile(df), indent=2))
    for name, fn in TOOLS.items():
        print(f"\n{name.upper()}:")
        print(json.dumps(fn(df), indent=2))