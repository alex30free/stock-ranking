"""
process.py
----------
Reads a Börsdata CSV export, computes Value / Quality / Momentum / QVM
percentile scores, and writes scores.json to data/scores.json.

Usage:
    python process.py --input data/börsdata_export.csv

Column mapping (edit COLUMN_MAP to match your actual Börsdata column names):
"""

import argparse
import json
import math
import os
from datetime import datetime

import pandas as pd
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# COLUMN MAP — edit these keys to match your exact Börsdata column headers
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    # identity
    "ticker":          "Ticker",
    "name":            "Bolagsnamn",        # or "Company Name"
    "sector":          "Sektor",
    "market_cap":      "Börsvärde",         # market cap in MSEK

    # value
    "pe":              "P/E",
    "pb":              "P/B",
    "ev_ebit":         "EV/EBIT",
    "ev_sales":        "EV/Omsättning",     # EV/Sales
    "p_fcf":           "P/FCF",
    "div_yield":       "Direktavkastning",  # %

    # quality
    "roe":             "ROE",               # %
    "roic":            "ROIC",              # %
    "gross_margin":    "Bruttomarginal",    # %
    "op_margin":       "Rörelsemarginal",   # %
    "net_margin":      "Nettomarginal",     # %
    "current_ratio":   "Kassalikviditet",
    "nd_ebitda":       "Nettoskuld/EBITDA",
    "rev_growth_3y":   "Omsättningstillväxt 3 år", # % CAGR

    # momentum (prefer pre-computed returns; fall back to raw prices)
    "ret_12m":         "Avkastning 12M",    # % — if available
    "ret_6m":          "Avkastning 6M",     # % — if available
    "ret_3m":          "Avkastning 3M",     # % — if available
    # fallback raw prices (uncomment if your export has prices instead):
    # "price_now":     "Kurs",
    # "price_3m":      "Kurs 3M",
    # "price_6m":      "Kurs 6M",
    # "price_12m":     "Kurs 12M",
}

# ---------------------------------------------------------------------------
# Weights  (must sum to 1.0 within each group)
# ---------------------------------------------------------------------------
VALUE_WEIGHTS = {
    "inv_pe":       0.20,
    "inv_pb":       0.20,
    "inv_ev_ebit":  0.25,
    "inv_p_fcf":    0.20,
    "div_yield":    0.15,
}

QUALITY_WEIGHTS = {
    "roe":          0.15,
    "roic":         0.15,
    "gross_margin": 0.10,
    "op_margin":    0.15,
    "net_margin":   0.10,
    "current_ratio":0.10,
    "inv_nd_ebitda":0.15,
    "rev_growth_3y":0.10,
}

MOMENTUM_WEIGHTS = {
    "ret_12m": 0.50,
    "ret_6m":  0.30,
    "ret_3m":  0.20,
}

QVM_WEIGHTS = {
    "value":    1/3,
    "quality":  1/3,
    "momentum": 1/3,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct_rank(series: pd.Series) -> pd.Series:
    """
    Returns a 0–100 percentile rank. NaN → NaN.
    Higher rank = better (already inverted for bad-is-high metrics upstream).
    Uses average method for ties.
    """
    not_null = series.notna()
    ranks = series.copy() * float("nan")
    if not_null.sum() == 0:
        return ranks
    vals = series[not_null].values
    r = rankdata(vals, method="average")
    # normalise to 0–100
    ranks[not_null] = (r - 1) / (len(r) - 1) * 100 if len(r) > 1 else 50.0
    return ranks


def winsorize(series: pd.Series, low=0.02, high=0.98) -> pd.Series:
    """Clip at 2nd and 98th percentile to reduce outlier influence."""
    lo = series.quantile(low)
    hi = series.quantile(high)
    return series.clip(lo, hi)


def safe_invert(series: pd.Series) -> pd.Series:
    """Return 1/x, treating zeros and negatives as NaN."""
    s = series.copy().astype(float)
    s[s <= 0] = float("nan")
    return 1.0 / s


def weighted_score(df: pd.DataFrame, weight_dict: dict) -> pd.Series:
    """
    Compute a weighted average of percentile-ranked columns.
    Missing individual factors reduce the effective weight proportionally.
    """
    total_w = pd.Series(0.0, index=df.index)
    total_score = pd.Series(0.0, index=df.index)
    for col, w in weight_dict.items():
        if col not in df.columns:
            continue
        col_rank = pct_rank(df[col])
        mask = col_rank.notna()
        total_score[mask] += col_rank[mask] * w
        total_w[mask] += w
    # normalise by actual weight used
    result = total_score / total_w.replace(0, float("nan"))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_scores(csv_path: str) -> list[dict]:
    cm = COLUMN_MAP

    # --- 1. Load -------------------------------------------------------
    df = pd.read_csv(csv_path, sep=None, engine="python", decimal=",",
                     thousands=" ")
    # Rename to internal names
    reverse_map = {v: k for k, v in cm.items()}
    df = df.rename(columns=reverse_map)

    # Keep only rows with a ticker
    df = df[df["ticker"].notna()].copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["name"]   = df.get("name", df["ticker"])

    n = len(df)
    print(f"  Loaded {n} stocks from {csv_path}")

    # --- 2. Compute raw momentum if prices given instead of returns -----
    for ret_col, price_now, price_ago in [
        ("ret_12m", "price_now", "price_12m"),
        ("ret_6m",  "price_now", "price_6m"),
        ("ret_3m",  "price_now", "price_3m"),
    ]:
        if ret_col not in df.columns and price_now in df.columns and price_ago in df.columns:
            p0 = pd.to_numeric(df[price_ago], errors="coerce")
            p1 = pd.to_numeric(df[price_now], errors="coerce")
            df[ret_col] = (p1 / p0 - 1) * 100

    # --- 3. Winsorize continuous numerics ------------------------------
    numeric_cols = [
        "pe", "pb", "ev_ebit", "ev_sales", "p_fcf", "div_yield",
        "roe", "roic", "gross_margin", "op_margin", "net_margin",
        "current_ratio", "nd_ebitda", "rev_growth_3y",
        "ret_12m", "ret_6m", "ret_3m",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = winsorize(df[col])

    # --- 4. Invert bad-is-high metrics ----------------------------------
    for src, dst in [
        ("pe",       "inv_pe"),
        ("pb",       "inv_pb"),
        ("ev_ebit",  "inv_ev_ebit"),
        ("p_fcf",    "inv_p_fcf"),
        ("nd_ebitda","inv_nd_ebitda"),
    ]:
        if src in df.columns:
            df[dst] = safe_invert(df[src])

    # --- 5. Compute pillar scores --------------------------------------
    df["value_score"]    = weighted_score(df, VALUE_WEIGHTS)
    df["quality_score"]  = weighted_score(df, QUALITY_WEIGHTS)
    df["momentum_score"] = weighted_score(df, MOMENTUM_WEIGHTS)

    # Re-rank pillar scores themselves to 0–100 percentiles
    df["value_rank"]    = pct_rank(df["value_score"]).round(1)
    df["quality_rank"]  = pct_rank(df["quality_score"]).round(1)
    df["momentum_rank"] = pct_rank(df["momentum_score"]).round(1)

    # --- 6. QVM composite rank ----------------------------------------
    qvm_raw = (
        df["value_rank"].fillna(50)    * QVM_WEIGHTS["value"]  +
        df["quality_rank"].fillna(50)  * QVM_WEIGHTS["quality"] +
        df["momentum_rank"].fillna(50) * QVM_WEIGHTS["momentum"]
    )
    df["qvm_rank"] = pct_rank(qvm_raw).round(1)

    # --- 7. Build output list -----------------------------------------
    out = []
    for _, row in df.iterrows():
        def fmt(v):
            if v is None:
                return None
            try:
                if math.isnan(float(v)):
                    return None
            except Exception:
                return str(v)
            return round(float(v), 2)

        out.append({
            "ticker":     str(row.get("ticker", "")),
            "name":       str(row.get("name", "")),
            "sector":     str(row.get("sector", "")) if "sector" in row else "",
            "market_cap": fmt(row.get("market_cap")),
            # display fundamentals
            "pe":         fmt(row.get("pe")),
            "pb":         fmt(row.get("pb")),
            "ev_ebit":    fmt(row.get("ev_ebit")),
            "div_yield":  fmt(row.get("div_yield")),
            "roe":        fmt(row.get("roe")),
            "op_margin":  fmt(row.get("op_margin")),
            "ret_12m":    fmt(row.get("ret_12m")),
            # scores
            "value_rank":    fmt(row.get("value_rank")),
            "quality_rank":  fmt(row.get("quality_rank")),
            "momentum_rank": fmt(row.get("momentum_rank")),
            "qvm_rank":      fmt(row.get("qvm_rank")),
        })

    # Sort by qvm_rank descending
    out.sort(key=lambda x: x["qvm_rank"] or 0, reverse=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Compute ranking scores from Börsdata CSV")
    parser.add_argument("--input",  default="data/börsdata_export.csv")
    parser.add_argument("--output", default="data/scores.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    stocks = build_scores(args.input)

    payload = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "count":   len(stocks),
        "stocks":  stocks,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"  Wrote {len(stocks)} rows → {args.output}")
    print(f"  Updated: {payload['updated']}")


if __name__ == "__main__":
    main()
