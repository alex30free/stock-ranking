"""
process.py — QVM Stock Ranking Engine
--------------------------------------
Reads a Börsdata CSV export, computes Value / Quality / Momentum / QVM
percentile scores, and writes data/scores.json.

Usage:
    python process.py --input data/Borsdata_export.csv

Configured for Börsdata export format:
  - Delimiter        : semicolon (;)
  - Decimal separator: comma (,)
  - Percentages      : include % sign e.g. "19,9%" — stripped automatically
  - Encoding         : UTF-8 with BOM
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# COLUMN MAP
# Left  = internal name used by this script  (do NOT change)
# Right = exact column header in your Börsdata CSV (change if yours differ)
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "ticker":        "Info - Ticker",
    "name":          "Bolagsnamn",
    "sector":        "Info - Sektor",
    "market_cap":    "Börsvärde - Senaste SEK",
    "pe":            "P/E - Senaste",
    "pb":            "P/B - Senaste",
    "ev_ebit":       "EV/EBIT - Senaste",
    "ev_sales":      "EV/S - Senaste",
    "p_fcf":         "P/FCF - Senaste",
    "roe":           "ROE - Senaste",
    "roic":          "ROIC - Senaste",
    "gross_margin":  "Bruttomarg - Senaste",
    "op_margin":     "EBIT-marg - Senaste",
    "net_margin":    "Vinstmarg - Senaste",
    "current_ratio": "Balanslik. - Senaste",
    "nd_ebitda":     "N.skuld/Ebitda - Senaste",
    "rev_growth_3y": "Omsätt. tillv. - År. tillv. 3år",
    "ret_3m":        "Kursutveck. - Utveck.  3m",
    "ret_6m":        "Kursutveck. - Utveck.  6m",
    "ret_12m":       "Kursutveck. - Utveck.  1år",
}

# ---------------------------------------------------------------------------
# Factor weights (each group must sum to 1.0)
# ---------------------------------------------------------------------------
VALUE_WEIGHTS = {
    "inv_pe":       0.20,
    "inv_pb":       0.15,
    "inv_ev_ebit":  0.30,
    "inv_p_fcf":    0.25,
    "inv_ev_sales": 0.10,
}

QUALITY_WEIGHTS = {
    "roe":           0.15,
    "roic":          0.15,
    "gross_margin":  0.10,
    "op_margin":     0.15,
    "net_margin":    0.10,
    "current_ratio": 0.10,
    "inv_nd_ebitda": 0.15,
    "rev_growth_3y": 0.10,
}

MOMENTUM_WEIGHTS = {
    "ret_12m": 0.50,
    "ret_6m":  0.30,
    "ret_3m":  0.20,
}

QVM_WEIGHTS = {
    "value":    1 / 3,
    "quality":  1 / 3,
    "momentum": 1 / 3,
}

# ---------------------------------------------------------------------------
# Minimum data threshold — stocks below this are excluded from rankings
# Must have at least 3 raw data points AND at least 2 of 3 pillar scores
# This removes index ETFs, shells, and newly listed stocks with no data
# ---------------------------------------------------------------------------
MIN_RAW_FIELDS   = 3   # out of: pe, pb, ev_ebit, p_fcf, roe, roic,
                       #         gross_margin, op_margin, net_margin,
                       #         ret_12m, ret_6m, ret_3m
MIN_PILLAR_SCORES = 2  # out of: value_rank, quality_rank, momentum_rank

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_numeric(series):
    """Strip %, swap comma decimals, coerce to float."""
    s = series.astype(str).str.strip()
    s = s.str.replace('%', '', regex=False)
    s = s.str.replace(',', '.', regex=False)
    s = s.str.replace('\xa0', '', regex=False)
    s = s.str.replace(' ', '', regex=False)
    return pd.to_numeric(s, errors='coerce')


def pct_rank(series):
    """0–100 percentile rank. NaN stays NaN. Higher = better."""
    mask = series.notna()
    result = series * float('nan')
    if mask.sum() < 2:
        return result
    vals = series[mask].values
    r = rankdata(vals, method='average')
    result[mask] = (r - 1) / (len(r) - 1) * 100
    return result


def winsorize(series, low=0.02, high=0.98):
    lo, hi = series.quantile(low), series.quantile(high)
    return series.clip(lo, hi)


def safe_invert(series):
    """1/x — negatives and zeros become NaN."""
    s = series.astype(float).copy()
    s[s <= 0] = float('nan')
    return 1.0 / s


def weighted_score(df, weight_dict):
    """Weighted average of percentile-ranked columns. Missing cols skipped."""
    total_w     = pd.Series(0.0, index=df.index)
    total_score = pd.Series(0.0, index=df.index)
    for col, w in weight_dict.items():
        if col not in df.columns:
            continue
        ranked = pct_rank(df[col])
        ok = ranked.notna()
        total_score[ok] += ranked[ok] * w
        total_w[ok]     += w
    return total_score / total_w.replace(0, float('nan'))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_scores(csv_path):
    # 1. Load
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig', dtype=str)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # 2. Rename columns
    reverse_map = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns=reverse_map)

    if 'ticker' not in df.columns:
        raise ValueError(
            "Column 'Info - Ticker' not found. "
            "Update COLUMN_MAP to match your CSV headers."
        )

    df['ticker'] = df['ticker'].astype(str).str.strip()
    df = df[df['ticker'].notna() & (df['ticker'] != '') & (df['ticker'] != 'nan')].copy()
    print(f"  After ticker filter: {len(df):,} rows")

    # 3. Parse numerics
    numeric_cols = [
        'market_cap',
        'pe', 'pb', 'ev_ebit', 'ev_sales', 'p_fcf',
        'roe', 'roic', 'gross_margin', 'op_margin', 'net_margin',
        'current_ratio', 'nd_ebitda', 'rev_growth_3y',
        'ret_3m', 'ret_6m', 'ret_12m',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # 4. Winsorize
    for col in numeric_cols:
        if col in df.columns and df[col].notna().sum() > 20:
            df[col] = winsorize(df[col])

    # 5. Invert bad-is-high metrics
    for src, dst in [
        ('pe',       'inv_pe'),
        ('pb',       'inv_pb'),
        ('ev_ebit',  'inv_ev_ebit'),
        ('ev_sales', 'inv_ev_sales'),
        ('p_fcf',    'inv_p_fcf'),
        ('nd_ebitda','inv_nd_ebitda'),
    ]:
        if src in df.columns:
            df[dst] = safe_invert(df[src])

    # 6. Pillar scores
    df['value_score']    = weighted_score(df, VALUE_WEIGHTS)
    df['quality_score']  = weighted_score(df, QUALITY_WEIGHTS)
    df['momentum_score'] = weighted_score(df, MOMENTUM_WEIGHTS)

    df['value_rank']    = pct_rank(df['value_score']).round(1)
    df['quality_rank']  = pct_rank(df['quality_score']).round(1)
    df['momentum_rank'] = pct_rank(df['momentum_score']).round(1)

    # 7. QVM composite
    qvm_raw = (
        df['value_rank'].fillna(50)    * QVM_WEIGHTS['value']    +
        df['quality_rank'].fillna(50)  * QVM_WEIGHTS['quality']  +
        df['momentum_rank'].fillna(50) * QVM_WEIGHTS['momentum']
    )
    df['qvm_rank'] = pct_rank(qvm_raw).round(1)

    # 8. ── FILTER OUT STOCKS WITH INSUFFICIENT DATA ──────────────────────
    # A stock must have at least MIN_RAW_FIELDS raw data points
    # AND at least MIN_PILLAR_SCORES computed pillar scores.
    # This removes: index ETFs, warrants, shells, newly-listed stubs.
    raw_check_cols = [
        'pe', 'pb', 'ev_ebit', 'p_fcf',
        'roe', 'roic', 'gross_margin', 'op_margin', 'net_margin',
        'ret_12m', 'ret_6m', 'ret_3m',
    ]
    raw_check_cols = [c for c in raw_check_cols if c in df.columns]

    df['_raw_filled']    = df[raw_check_cols].notna().sum(axis=1)
    df['_pillar_filled'] = df[['value_rank','quality_rank','momentum_rank']].notna().sum(axis=1)

    before = len(df)
    df = df[
        (df['_raw_filled']    >= MIN_RAW_FIELDS) &
        (df['_pillar_filled'] >= MIN_PILLAR_SCORES)
    ].copy()
    removed = before - len(df)
    print(f"  Removed {removed} stocks with insufficient data (ETFs/shells/stubs)")
    print(f"  Remaining: {len(df):,} scoreable stocks")

    # 9. Serialise
    def fmt(v):
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 2)
        except Exception:
            s = str(v)
            return None if s in ('nan', 'None', '') else s

    out = []
    for _, row in df.iterrows():
        out.append({
            'ticker':        str(row.get('ticker') or '').strip(),
            'name':          str(row.get('name')   or '').strip(),
            'sector':        fmt(row.get('sector'))  or '',
            'market_cap':    fmt(row.get('market_cap')),
            'pe':            fmt(row.get('pe')),
            'pb':            fmt(row.get('pb')),
            'ev_ebit':       fmt(row.get('ev_ebit')),
            'p_fcf':         fmt(row.get('p_fcf')),
            'roe':           fmt(row.get('roe')),
            'op_margin':     fmt(row.get('op_margin')),
            'ret_12m':       fmt(row.get('ret_12m')),
            'ret_6m':        fmt(row.get('ret_6m')),
            'ret_3m':        fmt(row.get('ret_3m')),
            'value_rank':    fmt(row.get('value_rank')),
            'quality_rank':  fmt(row.get('quality_rank')),
            'momentum_rank': fmt(row.get('momentum_rank')),
            'qvm_rank':      fmt(row.get('qvm_rank')),
        })

    out.sort(key=lambda x: x['qvm_rank'] or 0, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='QVM scoring from Börsdata CSV')
    parser.add_argument('--input',  default='data/Borsdata_export.csv')
    parser.add_argument('--output', default='data/scores.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Reading: {args.input}")
    stocks = build_scores(args.input)

    payload = {
        'updated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'count':   len(stocks),
        'stocks':  stocks,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"  Written {len(stocks):,} stocks → {args.output}")
    print(f"  Top 5 by StockRank:")
    for s in stocks[:5]:
        print(f"    {s['ticker']:12} QVM={s['qvm_rank']}  V={s['value_rank']}  Q={s['quality_rank']}  M={s['momentum_rank']}")
    print(f"  Bottom 3:")
    for s in stocks[-3:]:
        print(f"    {s['ticker']:12} QVM={s['qvm_rank']}  V={s['value_rank']}  Q={s['quality_rank']}  M={s['momentum_rank']}")


if __name__ == '__main__':
    main()
