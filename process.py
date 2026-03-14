"""
process.py — QVM Stock Ranking Engine
--------------------------------------
Reads a Börsdata CSV export, computes Value / Quality / Momentum / QVM
percentile scores, assigns Stockopedia-style archetypes, and writes
data/scores.json.

Usage:
    python process.py --input data/Borsdata_export.csv
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
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "ticker":        "Info - Ticker",
    "name":          "Bolagsnamn",
    "sector":        "Info - Sektor",
    "country":       "Info - Land",
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
# Weights
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
QVM_WEIGHTS = {"value": 1/3, "quality": 1/3, "momentum": 1/3}

# Minimum data threshold — removes ETFs, warrants, shells
MIN_RAW_FIELDS    = 3
MIN_PILLAR_SCORES = 2

# ---------------------------------------------------------------------------
# Stockopedia 8-archetype style classification
# Exactly mirrors the screenshot: 8 unique Q/V/M combinations
# ---------------------------------------------------------------------------
STYLE_MAP = {
    (True,  True,  True):  "Super Stock",    # High Q, High V, High M
    (True,  False, True):  "High Flyer",     # High Q, Low V,  High M
    (True,  True,  False): "Contrarian",     # High Q, High V, Low M
    (False, True,  True):  "Turnaround",     # Low Q,  High V, High M
    (False, False, False): "Sucker Stock",   # Low Q,  Low V,  Low M
    (False, False, True):  "Momentum Trap",  # Low Q,  Low V,  High M
    (True,  False, False): "Falling Star",   # High Q, Low V,  Low M
    (False, True,  False): "Value Trap",     # Low Q,  High V, Low M
}

def classify_style(q, v, m):
    """Return Stockopedia archetype string. Returns None if any score missing."""
    if q is None or v is None or m is None:
        return None
    return STYLE_MAP.get((q >= 50, v >= 50, m >= 50))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_numeric(series):
    s = series.astype(str).str.strip()
    s = s.str.replace('%', '', regex=False)
    s = s.str.replace(',', '.', regex=False)
    s = s.str.replace('\xa0', '', regex=False)
    s = s.str.replace(' ', '', regex=False)
    return pd.to_numeric(s, errors='coerce')

def pct_rank(series):
    mask = series.notna()
    result = series * float('nan')
    if mask.sum() < 2:
        return result
    vals = series[mask].values
    r = rankdata(vals, method='average')
    result[mask] = (r - 1) / (len(r) - 1) * 100
    return result

def unique_int_rank(series):
    """
    Convert a float percentile series to unique integer ranks 1..N.
    Rank 1 = highest score (best). No ties — random tiebreak via stable sort.
    """
    mask = series.notna()
    result = pd.Series([None] * len(series), dtype=object)
    if mask.sum() == 0:
        return result
    # argsort descending — stable so ties broken by original order
    ordered = series[mask].sort_values(ascending=False, kind='mergesort')
    for rank_pos, idx in enumerate(ordered.index, start=1):
        result[idx] = rank_pos
    return result

def winsorize(series, low=0.02, high=0.98):
    lo, hi = series.quantile(low), series.quantile(high)
    return series.clip(lo, hi)

def safe_invert(series):
    s = series.astype(float).copy()
    s[s <= 0] = float('nan')
    return 1.0 / s

def weighted_score(df, weight_dict):
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
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig', dtype=str)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    reverse_map = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns=reverse_map)

    if 'ticker' not in df.columns:
        raise ValueError("Column 'Info - Ticker' not found. Check COLUMN_MAP.")

    df['ticker'] = df['ticker'].astype(str).str.strip()
    df = df[df['ticker'].notna() & (df['ticker'] != '') & (df['ticker'] != 'nan')].copy()
    print(f"  After ticker filter: {len(df):,} rows")

    numeric_cols = [
        'market_cap', 'pe', 'pb', 'ev_ebit', 'ev_sales', 'p_fcf',
        'roe', 'roic', 'gross_margin', 'op_margin', 'net_margin',
        'current_ratio', 'nd_ebitda', 'rev_growth_3y',
        'ret_3m', 'ret_6m', 'ret_12m',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    for col in numeric_cols:
        if col in df.columns and df[col].notna().sum() > 20:
            df[col] = winsorize(df[col])

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

    # Pillar scores (0-100 percentile floats)
    df['value_score']    = weighted_score(df, VALUE_WEIGHTS)
    df['quality_score']  = weighted_score(df, QUALITY_WEIGHTS)
    df['momentum_score'] = weighted_score(df, MOMENTUM_WEIGHTS)

    df['value_rank']    = pct_rank(df['value_score']).round(1)
    df['quality_rank']  = pct_rank(df['quality_score']).round(1)
    df['momentum_rank'] = pct_rank(df['momentum_score']).round(1)

    # QVM composite (0-100 percentile float)
    qvm_raw = (
        df['value_rank'].fillna(50)    * QVM_WEIGHTS['value']    +
        df['quality_rank'].fillna(50)  * QVM_WEIGHTS['quality']  +
        df['momentum_rank'].fillna(50) * QVM_WEIGHTS['momentum']
    )
    df['qvm_score'] = pct_rank(qvm_raw).round(1)

    # ── Filter: remove ETFs, shells, stubs ─────────────────────────────────
    raw_check = [c for c in [
        'pe','pb','ev_ebit','p_fcf','roe','roic',
        'gross_margin','op_margin','net_margin',
        'ret_12m','ret_6m','ret_3m',
    ] if c in df.columns]

    df['_raw_filled']    = df[raw_check].notna().sum(axis=1)
    df['_pillar_filled'] = df[['value_rank','quality_rank','momentum_rank']].notna().sum(axis=1)
    before = len(df)
    df = df[
        (df['_raw_filled']    >= MIN_RAW_FIELDS) &
        (df['_pillar_filled'] >= MIN_PILLAR_SCORES)
    ].copy()
    print(f"  Removed {before-len(df)} ETFs/shells/stubs → {len(df):,} stocks remain")

    # ── Unique integer ranks 1..N (no ties) ────────────────────────────────
    # Re-rank within the filtered universe so rank 1 = best stock
    df['value_rank']    = pct_rank(df['value_score']).round(1)
    df['quality_rank']  = pct_rank(df['quality_score']).round(1)
    df['momentum_rank'] = pct_rank(df['momentum_score']).round(1)
    qvm_raw2 = (
        df['value_rank'].fillna(50)    * QVM_WEIGHTS['value']    +
        df['quality_rank'].fillna(50)  * QVM_WEIGHTS['quality']  +
        df['momentum_rank'].fillna(50) * QVM_WEIGHTS['momentum']
    )
    df['qvm_score'] = pct_rank(qvm_raw2).round(1)

    # Unique integer ranks: 1 = best, N = worst
    df['value_int']    = unique_int_rank(df['value_rank'])
    df['quality_int']  = unique_int_rank(df['quality_rank'])
    df['momentum_int'] = unique_int_rank(df['momentum_rank'])
    df['qvm_int']      = unique_int_rank(df['qvm_score'])

    # ── Style classification ────────────────────────────────────────────────
    df['style'] = df.apply(
        lambda r: classify_style(
            r.get('quality_rank'), r.get('value_rank'), r.get('momentum_rank')
        ), axis=1
    )

    # ── Serialise ───────────────────────────────────────────────────────────
    def fmt(v):
        if v is None: return None
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 2)
        except Exception:
            s = str(v)
            return None if s in ('nan','None','') else s

    out = []
    for _, row in df.iterrows():
        out.append({
            'ticker':        str(row.get('ticker') or '').strip(),
            'name':          str(row.get('name')   or '').strip(),
            'sector':        str(row.get('sector')  or ''),
            'country':       str(row.get('country') or ''),
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
            # 0-100 percentile scores (for colour circles)
            'value_rank':    fmt(row.get('value_rank')),
            'quality_rank':  fmt(row.get('quality_rank')),
            'momentum_rank': fmt(row.get('momentum_rank')),
            'qvm_rank':      fmt(row.get('qvm_score')),
            # unique integer ranks (for sorting/display)
            'value_pos':     int(row['value_int'])    if row.get('value_int')    is not None else None,
            'quality_pos':   int(row['quality_int'])  if row.get('quality_int')  is not None else None,
            'momentum_pos':  int(row['momentum_int']) if row.get('momentum_int') is not None else None,
            'qvm_pos':       int(row['qvm_int'])      if row.get('qvm_int')      is not None else None,
            # style archetype
            'style':         row.get('style') or '',
        })

    # Sort: best QVM first (lowest integer rank = rank 1)
    out.sort(key=lambda x: x['qvm_pos'] or 99999)
    return out

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
    print(f"  Top 5:")
    for s in stocks[:5]:
        print(f"    #{s['qvm_pos']:4}  {s['ticker']:12}  QVM={s['qvm_rank']}  style={s['style']}")
    print(f"  Style counts:")
    from collections import Counter
    sc = Counter(s['style'] for s in stocks)
    for style, n in sorted(sc.items(), key=lambda x: -x[1]):
        print(f"    {style:20} {n}")

if __name__ == '__main__':
    main()
