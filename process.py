"""
process.py — QVM Stock Ranking Engine
--------------------------------------
Scores are UNIQUE across all stocks for every column.
Formula: score = 100 * (N - rank) / (N - 1)
Rank 1 (best) → score 100.00
Rank N (worst) → score 0.00
No two stocks share the same score in any column.

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
    "peg":           "PEG - Senaste",
    "beta":          "Beta - Senaste",
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
# Factor weights
# ---------------------------------------------------------------------------
VALUE_WEIGHTS = {
    "inv_pe":       0.25,
    "inv_ev_ebit":  0.35,
    "inv_p_fcf":    0.30,
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

MIN_RAW_FIELDS    = 3
MIN_PILLAR_SCORES = 2

# ---------------------------------------------------------------------------
# Stockopedia 8-archetype style map
# ---------------------------------------------------------------------------
STYLE_MAP = {
    (True,  True,  True):  "Super Stock",
    (True,  False, True):  "High Flyer",
    (True,  True,  False): "Contrarian",
    (False, True,  True):  "Turnaround",
    (False, False, False): "Sucker Stock",
    (False, False, True):  "Momentum Trap",
    (True,  False, False): "Falling Star",
    (False, True,  False): "Value Trap",
}

def classify_style(q, v, m):
    if q is None or v is None or m is None:
        return None
    return STYLE_MAP.get((q >= 50, v >= 50, m >= 50))

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def clean_numeric(series):
    s = series.astype(str).str.strip()
    s = s.str.replace('%', '', regex=False)
    s = s.str.replace(',', '.', regex=False)
    s = s.str.replace('\xa0', '', regex=False)
    s = s.str.replace(' ', '', regex=False)
    return pd.to_numeric(s, errors='coerce')

def winsorize(series, low=0.02, high=0.98):
    lo, hi = series.quantile(low), series.quantile(high)
    return series.clip(lo, hi)

def safe_invert(series):
    s = series.astype(float).copy()
    s[s <= 0] = float('nan')
    return 1.0 / s

def weighted_raw_score(df, weight_dict):
    """
    Weighted average of 0-100 percentile ranks (used ONLY internally
    to get a composite raw score for final ranking — not shown to user).
    """
    from scipy.stats import rankdata as _rd
    total_w     = pd.Series(0.0, index=df.index)
    total_score = pd.Series(0.0, index=df.index)
    for col, w in weight_dict.items():
        if col not in df.columns:
            continue
        mask = df[col].notna()
        if mask.sum() < 2:
            continue
        # percentile rank of this factor (0-100)
        vals = df.loc[mask, col].values
        r = _rd(vals, method='average')
        pct = (r - 1) / (len(r) - 1) * 100
        total_score[mask] += pct * w
        total_w[mask]     += w
    return total_score / total_w.replace(0, float('nan'))

def unique_score(series):
    """
    Convert a raw composite score series into UNIQUE scores on a 0-100 scale.

    Method:
      1. Rank stocks by raw score (higher raw = better = lower rank number).
      2. score_i = 100 * (N - rank_i) / (N - 1)
         → rank 1 (best)  gets score 100.00
         → rank N (worst) gets score 0.00
      3. Round to 2 decimal places — guaranteed unique for any N.

    Stocks with NaN raw score are excluded from ranking and get NaN score.
    """
    mask = series.notna()
    result = pd.Series([None] * len(series), dtype=float)
    n = mask.sum()
    if n < 2:
        return result

    vals = series[mask].values
    # rank: highest val → rank 1  (ascending=False achieved by negating)
    ranks = rankdata(-vals, method='ordinal')   # ordinal = no ties, ever
    scores = [round(100.0 * (n - r) / (n - 1), 2) for r in ranks]
    result[mask] = scores
    return result

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

    # Parse numerics
    numeric_cols = [
        'market_cap', 'pe', 'peg', 'beta', 'ev_ebit', 'ev_sales', 'p_fcf',
        'roe', 'roic', 'gross_margin', 'op_margin', 'net_margin',
        'current_ratio', 'nd_ebitda', 'rev_growth_3y',
        'ret_3m', 'ret_6m', 'ret_12m',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # Winsorize
    for col in numeric_cols:
        if col in df.columns and df[col].notna().sum() > 20:
            df[col] = winsorize(df[col])

    # Invert bad-is-high metrics
    for src, dst in [
        ('pe',       'inv_pe'),
        ('ev_ebit',  'inv_ev_ebit'),
        ('ev_sales', 'inv_ev_sales'),
        ('p_fcf',    'inv_p_fcf'),
        ('nd_ebitda','inv_nd_ebitda'),
    ]:
        if src in df.columns:
            df[dst] = safe_invert(df[src])

    # Calculate PEGR = PEG * Beta (risk-adjusted PEG)
    if 'peg' in df.columns and 'beta' in df.columns:
        df['pegr'] = df['peg'] * df['beta'].abs()

    # Raw composite scores (internal use only)
    df['_val_raw']  = weighted_raw_score(df, VALUE_WEIGHTS)
    df['_qual_raw'] = weighted_raw_score(df, QUALITY_WEIGHTS)
    df['_mom_raw']  = weighted_raw_score(df, MOMENTUM_WEIGHTS)

    # ── Filter: remove ETFs, shells, stubs ─────────────────────────────────
    raw_check = [c for c in [
        'pe','peg','ev_ebit','p_fcf','roe','roic',
        'gross_margin','op_margin','net_margin',
        'ret_12m','ret_6m','ret_3m',
    ] if c in df.columns]

    df['_raw_filled']    = df[raw_check].notna().sum(axis=1)
    df['_pillar_filled'] = df[['_val_raw','_qual_raw','_mom_raw']].notna().sum(axis=1)

    before = len(df)
    df = df[
        (df['_raw_filled']    >= MIN_RAW_FIELDS) &
        (df['_pillar_filled'] >= MIN_PILLAR_SCORES)
    ].copy().reset_index(drop=True)
    print(f"  Removed {before - len(df)} ETFs/shells → {len(df):,} scoreable stocks")

    # ── UNIQUE SCORES — each pillar ranked independently ───────────────────
    # value_rank, quality_rank, momentum_rank: unique floats 0.00–100.00
    df['value_rank']    = unique_score(df['_val_raw'])
    df['quality_rank']  = unique_score(df['_qual_raw'])
    df['momentum_rank'] = unique_score(df['_mom_raw'])

    # QVM composite: weighted average of the three unique pillar scores
    # then ranked again → also fully unique
    df['_qvm_raw'] = (
        df['value_rank'].fillna(50)    * (1/3) +
        df['quality_rank'].fillna(50)  * (1/3) +
        df['momentum_rank'].fillna(50) * (1/3)
    )
    df['qvm_rank'] = unique_score(df['_qvm_raw'])

    # Verify uniqueness
    for col in ['value_rank','quality_rank','momentum_rank','qvm_rank']:
        vals = df[col].dropna()
        assert len(vals) == len(vals.unique()), f"DUPLICATE in {col}!"
    print(f"  Uniqueness check passed — all 4 score columns fully unique")

    # ── Style classification ────────────────────────────────────────────────
    # Uses 50 as threshold (same as Stockopedia)
    df['style'] = df.apply(
        lambda r: classify_style(
            r['quality_rank'], r['value_rank'], r['momentum_rank']
        ), axis=1
    )

    # ── Serialise ───────────────────────────────────────────────────────────
    def fmt(v):
        if v is None: return None
        try:
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 2)
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
            'peg':           fmt(row.get('peg')),
            'pegr':          fmt(row.get('pegr')),
            'ev_ebit':       fmt(row.get('ev_ebit')),
            'p_fcf':         fmt(row.get('p_fcf')),
            'roe':           fmt(row.get('roe')),
            'op_margin':     fmt(row.get('op_margin')),
            'ret_12m':       fmt(row.get('ret_12m')),
            'ret_6m':        fmt(row.get('ret_6m')),
            'ret_3m':        fmt(row.get('ret_3m')),
            # UNIQUE scores — 0.00 to 100.00, no two stocks share same value
            'value_rank':    fmt(row.get('value_rank')),
            'quality_rank':  fmt(row.get('quality_rank')),
            'momentum_rank': fmt(row.get('momentum_rank')),
            'qvm_rank':      fmt(row.get('qvm_rank')),
            'style':         row.get('style') or '',
        })

    # Sort: best StockRank first (score 100 at top)
    out.sort(key=lambda x: x['qvm_rank'] or 0, reverse=True)

    # Add display rank number (1 = best)
    for i, s in enumerate(out):
        s['qvm_pos'] = i + 1

    return out

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
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
    print(f"\n  Top 10 by StockRank:")
    for s in stocks[:10]:
        print(f"    #{s['qvm_pos']:4}  {s['ticker']:12}  QVM={s['qvm_rank']:6}  V={s['value_rank']:6}  Q={s['quality_rank']:6}  M={s['momentum_rank']:6}  {s['style']}")

    print(f"\n  Score uniqueness verification:")
    from collections import Counter
    for key in ['value_rank','quality_rank','momentum_rank','qvm_rank']:
        vals = [s[key] for s in stocks if s[key] is not None]
        dupes = {k:v for k,v in Counter(vals).items() if v > 1}
        print(f"    {key:15} {len(vals)} scores, {len(dupes)} duplicates ← {'OK' if not dupes else 'PROBLEM'}")

    print(f"\n  Style distribution:")
    from collections import Counter
    for style, n in sorted(Counter(s['style'] for s in stocks).items(), key=lambda x:-x[1]):
        print(f"    {style:20} {n}")

if __name__ == '__main__':
    main()
