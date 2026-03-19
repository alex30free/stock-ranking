"""
process.py  —  QVM Stock Ranking Engine
Confirmed working against Borsdata_export.csv (English export format)
"""

import argparse, json, math, os
from datetime import datetime, timezone
import pandas as pd
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# COLUMN MAP  (matches actual Börsdata English export exactly)
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "ticker":        "Info - Ticker",
    "name":          "Company",
    "sector":        "Info - Sector",
    "country":       "Info - Country",
    "market_cap":    "Market Cap - Current SEK",
    "pe":            "P/E - Current",
    "peg":           "PEG - Current",
    "beta":          "Beta - 1y",
    "pb":            "P/B - Current",
    "ev_ebit":       "EV/EBIT - Current",
    "ev_sales":      "EV/S - Current",
    "p_fcf":         "P/FCF - Current",
    "roe":           "ROE - Current",
    "roic":          "ROIC - Current",
    "gross_margin":  "Gross marg - Current",
    "op_margin":     "EBIT marg - Current",
    "net_margin":    "Profit marg - Current",
    "f_score":       "F-Score - Point",
    "current_ratio": "Current r. - Current",
    "rev_growth_3y": "Revenue g. - Growth 3y",
    "nd_ebitda":     "N.Debt/Ebitda - Current",
    "ret_3m":        "Performance - Perform. 3m",
    "ret_6m":        "Performance - Perform. 6m",
    "ret_12m":       "Performance - Perform. 1y",
}

VALUE_WEIGHTS    = {"inv_pe":0.25,"inv_ev_ebit":0.35,"inv_p_fcf":0.30,"inv_ev_sales":0.10}
QUALITY_WEIGHTS  = {"roe":0.12,"roic":0.12,"gross_margin":0.10,"op_margin":0.12,
                    "net_margin":0.07,"current_ratio":0.07,"inv_nd_ebitda":0.12,
                    "rev_growth_3y":0.08,"f_score":0.20}
MOMENTUM_WEIGHTS = {"ret_12m":0.50,"ret_6m":0.30,"ret_3m":0.20}

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

def clean_numeric(series):
    s = series.astype(str).str.strip()
    s = s.str.replace('%','',regex=False).str.replace(',','.',regex=False)
    s = s.str.replace('\xa0','',regex=False).str.replace(' ','',regex=False)
    return pd.to_numeric(s, errors='coerce')

def winsorize(series, low=0.02, high=0.98):
    return series.clip(series.quantile(low), series.quantile(high))

def safe_invert(series):
    s = series.astype(float).copy()
    s[s <= 0] = float('nan')
    return 1.0 / s

def weighted_raw_score(df, wd):
    tot_w = pd.Series(0.0, index=df.index)
    tot_s = pd.Series(0.0, index=df.index)
    for col, w in wd.items():
        if col not in df.columns: continue
        mask = df[col].notna()
        if mask.sum() < 2: continue
        vals = df.loc[mask, col].values
        r = rankdata(vals, method='average')
        pct = (r - 1) / (len(r) - 1) * 100
        tot_s[mask] += pct * w
        tot_w[mask] += w
    return tot_s / tot_w.replace(0, float('nan'))

def unique_score(series):
    mask = series.notna()
    result = pd.Series([None] * len(series), dtype=float)
    n = mask.sum()
    if n < 2: return result
    ranks = rankdata(-series[mask].values, method='ordinal')
    result[mask] = [round(100.0 * (n - r) / (n - 1), 2) for r in ranks]
    return result

def build_scores(csv_path):
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig', dtype=str)
    # Strip BOM, quotes, Windows carriage returns from column names
    df.columns = [c.strip().strip('"').strip('\r') for c in df.columns]
    print(f"  Loaded {len(df):,} rows — first col: {repr(df.columns[0])}")

    # Show match diagnostics
    unmatched = [v for v in COLUMN_MAP.values() if v not in df.columns]
    if unmatched:
        print(f"  WARNING — columns not found: {unmatched}")
    else:
        print(f"  All {len(COLUMN_MAP)} columns matched OK")

    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items()})
    if 'ticker' not in df.columns:
        raise ValueError(f"ticker column missing. Available: {list(df.columns)}")

    df['ticker'] = df['ticker'].astype(str).str.strip().str.strip('"')
    df = df[df['ticker'].notna() & (df['ticker'] != '') & (df['ticker'] != 'nan')].copy()
    print(f"  After ticker filter: {len(df):,} rows")

    # String cols
    for col in ['name', 'sector', 'country']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.strip('"')

    # Numeric parsing
    num_cols = ['market_cap','pe','peg','beta','pb','ev_ebit','ev_sales','p_fcf',
                'f_score','roe','roic','gross_margin','op_margin','net_margin',
                'current_ratio','nd_ebitda','rev_growth_3y','ret_3m','ret_6m','ret_12m']
    for col in num_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # Winsorize
    for col in num_cols:
        if col in df.columns and df[col].notna().sum() > 20:
            df[col] = winsorize(df[col])

    # Invert bad-is-high metrics
    for src, dst in [('pe','inv_pe'),('ev_ebit','inv_ev_ebit'),('ev_sales','inv_ev_sales'),
                     ('p_fcf','inv_p_fcf'),('nd_ebitda','inv_nd_ebitda')]:
        if src in df.columns: df[dst] = safe_invert(df[src])

    # PEGR = PEG × |Beta|
    if 'peg' in df.columns and 'beta' in df.columns:
        df['pegr'] = df['peg'] * df['beta'].abs()

    # Composite raw scores
    df['_vr'] = weighted_raw_score(df, VALUE_WEIGHTS)
    df['_qr'] = weighted_raw_score(df, QUALITY_WEIGHTS)
    df['_mr'] = weighted_raw_score(df, MOMENTUM_WEIGHTS)

    # Filter shells / ETFs with almost no data
    raw_check = [c for c in ['pe','peg','f_score','ev_ebit','p_fcf','roe','roic',
                 'gross_margin','op_margin','net_margin','ret_12m','ret_6m','ret_3m']
                 if c in df.columns]
    df['_rf'] = df[raw_check].notna().sum(axis=1)
    df['_pf'] = df[['_vr','_qr','_mr']].notna().sum(axis=1)
    before = len(df)
    df = df[(df['_rf'] >= 3) & (df['_pf'] >= 2)].copy().reset_index(drop=True)
    print(f"  Removed {before - len(df)} low-data rows → {len(df):,} scoreable stocks")

    # Unique scores (no two stocks share a value)
    df['value_rank']    = unique_score(df['_vr'])
    df['quality_rank']  = unique_score(df['_qr'])
    df['momentum_rank'] = unique_score(df['_mr'])
    df['_qvm'] = (df['value_rank'].fillna(50) * (1/3) +
                  df['quality_rank'].fillna(50) * (1/3) +
                  df['momentum_rank'].fillna(50) * (1/3))
    df['qvm_rank'] = unique_score(df['_qvm'])

    # Verify uniqueness
    for col in ['value_rank','quality_rank','momentum_rank','qvm_rank']:
        vals = df[col].dropna()
        assert len(vals) == len(vals.unique()), f"DUPLICATE in {col}!"
    print("  Uniqueness check passed")

    # Style archetype
    df['style'] = df.apply(lambda r: STYLE_MAP.get(
        (r['quality_rank'] >= 50, r['value_rank'] >= 50, r['momentum_rank'] >= 50)), axis=1)

    def fmt(v):
        if v is None: return None
        try:
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 2)
        except: return str(v) if str(v) not in ('nan','None','') else None

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
            'roic':          fmt(row.get('roic')),
            'f_score':       fmt(row.get('f_score')),
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
            'style':         row.get('style') or '',
        })

    out.sort(key=lambda x: x['qvm_rank'] or 0, reverse=True)
    for i, s in enumerate(out): s['qvm_pos'] = i + 1
    return out

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
    print(f"\n  Top 10:")
    for s in stocks[:10]:
        print(f"    #{s['qvm_pos']:4}  {s['ticker']:12}  QVM={s['qvm_rank']:6}  "
              f"V={s['value_rank']:6}  Q={s['quality_rank']:6}  M={s['momentum_rank']:6}  {s['style']}")
    from collections import Counter
    print(f"\n  Style distribution:")
    for style, n in sorted(Counter(s['style'] for s in stocks).items(), key=lambda x:-x[1]):
        print(f"    {style:20} {n}")

if __name__ == '__main__':
    main()
