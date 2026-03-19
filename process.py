import argparse, json, math, os
from datetime import datetime, timezone
import pandas as pd
from scipy.stats import rankdata

COLUMN_MAP = {
    "ticker":        "Info - Ticker",
    "name":          "Company",
    "sector":        "Info - Sector",
    "country":       "Info - Country",
    "market_cap":    "Market Cap - Current SEK",
    "pe":            "P/E - Current",
    "peg":           "PEG - Current",
    "beta":          "Beta - 1y",
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
    (True,True,True):"Super Stock",(True,False,True):"High Flyer",
    (True,True,False):"Contrarian",(False,True,True):"Turnaround",
    (False,False,False):"Sucker Stock",(False,False,True):"Momentum Trap",
    (True,False,False):"Falling Star",(False,True,False):"Value Trap",
}

def clean(s):
    s = s.astype(str).str.strip()
    s = s.str.replace('%','',regex=False)
    s = s.str.replace(',','.',regex=False)
    s = s.str.replace('\xa0','',regex=False)
    s = s.str.replace(' ','',regex=False)
    return pd.to_numeric(s, errors='coerce')

def winsorize(s):
    return s.clip(s.quantile(0.02), s.quantile(0.98))

def safe_inv(s):
    s2 = s.astype(float).copy()
    s2[s2 <= 0] = float('nan')
    return 1.0 / s2

def wraw(df, wd):
    tw = pd.Series(0.0, index=df.index)
    ts = pd.Series(0.0, index=df.index)
    for col, w in wd.items():
        if col not in df.columns: continue
        mask = df[col].notna()
        if mask.sum() < 2: continue
        r = rankdata(df.loc[mask, col].values, method='average')
        pct = (r - 1) / (len(r) - 1) * 100
        ts[mask] += pct * w
        tw[mask] += w
    return ts / tw.replace(0, float('nan'))

def unique_score(series):
    mask = series.notna()
    result = pd.Series([None] * len(series), dtype=float)
    n = mask.sum()
    if n < 2: return result
    ranks = rankdata(-series[mask].values, method='ordinal')
    result[mask] = [round(100.0 * (n - r) / (n - 1), 2) for r in ranks]
    return result

def build_scores(csv_path):
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8', dtype=str)
    df.columns = [c.strip('\r').strip() for c in df.columns]
    print(f"  Loaded {len(df)} rows. First col: {repr(df.columns[0])}")

    missing = [v for v in COLUMN_MAP.values() if v not in df.columns]
    if missing:
        print(f"  MISSING COLUMNS: {missing}")
    else:
        print(f"  All {len(COLUMN_MAP)} columns matched OK")

    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items()})
    df['ticker'] = df['ticker'].astype(str).str.strip()
    df = df[df['ticker'].notna() & (df['ticker'] != '') & (df['ticker'] != 'nan')].copy()
    print(f"  After ticker filter: {len(df)} rows")

    for col in ['market_cap','pe','peg','beta','ev_ebit','ev_sales','p_fcf','f_score',
                'roe','roic','gross_margin','op_margin','net_margin','current_ratio',
                'nd_ebitda','rev_growth_3y','ret_3m','ret_6m','ret_12m']:
        if col in df.columns:
            df[col] = clean(df[col])
            if df[col].notna().sum() > 20:
                df[col] = winsorize(df[col])

    for src, dst in [('pe','inv_pe'),('ev_ebit','inv_ev_ebit'),('ev_sales','inv_ev_sales'),
                     ('p_fcf','inv_p_fcf'),('nd_ebitda','inv_nd_ebitda')]:
        if src in df.columns:
            df[dst] = safe_inv(df[src])

    if 'peg' in df.columns and 'beta' in df.columns:
        df['pegr'] = df['peg'] * df['beta'].abs()

    df['_vr'] = wraw(df, VALUE_WEIGHTS)
    df['_qr'] = wraw(df, QUALITY_WEIGHTS)
    df['_mr'] = wraw(df, MOMENTUM_WEIGHTS)

    raw_check = [c for c in ['pe','peg','f_score','ev_ebit','p_fcf','roe','roic',
                 'gross_margin','op_margin','net_margin','ret_12m','ret_6m','ret_3m']
                 if c in df.columns]
    df['_rf'] = df[raw_check].notna().sum(axis=1)
    df['_pf'] = df[['_vr','_qr','_mr']].notna().sum(axis=1)
    before = len(df)
    df = df[(df['_rf'] >= 3) & (df['_pf'] >= 2)].copy().reset_index(drop=True)
    print(f"  Filtered {before - len(df)} rows -> {len(df)} scoreable stocks")

    df['value_rank']    = unique_score(df['_vr'])
    df['quality_rank']  = unique_score(df['_qr'])
    df['momentum_rank'] = unique_score(df['_mr'])
    df['_qvm'] = (df['value_rank'].fillna(50) * (1/3) +
                  df['quality_rank'].fillna(50) * (1/3) +
                  df['momentum_rank'].fillna(50) * (1/3))
    df['qvm_rank'] = unique_score(df['_qvm'])
    df['style'] = df.apply(lambda r: STYLE_MAP.get(
        (r['quality_rank'] >= 50, r['value_rank'] >= 50, r['momentum_rank'] >= 50)), axis=1)

    def fmt(v):
        if v is None: return None
        try:
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 2)
        except:
            s = str(v)
            return None if s in ('nan','None','') else s

    out = []
    for _, row in df.iterrows():
        out.append({
            'ticker':        str(row.get('ticker') or '').strip(),
            'name':          str(row.get('name') or '').strip(),
            'sector':        str(row.get('sector') or ''),
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
    for i, s in enumerate(out):
        s['qvm_pos'] = i + 1
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
    print(f"  Written {len(stocks)} stocks -> {args.output}")
    for s in stocks[:5]:
        print(f"    #{s['qvm_pos']} {s['ticker']} QVM={s['qvm_rank']} {s['style']}")

if __name__ == '__main__':
    main()
