# QVMRank — Börsdata Stock Ranking System

A self-hosted Stockopedia-style Value / Quality / Momentum ranking system
that runs fully automated on GitHub, updated with a single weekly CSV upload.

---

## Repository structure

```
├── index.html                        # Ranking webpage (GitHub Pages)
├── process.py                        # Scoring script
├── data/
│   ├── börsdata_export.csv           # ← You upload this weekly
│   └── scores.json                   # ← Auto-generated, do not edit
├── .github/
│   └── workflows/
│       └── update-scores.yml         # GitHub Actions automation
└── README.md
```

---

## One-time setup

### 1. Create GitHub repo & enable Pages
1. Create a new **public** GitHub repository (private requires GitHub Pro for Pages).
2. Go to **Settings → Pages → Source → Deploy from branch → main / (root)**.
3. Your site will be live at `https://<you>.github.io/<repo>/`.

### 2. Upload the four files
Put `index.html`, `process.py`, and `.github/workflows/update-scores.yml`
in the root. Create an empty `data/` folder (add a `.gitkeep` file).

### 3. Edit `COLUMN_MAP` in `process.py`
Open `process.py` and update the `COLUMN_MAP` dict (top of file) so the
right-hand values match your actual Börsdata CSV column headers exactly.
To see what columns you have, open any Börsdata export and check row 1.

---

## Weekly workflow (2 minutes)

1. Open **Börsdata → Screener / Export**.
2. Export the stock universe with the columns listed below.
3. Save the file to your local machine.
4. In the GitHub repo, navigate to `data/` and click **Add file → Upload files**.
5. Upload your CSV (you can overwrite the old one or keep a dated copy).
6. Click **Commit changes**.
7. GitHub Actions runs automatically → `data/scores.json` is updated →
   GitHub Pages serves the new rankings within ~60 seconds.

---

## Columns to export from Börsdata

| Category  | Columns (Swedish names)                                                               |
|-----------|--------------------------------------------------------------------------------------|
| Identity  | Ticker, Bolagsnamn, Sektor, Börsvärde                                                |
| Value     | P/E, P/B, EV/EBIT, EV/Omsättning, P/FCF, Direktavkastning                          |
| Quality   | ROE, ROIC, Bruttomarginal, Rörelsemarginal, Nettomarginal, Kassalikviditet, Nettoskuld/EBITDA, Omsättningstillväxt 3 år |
| Momentum  | Avkastning 3M, Avkastning 6M, Avkastning 12M (or raw prices Kurs, Kurs 3M, Kurs 6M, Kurs 12M) |

---

## Scoring methodology

### Value score (0–100)
Percentile rank of: inverse P/E (20%) · inverse P/B (20%) · inverse EV/EBIT (25%)
· inverse P/FCF (20%) · dividend yield (15%)

Lower valuation multiples = higher rank. Negatives and zeros are excluded.

### Quality score (0–100)
Percentile rank of: ROE (15%) · ROIC (15%) · gross margin (10%)
· operating margin (15%) · net margin (10%) · current ratio (10%)
· inverse net debt/EBITDA (15%) · 3Y revenue CAGR (10%)

### Momentum score (0–100)
Percentile rank of weighted return: 12M×50% + 6M×30% + 3M×20%

### QVM rank (0–100)
Final percentile rank of: Value×33% + Quality×33% + Momentum×33%

All factors are winsorized at 2nd/98th percentile before ranking to
reduce outlier distortion.

---

## Local development / testing

```bash
pip install pandas scipy openpyxl
python process.py --input data/börsdata_export.csv --output data/scores.json
# Then open index.html in a browser via a local server:
python -m http.server 8000
# Visit http://localhost:8000
```

---

## Customisation

| What to change | Where |
|---|---|
| Column names | `COLUMN_MAP` in `process.py` |
| Factor weights | `VALUE_WEIGHTS`, `QUALITY_WEIGHTS`, `MOMENTUM_WEIGHTS`, `QVM_WEIGHTS` in `process.py` |
| Universe (filter by market, size) | Add a filter step after `df = df[df["ticker"].notna()]` |
| Colour theme | CSS variables at top of `index.html` |
| Schedule | `cron:` line in `update-scores.yml` |
