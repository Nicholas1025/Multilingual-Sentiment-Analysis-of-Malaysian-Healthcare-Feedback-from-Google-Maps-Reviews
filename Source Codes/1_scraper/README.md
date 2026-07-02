# Data Collection — Healthcare Feedback (Melaka, Malaysia)

This folder collects public sentiment toward Malaysian healthcare services in
Melaka: government hospitals, private hospitals, specialist centres, and
government health clinics (Klinik Kesihatan).

## Pipeline

```
scrape_google_maps.py  ─┐
                        ├─→  Data/raw/*.csv  ─→  merge_and_check.py  ─→  healthcare_merged.csv
scrape_playstore.py    ─┘                                                      │
                                                                               ▼
                                                              verify_labels.py (human gold set)
                                                                               │
                                                                               ▼
                                                              healthcare_verified.csv
```

## Setup

```bash
pip install -r requirements.txt
```

The Google Maps scraper needs Google Chrome installed. Selenium 4.6+ auto-downloads
the matching ChromeDriver — no manual driver setup required.

## Step-by-step

| # | Command | Output |
|---|---------|--------|
| 1 | `python scrape_google_maps.py` | `Data/raw/healthcare_raw.csv` |
| 2 | `python scrape_playstore.py`   | `Data/raw/playstore_raw.csv` |
| 3 | `python merge_and_check.py`    | `Data/raw/healthcare_merged.csv` + quality report |
| 4 | `python verify_labels.py`      | `Data/raw/healthcare_verified.csv` (gold set) |

## Data sources

| Source | Platform | Content type |
|--------|----------|--------------|
| Google Maps | 12 Melaka hospitals & clinics | Star-rated public reviews |
| Google Play Store | MySejahtera + hospital apps | Star-rated app reviews |

## Labelling scheme

Auto-label from star rating, then **human-verified**:

| Stars | Sentiment |
|-------|-----------|
| 4 – 5 | positive |
| 3     | neutral |
| 1 – 2 | negative |

Star ratings are noisy proxies for sentiment, so `merge_and_check.py` flags every
star-vs-text mismatch and `verify_labels.py` lets a human correct them. The final
`healthcare_verified.csv` is the gold dataset used for training.

## Rubric target

Minimum: 360 reviews (120 positive / 120 neutral / 120 negative).
This pipeline collects **160 per class (480 total)** as a buffer, then balances
to the required minimum after verification.

## Notes for the Final Report (Data Collection section)

- Quote the full quality report printed by `merge_and_check.py`.
- Mention multi-source collection (Maps + Play Store) and language diversity
  (English / Malay / Manglish code-switching) — both detected automatically.
- Describe the auto-label + human-verification two-stage process as the
  reliability control.
- All scripts are reproducible: fixed random seeds, pinned dependency versions.
