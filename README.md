# 🪸 Coral Sentinel – Coral Bleaching Early-Warning System

A beautiful, funding-ready multi-site early-warning system that predicts the probability of dangerous heat stress (DHW ≥ 4 °C-weeks) on coral reefs.

## Highlights (Bangla-first, understood in ~10 seconds)

- **Plain, strong Bangla wording** for a general audience: *প্রবাল ক্ষয়*, *পর্যটক কমান*, *ভিড় কমান*, *পর্যটন বন্ধ করুন* (instead of "ব্লিচিং").
- **Welcome page flow:** সমস্যা (৩ ধাপ) → সুস্থ বনাম ক্ষয়ে যাওয়া প্রবাল → সমাধান → ৫ রঙের সংকেত → কে কী করবেন।
- **5-colour traffic-light alert**, same colours everywhere (banner, risk bar, chart zones):

  | Level | Colour | Bangla | Action shown |
  |-------|--------|--------|--------------|
  | 0 | 🟢 Green | নিরাপদ | Normal tourism, don't touch coral |
  | 1 | 🟡 Yellow | সাবধান | Don't add new crowds |
  | 2 | 🟠 Orange | বিপদ | **পর্যটক কমান, ভিড় কমান** |
  | 3 | 🔴 Red | মহাবিপদ | **পর্যটন সীমিত করুন** |
  | 4 | 🟣 Purple | চরম জরুরি | **পর্যটন বন্ধ করুন** |

- Big risk bar ("আগামী N দিনে প্রবাল ক্ষয়ের ঝুঁকি: ৭২%") with Bangla numerals.
- DHW chart with coloured risk zones.
- Multi-site support: Saint Martin, Havelock, North Andaman.
- Calibrated probabilities + reliability indicator; forecast mode using Open-Meteo.
- Theme set in `.streamlit/config.toml` (light, ocean + coral palette).

## Real "after" photo (optional)

The welcome page shows a simulated bleached look by default. To show a real photo,
save one as `assets/bleached_coral.jpg` (see `assets/README.txt`).

## Performance notes

- Forecast mode downloads only the *missing recent days* from NOAA CRW (cached in `data/crw_cache/`),
  fetches in parallel, and uses short timeouts, so it no longer hangs on slow connections.
- Model bundles are cached for the whole session; results of a forecast stay on screen while you use the sidebar.

## Probability vs accuracy (important)

| Quantity | What it means |
|----------|----------------|
| **Calibrated probability** (e.g. 99.9% or 0%) | Model's estimate that *this* situation will reach DHW ≥ 4 within *n* days. |
| **Historical accuracy @ 0.50** | Fraction of past test days correctly classified at threshold 0.50. |

A high probability means “conditions look dangerous *now*”. High accuracy means “the model was usually right on past test days”.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the browser → you will land on the **Welcome / Story** page.  
Click **“আজকের প্রবাল ঝুঁকি দেখুন (ড্যাশবোর্ড)”** to use the prediction tools.

## Train models (if needed)

```bash
python scripts/train_and_save_models.py --sites all --quick          # 7,14,28
python scripts/train_and_save_models.py --sites all --horizons 1,3,7,14,21,28
```

## Project structure

- `app.py` – Beautiful Streamlit UI (welcome + dashboard)
- `models/` – Trained model bundles
- `features/` – Feature engineering
- `data/` – Data clients & cache
- `utils/alerts.py` – Alert mapping (5 levels)
- `docs/` – Model card, data card, reliability notes
- `tests/` – Unit tests

## Why this matters

Coral reefs are under severe heat stress worldwide.  
Early, understandable warnings give local communities, managers and researchers time to act **before** bleaching becomes irreversible.

This system is a research prototype that complements (does not replace) NOAA Coral Reef Watch.
