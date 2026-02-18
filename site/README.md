# 🎰 EuroMLions

### _An ML model that predicts EuroMillions numbers. It will not work, probably. And that's ok._

---

## What is this?

EuroMLions (EuroML for short) is a self-aware, confidently wrong machine learning project that attempts to predict EuroMillions lottery numbers twice a week.

There are two concurrent models - one trained on the historical data from Kaggle _after 27th Septmber 2016_ and the other is trained on the data from the last 6 months, from the National Lottery Site. These models generate predictions before each Tuesday and Friday draw, compares them to the actual results, and retrains themselves accordingly — all automatically, without human intervention.

It will probably not win the lottery. This is known. This is fine. 💙

---

## How it works

1. Historical EuroMillions data is loaded and analysed
2. A model identifies statistical patterns (frequency, gaps, etc.)
3. Predictions are generated for the next draw
4. GitHub Actions runs the whole thing automatically every Tuesday and Friday
5. Results are compared, the model retrains, new predictions are saved
6. A summary email lands in the inbox — a fun reminder of how wrong it was
7. A password-protected website displays the full history and stats

---

## Tech Stack

| Layer         | Technology                          |
| ------------- | ----------------------------------- |
| ML model      | TensorFlow                          |
| Automation    | GitHub Actions (scheduled workflow) |
| Notifications | GitHub Actions email summary        |
| Website       | HTML / CSS / JavaScript             |
| Hosting       | Netlify (password protected)        |
| Repo          | Private GitHub                      |

---

## Project Structure

```
/euroml
  /models
    train.py                # Model training + prediction logic
    predict.py              # Inference-only (used by GitHub Actions)
    fetch_results.py        # Fetch latest EuroMillions results from NL site
    update.py               # Orchestrator called by GitHub Actions
    model_combined.keras    # Pre-trained Model A (combined)
    model_main.keras        # Pre-trained Model B (main numbers)
    model_stars.keras       # Pre-trained Model C (lucky stars)
  /data
    historical-kaggle-clean.csv   # Historical draw results (post-Sept 2016)
    predictions.json              # Latest predictions (read by website)
    history-predictions.json      # Predictions over time (read by website)
  /site
    index.html              # Website
    style.css               # Styling
    script.js               # Reads JSON files, renders UI
  /.github
    /workflows
      predict.yml           # Scheduled workflow (Tues/Fri) + email summary
  netlify.toml              # Netlify build config
  requirements.txt          # Python dependencies
  README.md                 # You are here
```

---

## Build Order

**Phase 1 — Get the plumbing working**

- [x] Create repo
- [x] Source and clean historical EuroMillions data
- [x] Build neural network model (LSTM with feature engineering)
- [x] Train three models: combined, main specialist, stars specialist
- [x] Ensemble predictions with MC dropout uncertainty

**Phase 2 — Automate + Deploy**

- [x] Build static website (dark theme, lottery balls, playful)
- [x] Connect site to predictions.json + history-predictions.json
- [x] Set up GitHub Actions workflow (Tue/Fri schedule)
- [x] Add email summary step to workflow
- [ ] Deploy to Netlify with password protection
- [ ] Configure SMTP secrets for email notifications

**Phase 3 — Polish**

- [ ] Humour, comedy elements, and general chaos
- [ ] Claude API for sarky weekly commentary (optional)
- [ ] NL-specific model using machine + ball set features

---

## Predictions format (`predictions.json`)

```json
{
  "generated": "2026-02-18T20:00:00Z",
  "next_draw": "2026-02-20",
  "main_numbers": [7, 14, 23, 35, 42],
  "lucky_stars": [3, 11],
  "confidence": "extremely low",
  "last_draw": {
    "date": "2026-02-18",
    "actual_main": [4, 19, 23, 31, 50],
    "actual_lucky_stars": [1, 12],
    "numbers_correct": 1,
    "lucky_stars_correct": 0
  }
}
```

---

## Requirements

```
numpy
pandas
tensorflow
requests
beautifulsoup4
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Running locally

```bash
# Run the full pipeline (auto-detects pre-draw vs post-draw)
python models/update.py

# Or specify mode explicitly
python models/update.py --mode predict    # generate predictions
python models/update.py --mode compare    # fetch results + compare

# Or run steps individually
python models/predict.py                  # inference only
python models/fetch_results.py            # fetch latest results from NL
python models/train.py                    # full training (run on Colab)
```

---

## A note on randomness

EuroMillions draws are genuinely, mathematically random. Each draw is completely independent of all previous draws. No model — however sophisticated — can predict truly random events.

---

_EuroMLions — predicting EuroMillions with ML since 2026_ 🍀
