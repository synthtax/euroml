"""
EuroMLions — Inference-Only Prediction Script

Lightweight script for generating predictions from pre-trained models.
No training, no GPU required — just loads models and runs forward passes.
Designed to run on GitHub Actions in seconds.

Imports shared functions from train.py for feature engineering and
prediction utilities.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
import os
import sys
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Allow imports from the models directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (
    engineer_features, FocalLoss, focal_loss,
    predict_with_uncertainty, get_probabilities,
    format_predictions, ensemble_predictions,
    SEQUENCE_LENGTH, MC_DROPOUT_SAMPLES,
    ENSEMBLE_WEIGHT_SPECIALIST, ENSEMBLE_WEIGHT_COMBINED,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
CSV_PATH = os.path.join(DATA_DIR, 'historical-kaggle-clean.csv')
PREDICTIONS_PATH = os.path.join(DATA_DIR, 'predictions.json')
HISTORY_PATH = os.path.join(DATA_DIR, 'history-predictions.json')

MODEL_PATHS = {
    'combined': os.path.join(MODELS_DIR, 'model_combined.keras'),
    'main': os.path.join(MODELS_DIR, 'model_main.keras'),
    'stars': os.path.join(MODELS_DIR, 'model_stars.keras'),
}


def get_next_draw_date():
    """Calculate the next EuroMillions draw date (Tuesday or Friday)."""
    now = datetime.now()
    day = now.weekday()  # 0=Mon, 1=Tue, ..., 4=Fri

    if day < 1:         # Mon -> Tue
        days_ahead = 1 - day
    elif day < 4:       # Tue-Thu -> Fri
        days_ahead = 4 - day
    elif day == 4:      # Fri -> next Tue
        days_ahead = 4 if now.hour >= 21 else 0
    elif day == 5:      # Sat -> Tue
        days_ahead = 3
    else:               # Sun -> Tue
        days_ahead = 2

    # If it's draw day but after 21:00, next draw
    if day == 1 and now.hour >= 21:
        days_ahead = 3  # Tue -> Fri

    from datetime import timedelta
    next_draw = now + timedelta(days=days_ahead)
    return next_draw.strftime('%Y-%m-%d')


def load_models():
    """Load all three pre-trained models."""
    print("\nLoading pre-trained models...")
    models = {}

    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[name] = keras.models.load_model(path)
            print(f"  ✓ Loaded {name} from {os.path.basename(path)}")
        else:
            print(f"  ✗ Model not found: {path}")

    return models


def generate_predictions():
    """
    Full prediction pipeline:
    1. Load data and engineer features
    2. Load pre-trained models
    3. Run MC dropout inference
    4. Ensemble and save predictions
    5. Add entry to history
    """
    print("=" * 50)
    print("EuroMLions — Generating Predictions")
    print("=" * 50)

    # Load data
    print(f"\nLoading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df)} draws")

    # Feature engineering
    X_seq, X_agg, _, _ = engineer_features(df)

    # Load models
    models = load_models()

    if not models:
        print("✗ No models found. Cannot generate predictions.")
        sys.exit(1)

    # Prepare input (last sequence)
    last_seq = X_seq[-1:]
    last_agg = X_agg[-1:]
    inputs = [last_seq, last_agg]

    # Generate predictions from each available model
    print("\nRunning inference (MC dropout)...")
    probs = {}

    if 'combined' in models:
        probs['combined'] = get_probabilities(models['combined'], inputs, 'combined')
        print("  ✓ Model A (Combined)")

    if 'main' in models:
        probs['main'] = get_probabilities(models['main'], inputs, 'main')
        print("  ✓ Model B (Main Numbers)")

    if 'stars' in models:
        probs['stars'] = get_probabilities(models['stars'], inputs, 'stars')
        print("  ✓ Model C (Lucky Stars)")

    # Format individual predictions
    pred_combined = format_predictions(probs['combined']) if 'combined' in probs else {}
    pred_main = format_predictions(probs['main']) if 'main' in probs else {}
    pred_stars = format_predictions(probs['stars']) if 'stars' in probs else {}

    # Ensemble (needs combined + main + stars)
    if all(k in probs for k in ['combined', 'main', 'stars']):
        pred_ensemble = ensemble_predictions(probs['combined'], probs['main'], probs['stars'])
    elif 'combined' in probs:
        pred_ensemble = pred_combined
    else:
        pred_ensemble = pred_main

    next_draw = get_next_draw_date()

    # Save predictions.json
    output = {
        'generated': datetime.now().isoformat(),
        'next_draw': next_draw,
        'trained_on': f"{len(df)} draws (post-Sept 2016)",
        'sequence_length': SEQUENCE_LENGTH,
        'ensemble': pred_ensemble,
        'model_A_combined': pred_combined,
        'model_B_main_only': pred_main,
        'model_C_stars_only': pred_stars,
    }

    with open(PREDICTIONS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Predictions saved to {PREDICTIONS_PATH}")

    # Add entry to history (all models, not just ensemble)
    add_to_history(pred_ensemble, pred_combined, pred_main, pred_stars, next_draw)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"🎯 Predictions for {next_draw}:")
    print(f"   Main: {pred_ensemble.get('main_numbers', '?')}")
    print(f"   Stars: {pred_ensemble.get('lucky_stars', '?')}")
    print(f"{'=' * 50}")

    return output


def add_to_history(pred_ensemble, pred_combined, pred_main, pred_stars, draw_date):
    """Add a new prediction entry to history-predictions.json with all models."""
    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            history = json.load(f)

    # Check if we already have a prediction for this draw date
    existing_dates = [h['draw_date'] for h in history]
    if draw_date in existing_dates:
        print(f"✓ History already has entry for {draw_date}")
        return

    entry = {
        'draw_date': draw_date,
        'generated': datetime.now().isoformat(),
        'model_version': 'v1 (LSTM)',
        'ensemble': {
            'main': pred_ensemble.get('main_numbers', []),
            'stars': pred_ensemble.get('lucky_stars', []),
        },
        'model_A': {
            'main': pred_combined.get('main_numbers', []),
            'stars': pred_combined.get('lucky_stars', []),
        },
        'model_B': {
            'main': pred_main.get('main_numbers', []),
        },
        'model_C': {
            'stars': pred_stars.get('lucky_stars', []),
        },
        'actual_main': None,
        'actual_stars': None,
        'main_matched': None,
        'stars_matched': None,
    }

    history.append(entry)

    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Added prediction to history for {draw_date}")


if __name__ == '__main__':
    generate_predictions()
