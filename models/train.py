"""
EuroMLions - TensorFlow Models (v2)

Three neural network models trained on EuroMillions data:
1. Combined (main numbers + Lucky Stars) - dual-output LSTM
2. Main numbers only - specialist LSTM
3. Lucky Stars only - specialist LSTM

v2 improvements over v1:
- LSTM architecture for temporal pattern learning
- Rich feature engineering (frequencies, gaps, draw statistics)
- Input normalisation
- Focal loss for sparse multi-hot targets
- Learning rate scheduling + model checkpointing
- Ensemble predictions blending all three models
- Monte Carlo dropout for uncertainty estimation
- Lottery-specific evaluation metrics
- Training history saved for analysis
- Reproducible via seed control
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from datetime import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─── Configuration ────────────────────────────────────────────────────────────

SEED = 42
SEQUENCE_LENGTH = 100
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
MC_DROPOUT_SAMPLES = 50
ENSEMBLE_WEIGHT_SPECIALIST = 0.6
ENSEMBLE_WEIGHT_COMBINED = 0.4
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# ─── Reproducibility ─────────────────────────────────────────────────────────

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# Loss Function
# ═══════════════════════════════════════════════════════════════════════════════

@keras.utils.register_keras_serializable(package='EuroMLions')
class FocalLoss(keras.losses.Loss):
    """
    Focal loss for imbalanced multi-hot targets.

    Standard binary crossentropy treats all 50 number slots equally,
    but only 5 are positive per draw (10%). The model can score high
    accuracy by predicting low probability everywhere. Focal loss
    down-weights easy negatives and concentrates learning on the
    harder task of predicting which numbers WILL be drawn.

    Registered with Keras serialization so saved models load cleanly
    in other scripts (update.py, etc.) without custom_objects hacks.
    """
    def __init__(self, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return tf.reduce_mean(focal_weight * bce, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config


def focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """Convenience wrapper that returns a FocalLoss instance."""
    return FocalLoss(gamma=gamma, alpha=alpha)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df, sequence_length=SEQUENCE_LENGTH):
    """
    Transform raw draw data into rich features for the LSTM.

    Per-timestep features (fed into the LSTM path):
        - 5 normalised main numbers (/50)
        - 2 normalised lucky stars (/12)
        - Is-Friday flag (binary: 0 = Tuesday, 1 = Friday)
        - Main number sum (normalised)
        - Main number range (normalised)
        - Main number mean (normalised)
        - Lucky star sum (normalised)
        -> 12 features per timestep

    Aggregate features (computed per window, fed into the dense path):
        - 50 frequency values for each main number
        - 12 frequency values for each lucky star
        - 50 recency-gap values for each main number
        - 12 recency-gap values for each lucky star
        -> 124 features per sample

    Returns:
        X_seq:   (n_samples, sequence_length, 12) - LSTM input
        X_agg:   (n_samples, 124)                 - dense input
        y_main:  (n_samples, 50) multi-hot
        y_stars: (n_samples, 12) multi-hot
    """
    print("\n--- Feature Engineering ---")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['is_friday'] = (df['date'].dt.dayofweek == 4).astype(np.float32)

    main_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
    star_cols = ['s1', 's2']

    raw_main = df[main_cols].values.astype(np.float32)
    raw_stars = df[star_cols].values.astype(np.float32)
    is_friday = df['is_friday'].values

    # Normalise raw numbers to [0, 1]
    norm_main = raw_main / 50.0
    norm_stars = raw_stars / 12.0

    # Per-draw statistics (normalised)
    main_sum = raw_main.sum(axis=1) / 250.0
    main_range = (raw_main.max(axis=1) - raw_main.min(axis=1)) / 49.0
    main_mean = raw_main.mean(axis=1) / 50.0
    stars_sum = raw_stars.sum(axis=1) / 23.0

    # Assemble per-timestep features: 5 + 2 + 1 + 1 + 1 + 1 + 1 = 12
    per_step = np.column_stack([
        norm_main,     # 5 features
        norm_stars,    # 2 features
        is_friday,     # 1 feature
        main_sum,      # 1 feature
        main_range,    # 1 feature
        main_mean,     # 1 feature
        stars_sum,     # 1 feature
    ])

    features_per_step = per_step.shape[1]

    X_seq_list = []
    X_agg_list = []
    y_main_list = []
    y_stars_list = []

    for i in range(sequence_length, len(df)):
        # Per-timestep sequence for the LSTM
        X_seq_list.append(per_step[i - sequence_length:i])

        # Aggregate features over the window
        window_main = raw_main[i - sequence_length:i]
        window_stars = raw_stars[i - sequence_length:i]

        # Frequency: how often each number appeared in this window
        main_freq = np.zeros(50, dtype=np.float32)
        stars_freq = np.zeros(12, dtype=np.float32)
        for row in window_main:
            for num in row:
                main_freq[int(num) - 1] += 1
        for row in window_stars:
            for star in row:
                stars_freq[int(star) - 1] += 1
        main_freq /= sequence_length
        stars_freq /= sequence_length

        # Recency gap: how many draws since each number last appeared
        main_gap = np.full(50, float(sequence_length), dtype=np.float32)
        stars_gap = np.full(12, float(sequence_length), dtype=np.float32)
        for j in range(sequence_length):
            draws_ago = sequence_length - 1 - j
            for num in window_main[j]:
                idx = int(num) - 1
                if draws_ago < main_gap[idx]:
                    main_gap[idx] = draws_ago
            for star in window_stars[j]:
                idx = int(star) - 1
                if draws_ago < stars_gap[idx]:
                    stars_gap[idx] = draws_ago
        main_gap /= sequence_length
        stars_gap /= sequence_length

        X_agg_list.append(np.concatenate([main_freq, stars_freq, main_gap, stars_gap]))

        # Targets: multi-hot encoding
        target_main = raw_main[i]
        target_stars = raw_stars[i]

        main_hot = np.zeros(50, dtype=np.float32)
        for num in target_main:
            main_hot[int(num) - 1] = 1.0
        y_main_list.append(main_hot)

        stars_hot = np.zeros(12, dtype=np.float32)
        for star in target_stars:
            stars_hot[int(star) - 1] = 1.0
        y_stars_list.append(stars_hot)

    X_seq = np.array(X_seq_list)
    X_agg = np.array(X_agg_list)
    y_main = np.array(y_main_list)
    y_stars = np.array(y_stars_list)

    print(f"✓ Created {len(X_seq)} samples")
    print(f"  Sequence input:  {X_seq.shape}  (samples x timesteps x {features_per_step} features)")
    print(f"  Aggregate input: {X_agg.shape}  (samples x 124 features)")
    print(f"  Main targets:    {y_main.shape}")
    print(f"  Stars targets:   {y_stars.shape}")

    return X_seq, X_agg, y_main, y_stars


# ═══════════════════════════════════════════════════════════════════════════════
# Model Architectures
# ═══════════════════════════════════════════════════════════════════════════════

def build_combined_model(seq_length, seq_features, agg_features):
    """
    Model A: Dual-input LSTM predicting both main numbers AND Lucky Stars.

    Sequence data (100 draws x 12 features) flows through stacked LSTMs.
    Aggregate data (124 features) flows through a dense layer.
    Both paths merge before splitting into two output heads.
    """
    print("\n=== Building Model A: Combined (Numbers + Stars) ===")

    # Sequence path -> LSTM
    seq_input = keras.Input(shape=(seq_length, seq_features), name='sequence_input')
    x = layers.LSTM(128, return_sequences=True, name='lstm_1')(seq_input)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, name='lstm_2')(x)
    x = layers.Dropout(0.3)(x)

    # Aggregate path -> Dense
    agg_input = keras.Input(shape=(agg_features,), name='aggregate_input')
    a = layers.Dense(64, activation='relu', name='agg_dense')(agg_input)
    a = layers.Dropout(0.2)(a)

    # Merge paths
    merged = layers.Concatenate(name='merge')([x, a])
    shared = layers.Dense(128, activation='relu', name='shared_1')(merged)
    shared = layers.Dropout(0.3)(shared)
    shared = layers.Dense(64, activation='relu', name='shared_2')(shared)

    # Output head: Main numbers (50 sigmoid outputs)
    main_h = layers.Dense(64, activation='relu', name='main_dense')(shared)
    main_output = layers.Dense(50, activation='sigmoid', name='main_output')(main_h)

    # Output head: Lucky Stars (12 sigmoid outputs)
    stars_h = layers.Dense(32, activation='relu', name='stars_dense')(shared)
    stars_output = layers.Dense(12, activation='sigmoid', name='stars_output')(stars_h)

    model = keras.Model(
        inputs=[seq_input, agg_input],
        outputs=[main_output, stars_output],
        name='combined_model',
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={'main_output': focal_loss(), 'stars_output': focal_loss()},
        metrics={'main_output': ['accuracy'], 'stars_output': ['accuracy']},
    )

    print(f"✓ Model created with {model.count_params():,} parameters")
    return model


def build_main_numbers_model(seq_length, seq_features, agg_features):
    """Model B: Specialist dual-input LSTM for main numbers (1-50) only."""
    print("\n=== Building Model B: Main Numbers Only ===")

    seq_input = keras.Input(shape=(seq_length, seq_features), name='sequence_input')
    x = layers.LSTM(128, return_sequences=True, name='lstm_1')(seq_input)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, name='lstm_2')(x)
    x = layers.Dropout(0.3)(x)

    agg_input = keras.Input(shape=(agg_features,), name='aggregate_input')
    a = layers.Dense(64, activation='relu', name='agg_dense')(agg_input)
    a = layers.Dropout(0.2)(a)

    merged = layers.Concatenate(name='merge')([x, a])
    h = layers.Dense(128, activation='relu', name='dense_1')(merged)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(64, activation='relu', name='dense_2')(h)
    h = layers.Dropout(0.2)(h)
    output = layers.Dense(50, activation='sigmoid', name='main_output')(h)

    model = keras.Model(
        inputs=[seq_input, agg_input],
        outputs=output,
        name='main_numbers_model',
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(),
        metrics=['accuracy'],
    )

    print(f"✓ Model created with {model.count_params():,} parameters")
    return model


def build_lucky_stars_model(seq_length, seq_features, agg_features):
    """Model C: Specialist dual-input LSTM for Lucky Stars (1-12) only."""
    print("\n=== Building Model C: Lucky Stars Only ===")

    seq_input = keras.Input(shape=(seq_length, seq_features), name='sequence_input')
    x = layers.LSTM(64, return_sequences=True, name='lstm_1')(seq_input)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32, name='lstm_2')(x)
    x = layers.Dropout(0.2)(x)

    agg_input = keras.Input(shape=(agg_features,), name='aggregate_input')
    a = layers.Dense(32, activation='relu', name='agg_dense')(agg_input)
    a = layers.Dropout(0.2)(a)

    merged = layers.Concatenate(name='merge')([x, a])
    h = layers.Dense(64, activation='relu', name='dense_1')(merged)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(32, activation='relu', name='dense_2')(h)
    output = layers.Dense(12, activation='sigmoid', name='stars_output')(h)

    model = keras.Model(
        inputs=[seq_input, agg_input],
        outputs=output,
        name='lucky_stars_model',
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(),
        metrics=['accuracy'],
    )

    print(f"✓ Model created with {model.count_params():,} parameters")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, X_train, y_train, X_val, y_val, model_name, save_path,
                epochs=EPOCHS):
    """
    Train with early stopping, learning rate reduction on plateau,
    and model checkpointing. Saves training history to JSON for
    later analysis and visualisation on the website.
    """
    print(f"\n--- Training {model_name} ---")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            save_path, monitor='val_loss', save_best_only=True, verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Persist training curves for analysis
    history_path = save_path.replace('.keras', '_history.json')
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✓ Training history saved to {history_path}")

    print(f"✓ Training complete for {model_name}")
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction & Uncertainty
# ═══════════════════════════════════════════════════════════════════════════════

def predict_with_uncertainty(model, inputs, n_samples=MC_DROPOUT_SAMPLES):
    """
    Monte Carlo dropout: run inference n_samples times with dropout active.

    The mean across runs gives a smoothed prediction.
    The standard deviation gives per-number uncertainty — high std means
    the model is unsure about that number slot. Low std means it's
    confident (whether correctly or not... this is a lottery after all).
    """
    tensor_inputs = [tf.constant(inp) for inp in inputs]

    all_preds = []
    for _ in range(n_samples):
        pred = model(tensor_inputs, training=True)
        if isinstance(pred, (list, tuple)):
            all_preds.append([p.numpy() for p in pred])
        else:
            all_preds.append(pred.numpy())

    if isinstance(all_preds[0], list):
        n_outputs = len(all_preds[0])
        means, stds = [], []
        for out_idx in range(n_outputs):
            stacked = np.array([p[out_idx] for p in all_preds])
            means.append(np.mean(stacked, axis=0))
            stds.append(np.std(stacked, axis=0))
        return means, stds
    else:
        stacked = np.array(all_preds)
        return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def get_probabilities(model, inputs, model_type='combined'):
    """
    Get MC dropout probability vectors for each output of a model.
    Returns dict with 'main' and/or 'stars' keys, each a (mean, std) tuple
    of 1D arrays (50 values for main, 12 for stars).
    """
    means, stds = predict_with_uncertainty(model, inputs)

    if model_type == 'combined':
        return {
            'main': (means[0][0], stds[0][0]),
            'stars': (means[1][0], stds[1][0]),
        }
    elif model_type == 'main':
        return {'main': (means[0], stds[0])}
    elif model_type == 'stars':
        return {'stars': (means[0], stds[0])}


def format_predictions(probs_dict):
    """Convert raw probability vectors into sorted number picks with metadata."""
    result = {}

    if 'main' in probs_dict:
        mean, std = probs_dict['main']
        top_5 = np.argsort(mean)[-5:][::-1]
        result['main_numbers'] = sorted([int(i + 1) for i in top_5])
        result['main_probabilities'] = [round(float(mean[i]), 4) for i in top_5]
        result['main_uncertainty'] = [round(float(std[i]), 4) for i in top_5]

    if 'stars' in probs_dict:
        mean, std = probs_dict['stars']
        top_2 = np.argsort(mean)[-2:][::-1]
        result['lucky_stars'] = sorted([int(i + 1) for i in top_2])
        result['lucky_star_probabilities'] = [round(float(mean[i]), 4) for i in top_2]
        result['lucky_star_uncertainty'] = [round(float(std[i]), 4) for i in top_2]

    return result


def ensemble_predictions(probs_combined, probs_main, probs_stars):
    """
    Blend predictions from all three models into a single set of picks.

    Specialist models (B, C) get higher weight than the combined model (A)
    because they focus all their capacity on one task and tend to produce
    sharper, more differentiated probability distributions.
    """
    w_s = ENSEMBLE_WEIGHT_SPECIALIST
    w_c = ENSEMBLE_WEIGHT_COMBINED

    combined_main_mean, combined_main_std = probs_combined['main']
    specialist_main_mean, specialist_main_std = probs_main['main']
    blended_main = w_c * combined_main_mean + w_s * specialist_main_mean
    blended_main_std = np.sqrt(
        w_c ** 2 * combined_main_std ** 2 + w_s ** 2 * specialist_main_std ** 2
    )

    combined_stars_mean, combined_stars_std = probs_combined['stars']
    specialist_stars_mean, specialist_stars_std = probs_stars['stars']
    blended_stars = w_c * combined_stars_mean + w_s * specialist_stars_mean
    blended_stars_std = np.sqrt(
        w_c ** 2 * combined_stars_std ** 2 + w_s ** 2 * specialist_stars_std ** 2
    )

    return format_predictions({
        'main': (blended_main, blended_main_std),
        'stars': (blended_stars, blended_stars_std),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_on_validation(model, X_seq_val, X_agg_val, y_main_val, y_stars_val,
                           model_type='combined'):
    """
    Lottery-specific evaluation: how many numbers does the model actually match?

    Random baselines for context:
        Main numbers: picking 5 from 50 -> expected 0.50 matches per draw
        Lucky Stars:  picking 2 from 12 -> expected 0.33 matches per draw

    If the model consistently beats these, it's found something in the data
    (even if that something is probably just overfitting on frequency bias).
    """
    print(f"\n--- Evaluation ({model_type}) ---")

    preds = model.predict([X_seq_val, X_agg_val], verbose=0)

    if model_type == 'combined':
        main_preds, stars_preds = preds
    elif model_type == 'main':
        main_preds, stars_preds = preds, None
    elif model_type == 'stars':
        main_preds, stars_preds = None, preds

    results = {}

    if main_preds is not None:
        matches = []
        for pred, actual in zip(main_preds, y_main_val):
            top_5 = set(np.argsort(pred)[-5:])
            actual_set = set(np.where(actual == 1)[0])
            matches.append(len(top_5 & actual_set))
        avg = np.mean(matches)
        dist = {str(i): int(matches.count(i)) for i in range(6)}
        results['main_numbers'] = {
            'avg_matched': round(float(avg), 3),
            'match_distribution': dist,
            'random_baseline': 0.5,
        }
        print(f"  Main: avg {avg:.2f}/5 matched  (random baseline: 0.50)")
        print(f"  Distribution: {dist}")

    if stars_preds is not None:
        matches = []
        for pred, actual in zip(stars_preds, y_stars_val):
            top_2 = set(np.argsort(pred)[-2:])
            actual_set = set(np.where(actual == 1)[0])
            matches.append(len(top_2 & actual_set))
        avg = np.mean(matches)
        dist = {str(i): int(matches.count(i)) for i in range(3)}
        results['lucky_stars'] = {
            'avg_matched': round(float(avg), 3),
            'match_distribution': dist,
            'random_baseline': 0.33,
        }
        print(f"  Stars: avg {avg:.2f}/2 matched  (random baseline: 0.33)")
        print(f"  Distribution: {dist}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("EuroMLions - TensorFlow Training (v2)")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading cleaned Kaggle data...")
    df = pd.read_csv('data/historical-kaggle-clean.csv')
    print(f"✓ Loaded {len(df)} draws")

    # ── Feature engineering ────────────────────────────────────────────────
    X_seq, X_agg, y_main, y_stars = engineer_features(df)

    # ── Chronological split (no shuffling — this is time series) ──────────
    split = int(TRAIN_SPLIT * len(X_seq))
    X_seq_train, X_seq_val = X_seq[:split], X_seq[split:]
    X_agg_train, X_agg_val = X_agg[:split], X_agg[split:]
    y_main_train, y_main_val = y_main[:split], y_main[split:]
    y_stars_train, y_stars_val = y_stars[:split], y_stars[split:]

    print(f"\n✓ Train: {len(X_seq_train)} samples | Val: {len(X_seq_val)} samples")

    seq_length = X_seq.shape[1]
    seq_features = X_seq.shape[2]
    agg_features = X_agg.shape[1]

    # ── Model A: Combined ─────────────────────────────────────────────────
    model_combined = build_combined_model(seq_length, seq_features, agg_features)
    train_model(
        model_combined,
        [X_seq_train, X_agg_train],
        {'main_output': y_main_train, 'stars_output': y_stars_train},
        [X_seq_val, X_agg_val],
        {'main_output': y_main_val, 'stars_output': y_stars_val},
        'Model A (Combined)',
        'models/model_combined.keras',
    )

    # ── Model B: Main Numbers ─────────────────────────────────────────────
    model_main = build_main_numbers_model(seq_length, seq_features, agg_features)
    train_model(
        model_main,
        [X_seq_train, X_agg_train], y_main_train,
        [X_seq_val, X_agg_val], y_main_val,
        'Model B (Main Numbers)',
        'models/model_main.keras',
    )

    # ── Model C: Lucky Stars ──────────────────────────────────────────────
    model_stars = build_lucky_stars_model(seq_length, seq_features, agg_features)
    train_model(
        model_stars,
        [X_seq_train, X_agg_train], y_stars_train,
        [X_seq_val, X_agg_val], y_stars_val,
        'Model C (Lucky Stars)',
        'models/model_stars.keras',
    )

    # ── Evaluate all models on validation set ─────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    eval_combined = evaluate_on_validation(
        model_combined, X_seq_val, X_agg_val, y_main_val, y_stars_val, 'combined',
    )
    eval_main = evaluate_on_validation(
        model_main, X_seq_val, X_agg_val, y_main_val, y_stars_val, 'main',
    )
    eval_stars = evaluate_on_validation(
        model_stars, X_seq_val, X_agg_val, y_main_val, y_stars_val, 'stars',
    )

    # ── Generate predictions with MC dropout ──────────────────────────────
    print("\n" + "=" * 60)
    print("Generating Predictions (Monte Carlo Dropout)")
    print("=" * 60)

    last_seq = X_seq[-1:]
    last_agg = X_agg[-1:]
    inputs = [last_seq, last_agg]

    probs_combined = get_probabilities(model_combined, inputs, 'combined')
    probs_main = get_probabilities(model_main, inputs, 'main')
    probs_stars = get_probabilities(model_stars, inputs, 'stars')

    pred_combined = format_predictions(probs_combined)
    pred_main = format_predictions(probs_main)
    pred_stars = format_predictions(probs_stars)
    pred_ensemble = ensemble_predictions(probs_combined, probs_main, probs_stars)

    # ── Save everything ───────────────────────────────────────────────────
    output = {
        'generated': datetime.now().isoformat(),
        'trained_on': f"{len(df)} draws (post-Sept 2016)",
        'sequence_length': SEQUENCE_LENGTH,
        'ensemble': pred_ensemble,
        'model_A_combined': pred_combined,
        'model_B_main_only': pred_main,
        'model_C_stars_only': pred_stars,
        'evaluation': {
            'model_A_combined': eval_combined,
            'model_B_main_only': eval_main,
            'model_C_stars_only': eval_stars,
        },
    }

    with open('data/predictions.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n✓ Predictions saved to data/predictions.json")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎯 ENSEMBLE (blended from all models):")
    print(f"   Main numbers: {pred_ensemble['main_numbers']}")
    print(f"   Lucky Stars:  {pred_ensemble['lucky_stars']}")
    print(f"   Uncertainty:  main {pred_ensemble['main_uncertainty']}")
    print(f"                 stars {pred_ensemble['lucky_star_uncertainty']}")
    print("\n🎰 Model A (Combined):")
    print(f"   Main: {pred_combined['main_numbers']}  Stars: {pred_combined['lucky_stars']}")
    print("\n🎲 Model B (Main Only):")
    print(f"   Main: {pred_main['main_numbers']}")
    print("\n⭐ Model C (Stars Only):")
    print(f"   Stars: {pred_stars['lucky_stars']}")
    print("=" * 60)

    print("\n✓✓✓ All done! Models trained, evaluated, and predictions generated ✓✓✓")
    print("\nReminder: These predictions are confidently wrong. That's the whole thing. 😂")


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    main()
