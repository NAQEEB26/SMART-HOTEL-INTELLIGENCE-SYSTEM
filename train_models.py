"""
Smart Hotel Intelligence System - Model Training Script
=======================================================
Trains two ML models on the hotel booking dataset:
  1. Cancellation Predictor  (Random Forest Classifier)
  2. Revenue Estimator       (Random Forest Regressor)
Saves trained models + encoders + metadata to the model/ directory.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Month name → number mapping (matches CSV values) ──────────────────────────
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}


# ── Full dataset download URL (GitHub Release asset) ──────────────────────────
DATASET_URL = "https://github.com/NAQEEB26/SMART-HOTEL-INTELLIGENCE-SYSTEM/releases/download/v1.0/hotel_booking.csv"


def download_full_dataset(dest='hotel_booking.csv'):
    """Download full dataset from GitHub Release if not present locally."""
    try:
        import urllib.request
        print(f"Downloading full dataset from GitHub Release...")
        urllib.request.urlretrieve(DATASET_URL, dest)
        print(f"✅ Downloaded to {dest}")
        return dest
    except Exception as e:
        print(f"⚠️  Download failed: {e}")
        return None


def find_data_file():
    # Check local paths first
    for p in [
        'hotel_booking.csv',
        'data/hotel_booking.csv',
        '../hotel_booking.csv',
    ]:
        if os.path.exists(p):
            return p

    # Try downloading full dataset from GitHub Release
    result = download_full_dataset('hotel_booking.csv')
    if result and os.path.exists(result):
        return result

    # Last fallback: sample CSV
    for p in ['data/hotel_booking_sample.csv', '../data/hotel_booking_sample.csv']:
        if os.path.exists(p):
            return p

    return None


def train_and_save_models(data_path=None, model_dir='model'):
    """
    Main training function.
    Returns (cancel_accuracy, adr_r2, adr_mae).
    """
    # ── Locate dataset ─────────────────────────────────────────────────────────
    if data_path is None:
        data_path = find_data_file()
    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(
            "Dataset not found. Expected hotel_booking.csv or data/hotel_booking_sample.csv"
        )

    os.makedirs(model_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading dataset from '{data_path}' ...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} records x {df.shape[1]} columns")

    # ── Drop PII and data-leakage columns ──────────────────────────────────────
    drop_cols = [
        'name', 'email', 'phone-number', 'credit_card',
        'reservation_status', 'reservation_status_date'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Handle missing values ──────────────────────────────────────────────────
    df['children'] = df['children'].fillna(0)
    df['agent']    = df['agent'].fillna(0)
    df['company']  = df['company'].fillna(0)
    df['country']  = df['country'].fillna('Unknown')

    # Cap extreme ADR outliers (keep 99th percentile)
    adr_cap = df['adr'].quantile(0.99)
    df['adr'] = df['adr'].clip(0, adr_cap)

    # ── Feature engineering ────────────────────────────────────────────────────
    df['arrival_month_num'] = df['arrival_date_month'].map(MONTH_MAP)
    df['total_guests']      = df['adults'] + df['children'] + df['babies']
    df['total_nights']      = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    # ── Categorical encoding ───────────────────────────────────────────────────
    cat_cols = [
        'hotel', 'meal', 'market_segment', 'distribution_channel',
        'reserved_room_type', 'assigned_room_type',
        'deposit_type', 'customer_type', 'country'
    ]

    label_encoders   = {}
    cat_unique_values = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        label_encoders[col]    = le
        cat_unique_values[col] = sorted(df[col].astype(str).unique().tolist())

    # ── Define feature sets ────────────────────────────────────────────────────
    cancel_features = [
        'hotel_enc', 'lead_time', 'arrival_month_num',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'total_nights',
        'adults', 'children', 'babies', 'total_guests',
        'meal_enc', 'market_segment_enc', 'deposit_type_enc', 'customer_type_enc',
        'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'days_in_waiting_list', 'total_of_special_requests',
        'required_car_parking_spaces', 'adr'
    ]

    adr_features = [
        'hotel_enc', 'lead_time', 'arrival_month_num',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'total_nights',
        'adults', 'children', 'babies', 'total_guests',
        'meal_enc', 'market_segment_enc', 'deposit_type_enc', 'customer_type_enc',
        'reserved_room_type_enc', 'is_repeated_guest',
        'booking_changes', 'total_of_special_requests', 'required_car_parking_spaces'
    ]

    # ── Model 1 : Cancellation Prediction (Classification) ────────────────────
    print("\nTraining Model 1: Cancellation Predictor ...")
    X_c = df[cancel_features]
    y_c = df['is_canceled']

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
    )

    cancel_model = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    cancel_model.fit(Xc_tr, yc_tr)

    cancel_acc = accuracy_score(yc_te, cancel_model.predict(Xc_te))
    print(f"  Accuracy : {cancel_acc * 100:.2f}%")

    cancel_importances = dict(zip(cancel_features, cancel_model.feature_importances_))

    # ── Model 2 : ADR / Revenue Prediction (Regression) ───────────────────────
    print("\nTraining Model 2: Revenue Estimator ...")
    df_adr = df[(df['adr'] > 10)].copy()
    X_a = df_adr[adr_features]
    y_a = df_adr['adr']

    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        X_a, y_a, test_size=0.2, random_state=42
    )

    adr_model = RandomForestRegressor(
        n_estimators=200, max_depth=20,
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    adr_model.fit(Xa_tr, ya_tr)

    ya_pred  = adr_model.predict(Xa_te)
    adr_r2   = r2_score(ya_te, ya_pred)
    adr_mae  = mean_absolute_error(ya_te, ya_pred)
    print(f"  R² Score : {adr_r2:.4f}")
    print(f"  MAE      : €{adr_mae:.2f} per night")

    adr_importances = dict(zip(adr_features, adr_model.feature_importances_))

    # ── Save everything ────────────────────────────────────────────────────────
    print(f"\nSaving models to '{model_dir}/' ...")

    with open(f'{model_dir}/cancel_model.pkl', 'wb') as f:
        pickle.dump(cancel_model, f)
    with open(f'{model_dir}/adr_model.pkl', 'wb') as f:
        pickle.dump(adr_model, f)
    with open(f'{model_dir}/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    feature_info = {
        'cancel_features'    : cancel_features,
        'adr_features'       : adr_features,
        'cancel_accuracy'    : cancel_acc,
        'adr_r2'             : adr_r2,
        'adr_mae'            : adr_mae,
        'month_map'          : MONTH_MAP,
        'cat_unique_values'  : cat_unique_values,
        'cancel_importances' : cancel_importances,
        'adr_importances'    : adr_importances,
        'data_stats': {
            'total_records' : int(len(df)),
            'cancel_rate'   : float(df['is_canceled'].mean()),
            'avg_adr'       : float(df['adr'].mean()),
            'avg_lead_time' : float(df['lead_time'].mean()),
        }
    }

    with open(f'{model_dir}/feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)

    print("\n✅ All models saved successfully!")
    print(f"   Cancellation Model → Accuracy : {cancel_acc * 100:.2f}%")
    print(f"   Revenue Model      → R² : {adr_r2:.4f}  |  MAE : €{adr_mae:.2f}")

    return cancel_acc, adr_r2, adr_mae


if __name__ == '__main__':
    train_and_save_models()
