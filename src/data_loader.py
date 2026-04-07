import pandas as pd
import numpy as np
import torch
import pickle

import pandas as pd
import numpy as np
import torch
import pickle

def get_market_data_for_date(csv_path, scaler_path, target_date=None, lookback=30, expected_features=47):
    """
    Fetches data up to a specific target_date to simulate Point-in-Time forecasting.
    """
    # 1. Load the data
    df = pd.read_csv(csv_path)
    
    # --- THE FIX: FORWARD FILL MISSING DATA ---
    TARGET_COLS = ['target_ret_1d', 'target_dir_1d', 'target_quintile_1d']
    feature_cols_to_check = [col for col in df.columns if col not in TARGET_COLS]
    
    # 1. If Yahoo returned a blank cell for a feature, borrow the previous day's number
    df[feature_cols_to_check] = df[feature_cols_to_check].ffill()
    
    # 2. Now it is safe to drop any rows that are fundamentally broken
    df = df.dropna(subset=feature_cols_to_check).reset_index(drop=True)
    # ------------------------------------------
    # --------------------------------------------------
    
    # Ensure date column is datetime for accurate filtering
    df['date'] = pd.to_datetime(df['date'])
    
    # --- TIME MACHINE LOGIC ---
    if target_date:
        target_date = pd.to_datetime(target_date)
        df = df[df['date'] <= target_date]
        
        if len(df) < lookback:
            raise ValueError(f"Not enough historical data before {target_date.date()} to create a {lookback}-day sequence.")
    
    # ... (Keep the rest of the file exactly the same from here down!)
    # --------------------------

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    TARGET_COLS = ['target_ret_1d', 'target_dir_1d', 'target_quintile_1d']
    CATEGORICAL_COLS = ['regime_200MA', 'vol_quartile_63d', 'is_monday', 'is_friday', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos']
    CATEGORICAL_COLS = [col for col in CATEGORICAL_COLS if col in df.columns]
    
    if hasattr(scaler, 'feature_names_in_'):
        NUMERICAL_COLS = [col for col in list(scaler.feature_names_in_) if col in df.columns]
    else:
        NUMERICAL_COLS = [col for col in df.columns if col not in ['date'] + TARGET_COLS + CATEGORICAL_COLS]

    # Get the 30 days leading up to (and including) the target date
    recent_data = df.tail(lookback).copy()
    
    latest_regime_200 = recent_data.iloc[-1]['regime_200MA']
    latest_vol_quartile = recent_data.iloc[-1]['vol_quartile_63d']
    latest_date_str = recent_data.iloc[-1]['date'].strftime('%Y-%m-%d') # Format for UI

    # Scale and Extract
    recent_data[NUMERICAL_COLS] = scaler.transform(recent_data[NUMERICAL_COLS])
    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X_features = recent_data[feature_cols].values 
    
    # Production Fail-Safe (Padding)
    current_features = X_features.shape[1]
    if current_features < expected_features:
        missing_count = expected_features - current_features
        padding = np.zeros((X_features.shape[0], missing_count))
        X_features = np.hstack((X_features, padding))
    elif current_features > expected_features:
        X_features = X_features[:, :expected_features]
    
    # Format Tensors
    X_today_2d = X_features[-1].reshape(1, -1)
    X_window_3d = torch.tensor(X_features.reshape(1, lookback, expected_features), dtype=torch.float32)
    
    return {
        "date": latest_date_str,
        "X_2d": X_today_2d,
        "X_3d": X_window_3d,
        "regime_200": latest_regime_200,
        "vol_quartile": latest_vol_quartile
    }
