import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_technical_features(df):
    """Applies the exact math from your 02_feature_engineering.ipynb"""
    # 2. Multi-Horizon Log Returns
    for d in [1, 2, 3, 5, 10, 21]:
        df[f'log_ret_{d}d'] = np.log(df['close'] / df['close'].shift(d))
        
    # 3. Market Microstructure
    df['hl_pct'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['oc_pct'] = (df['close'] - df['open']) / df['open']
    df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # 4. Volatility Estimators (The VIX Edge)
    for w in [5, 10, 21]:
        df[f'rvol_{w}d'] = df['log_ret_1d'].rolling(w).std() * np.sqrt(252)
        rs = (1.0 / (4.0 * np.log(2.0))) * ((np.log(df['high'] / df['low']))**2)
        df[f'parkinson_{w}d'] = np.sqrt(rs.rolling(w).mean()) * np.sqrt(252)
        
    df['vix_zscore_21d'] = (df['india_vix'] - df['india_vix'].rolling(21).mean()) / (df['india_vix'].rolling(21).std() + 1e-8)
    df['vix_ma_ratio_5_21'] = df['india_vix'].rolling(5).mean() / df['india_vix'].rolling(21).mean()
    
    # 5. Technical & Momentum Indicators
    for ma in [10, 20, 50, 200]:
        df[f'sma_{ma}'] = df['close'].rolling(ma).mean()
        df[f'price_vs_sma_{ma}'] = df['close'] / df[f'sma_{ma}'] - 1
        
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    df['tr'] = np.maximum((df['high'] - df['low']), 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr_14'] = df['tr'].rolling(14).mean() / df['close']
    
    # 6. Rolling Memory
    df['drawdown_20d'] = df['close'] / df['close'].rolling(20).max() - 1
    
    def calc_autocorr(x):
        if np.isnan(x).any() or len(x) < 2: return np.nan
        return pd.Series(x).autocorr(lag=1)
    df['autocorr_1d_20d'] = df['log_ret_1d'].rolling(20).apply(calc_autocorr, raw=True)
    
    # 7. Cyclic Calendar
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 5)
    
    # 8. Regime Detection
    df['regime_200MA'] = (df['close'] > df['sma_200']).astype(int)
    
    def calc_quartile(x):
        if np.isnan(x).any(): return np.nan
        bins = np.percentile(x, [25, 50, 75])
        return np.digitize(x[-1], bins)
    df['vol_quartile_63d'] = df['rvol_21d'].rolling(63).apply(calc_quartile, raw=True)
    
    # 9. Targets (DO NOT DROP THE LAST ROW FOR LIVE DATA)
    df['target_ret_1d'] = df['log_ret_1d'].shift(-1)
    df['target_dir_1d'] = (df['target_ret_1d'] > 0).astype(float) # Changed to float to handle NaNs
    
    def calc_quintile(x):
        if np.isnan(x).any(): return np.nan
        bins = np.percentile(x, [20, 40, 60, 80])
        return np.digitize(x[-1], bins)
    df['target_quintile_1d'] = df['target_ret_1d'].rolling(252).apply(calc_quintile, raw=True)
    
    return df

def update_dataset(csv_path="E:/fourth_sem/nifty_ml_hybrid/datasets/processed/nifty_engineered_features.csv"):
    print("--- NIFTY 50 Quant Pipeline Updater ---")
    
    # 1. Load existing data to find the last updated date
    try:
        df_existing = pd.read_csv(csv_path)
        df_existing['date'] = pd.to_datetime(df_existing['date'])
        last_date = df_existing['date'].max()
        print(f"Current CSV ends on: {last_date.strftime('%Y-%m-%d')}")
    except FileNotFoundError:
        print(f"Error: Could not find CSV at {csv_path}")
        return

    if last_date.date() >= datetime.date.today():
        print("Dataset is already fully up to date for today!")
        return

    # 2. Download Live Data (Nifty + India VIX)
    print("Fetching live market data (300-day lookback for MAs)...")
    nifty = yf.Ticker("^NSEI").history(period="300d").reset_index()
    vix = yf.Ticker("^INDIAVIX").history(period="300d").reset_index()
    
    # Standardize column names to match your notebook
    nifty = nifty[['Date', 'Open', 'High', 'Low', 'Close']]
    nifty.columns = ['date', 'open', 'high', 'low', 'close']
    vix = vix[['Date', 'Close']]
    vix.columns = ['date', 'india_vix']
    
    # Merge and clean dates - Use LEFT join so missing VIX doesn't delete Nifty data!
    df_live = pd.merge(nifty, vix, on='date', how='left')
    df_live['india_vix'] = df_live['india_vix'].ffill() # Forward-fill missing VIX
    df_live['date'] = pd.to_datetime(df_live['date']).dt.tz_localize(None)
    
    # 3. Apply Feature Engineering
    print("Calculating Technical & Microstructure features...")
    df_live = calculate_technical_features(df_live)
    
    # 4. Filter only the NEW rows that aren't in your CSV yet
    new_rows = df_live[df_live['date'] > last_date].copy()
    
    if new_rows.empty:
        print("No new trading days found to append.")
        return
        
    # 5. Append and Save
    df_final = pd.concat([df_existing, new_rows], ignore_index=True)
    df_final.to_csv(csv_path, index=False)
    
    print(f"✅ Success! Appended {len(new_rows)} new day(s) to the dataset.")
    print(f"New CSV end date: {df_final['date'].max().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    update_dataset()