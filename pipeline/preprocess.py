def preprocess_data(df):
    # Fill NA, generate lags, features, etc.
    df = df.fillna(method='ffill')
    # Feature engineering for ML models
    df['month'] = df.index.month
    df['lag_1'] = df['sales'].shift(1)
    df = df.dropna()
    return df