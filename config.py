config = {
    "key_column": "key",
    "target_column": "sales",
    "date_column": "week",
    "extra_features": ["promo", "price"],
    "mode": "multivariate",           # "univariate" or "multivariate"
    "frequency": "W",                 # "D", "W", "M"
    "pipeline_mode": "testing",       # "testing" or "training-inferencing"
    "production_models": {},          # Fill after testing to override best models
    "backtests": [
        {"train_end": "2024-01-29", "forecast_horizon": 2},
        {"train_end": "2024-02-05", "forecast_horizon": 1}
    ],
    "models": {
        "statsmodels": [
            {"model": "arima", "params": {"order": (1,1,0)}},
            {"model": "ExponentialSmoothing", "params": {"trend": "add", "seasonal": "add", "seasonal_periods": 4}}
        ],
        "statsforecast": [
            {"model": "AutoARIMA", "params": {}},
            {"model": "ETS", "params": {"model": "additive"}},
            {"model": "Theta", "params": {}}
        ],
        "prophet": [
            {
                "model": "Prophet",
                "params": {
                    "seasonality_mode": "additive",
                    "yearly_seasonality": 2,
                    "weekly_seasonality": 1,
                    "daily_seasonality": 0
                }
            }
        ],
        "mlforecast": [
            {"model": "LightGBM", "params": {"n_estimators": 10}},
            {"model": "RandomForest", "params": {"n_estimators": 10}},
            {"model": "CatBoost", "params": {"iterations": 10, "verbose": 0}}
        ],
        "autogluon": [
            {"model": "RF", "params": {}},
            {"model": "LightGBM", "params": {}},
            {"model": "CatBoost", "params": {}},
            {"model": "TFT", "params": {"epochs": 2, "prediction_length": 2}}
        ]
    }
}