# Modular Time Series Forecasting Pipeline

This repository provides a robust, entity-agnostic forecasting pipeline for time series problems. It supports univariate and multivariate modeling, multiple model frameworks, configurable backtesting, and entity-level model selection.

**Key Features:**

- **Entity-Agnostic:** Use any column as the key (e.g., store, product, region).
- **Flexible Frequency:** Supports daily, weekly, or monthly data.
- **Univariate & Multivariate:** Add external regressors (e.g., promo, price).
- **Config-Driven:** All modeling, hyperparams, and backtest windows are in `config.py`.
- **Model Zoo:** ARIMA, ExponentialSmoothing, StatsForecast (AutoARIMA, ETS, Theta), Prophet, MLForecast (LightGBM, RF, CatBoost), AutoGluon (RF, LGBM, CatBoost, TFT).
- **Multiple Modes:** `"testing"` for model selection, `"training-inferencing"` for production.
- **Backtesting:** Supports multiple custom backtest windows.

---

## 🚀 Quickstart

### 1. Open in Codespaces or clone locally

```bash

cd modular-forecasting-pipeline
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Inspect/Edit Config

- All modeling settings, keys, features, and backtest windows are in `config.py`.
- Use `key_column` to set your entity.

### 4. Run Model Selection (Testing Mode)

```bash
python main.py
```

- Outputs best models and backtest MAEs per entity in `best_models.csv` and `backtest_scores.csv`.

### 5. (Optional) Override Models for Production

- Edit `production_models` in `config.py` to force your own choices (`key: model_tag`).

### 6. Run Final Training & Inference

- Set `"pipeline_mode": "training-inferencing"` in `config.py`.
- Run again:

```bash
python main.py
```

---

## 📁 File Tree

```
.
├── config.py
├── data
│   └── sales.csv
├── main.py
├── models
│   ├── autogluon_wrapper.py
│   ├── mlforecast_wrapper.py
│   ├── prophet_wrapper.py
│   ├── statsforecast_wrapper.py
│   └── statsmodels_wrapper.py
├── pipeline
│   ├── backtest.py
│   ├── infer.py
│   └── train.py
├── requirements.txt
└── utils
    ├── evaluation.py
    └── plot.py
```

---

## 🗂️ Sample Data (`data/sales.csv`)

```csv
week,key,sales,promo,price
2024-01-01,store_1,870,0,10.2
2024-01-01,store_2,940,1,11.1
2024-01-01,store_3,820,0,9.8
2024-01-01,store_4,780,1,9.9
2024-01-01,store_5,900,0,10.3
2024-01-08,store_1,880,1,10.5
2024-01-08,store_2,950,0,11.3
2024-01-08,store_3,830,1,10.1
2024-01-08,store_4,790,0,10.2
2024-01-08,store_5,910,1,10.6
2024-01-15,store_1,890,0,10.7
2024-01-15,store_2,960,1,11.5
2024-01-15,store_3,840,0,10.3
2024-01-15,store_4,800,1,10.4
2024-01-15,store_5,920,0,10.8
2024-01-22,store_1,900,1,10.9
2024-01-22,store_2,970,0,11.7
2024-01-22,store_3,850,1,10.5
2024-01-22,store_4,810,0,10.6
2024-01-22,store_5,930,1,11.0
2024-01-29,store_1,910,0,11.1
2024-01-29,store_2,980,1,11.9
2024-01-29,store_3,860,0,10.7
2024-01-29,store_4,820,1,10.8
2024-01-29,store_5,940,0,11.2
2024-02-05,store_1,920,1,11.3
2024-02-05,store_2,990,0,12.1
2024-02-05,store_3,870,1,10.9
2024-02-05,store_4,830,0,11.0
2024-02-05,store_5,950,1,11.4
2024-02-12,store_1,930,0,11.5
2024-02-12,store_2,1000,1,12.3
2024-02-12,store_3,880,0,11.1
2024-02-12,store_4,840,1,11.2
2024-02-12,store_5,960,0,11.6
```

---

## 🛠️ Sample Config (`config.py`)

```python
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
```

---

```python name=requirements.txt
pandas
numpy
matplotlib
scikit-learn
statsmodels
statsforecast
mlforecast
lightgbm
catboost
autogluon
prophet
```

---

```python name=main.py
import pandas as pd
from config import config
from pipeline.backtest import backtest_models
from pipeline.train import train_final_model
from pipeline.infer import forecast_with_model
from utils.plot import plot_forecasts

def run_testing(df, config):
    scores, best_model_by_key = backtest_models(df, config)
    print("Backtest scores (MAE) by key and model:")
    for key, model_scores in scores.items():
        print(f"{key}:")
        for k, v in model_scores.items():
            print(f"  {k}: {v}")
    print("Best model per key (by avg MAE):", best_model_by_key)
    pd.DataFrame(scores).to_csv("backtest_scores.csv")
    pd.Series(best_model_by_key).to_csv("best_models.csv")
    return best_model_by_key

def run_training_inferencing(df, config, best_model_by_key):
    use_models = config["production_models"] if config["production_models"] else best_model_by_key
    print("Using the following models for final training/inferencing:")
    print(use_models)
    train_df = df
    key_models = train_final_model(train_df, config, use_models)
    forecast_inputs = {}
    for key in df[config["key_column"]].unique():
        forecast_inputs[key] = df[df[config["key_column"]] == key]
    forecasts = forecast_with_model(key_models, forecast_inputs, config)
    plot_forecasts(df, forecasts, config)

def main():
    df = pd.read_csv("data/sales.csv", parse_dates=[config["date_column"]])
    if config["pipeline_mode"] == "testing":
        best_model_by_key = run_testing(df, config)
    elif config["pipeline_mode"] == "training-inferencing":
        try:
            best_model_by_key = pd.read_csv("best_models.csv", index_col=0, squeeze=True).to_dict()
        except Exception:
            print("Warning: No best models found, running testing phase first.")
            best_model_by_key = run_testing(df, config)
        run_training_inferencing(df, config, best_model_by_key)
    else:
        raise ValueError("Unknown pipeline_mode in config.")

if __name__ == "__main__":
    main()
```


---

