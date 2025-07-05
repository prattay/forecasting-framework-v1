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

```python name=models/statsmodels_wrapper.py
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fit_arima(y, order):
    model = ARIMA(y, order=order)
    fitted = model.fit()
    return fitted

def arima_predict(fitted, periods):
    return fitted.forecast(steps=periods)

def fit_expsmooth(y, trend=None, seasonal=None, seasonal_periods=None):
    model = ExponentialSmoothing(y, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fitted = model.fit()
    return fitted

def expsmooth_predict(fitted, periods):
    return fitted.forecast(periods)
```

---

```python name=models/statsforecast_wrapper.py
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, ETS, Theta

def fit_statsforecast(df, key_col, date_col, target_col, model_name, params, freq="D"):
    df = df.rename(columns={date_col: "ds", key_col: "unique_id", target_col: "y"})
    if model_name == "AutoARIMA":
        models = [AutoARIMA(**params)]
    elif model_name == "ETS":
        models = [ETS(**params)]
    elif model_name == "Theta":
        models = [Theta(**params)]
    else:
        raise ValueError("Unknown model for statsforecast")
    sf = StatsForecast(models=models, freq=freq)
    sf.fit(df)
    return sf

def statsforecast_predict(sf, h, model_name, key):
    pred = sf.predict(h)
    return pred[pred["unique_id"] == key][model_name].values
```

---

```python name=models/mlforecast_wrapper.py
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

def get_ml_model(model_name, params):
    if model_name == "LightGBM":
        return LGBMRegressor(**params)
    elif model_name == "RandomForest":
        return RandomForestRegressor(**params)
    elif model_name == "CatBoost":
        return CatBoostRegressor(**params)
    else:
        raise ValueError(f"Unknown ML model: {model_name}")

def fit_mlforecast(df, key_col, date_col, target_col, model_name, params, freq="D", extra_features=None):
    lags = [1, 2]
    model = get_ml_model(model_name, params)
    fc = MLForecast(models=[model], freq=freq, lags=lags)
    fc.fit(df, id_col=key_col, time_col=date_col, target_col=target_col, static_features=extra_features)
    return fc

def mlforecast_predict(fc, h, key_col, model_name, key):
    pred = fc.predict(h)
    return pred[pred[key_col] == key][model_name].values
```

---

```python name=models/prophet_wrapper.py
from prophet import Prophet
import pandas as pd

def fit_prophet(df, params, target_col, date_col, extra_features=None):
    df_cp = df[[date_col, target_col] + (extra_features or [])].copy()
    df_cp.rename(columns={date_col: "ds", target_col: "y"}, inplace=True)
    m = Prophet(
        seasonality_mode=params.get("seasonality_mode", "additive"),
        yearly_seasonality=params.get("yearly_seasonality", 10),
        weekly_seasonality=params.get("weekly_seasonality", 3),
        daily_seasonality=params.get("daily_seasonality", 1)
    )
    if extra_features:
        for feat in extra_features:
            m.add_regressor(feat)
    m.fit(df_cp)
    return m

def prophet_predict(m, future_df, periods, date_col, extra_features=None):
    last_date = future_df[date_col].max()
    freq = pd.infer_freq(future_df[date_col])
    future = pd.date_range(last_date, periods=periods+1, freq=freq)[1:]
    future_df_new = pd.DataFrame({date_col: future})
    if extra_features:
        for feat in extra_features:
            future_df_new[feat] = future_df[feat].iloc[-1]
    future_df_new.rename(columns={date_col: "ds"}, inplace=True)
    forecast = m.predict(future_df_new)
    return forecast["yhat"].values
```

---

```python name=models/autogluon_wrapper.py
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor

def fit_autogluon(train_df, target_col, problem_type="regression", model_name=None, params=None, freq="D"):
    if model_name == "TFT":
        predictor = TimeSeriesPredictor(
            target=target_col,
            prediction_length=params.get("prediction_length", 3),
            freq=freq,
            eval_metric="MAE"
        ).fit(train_df, presets="medium_quality", hyperparameters={"TFT": params})
        return predictor
    else:
        predictor = TabularPredictor(label=target_col, problem_type=problem_type).fit(
            train_df,
            presets="best_quality",
            hyperparameters={model_name: params or {}}
        )
        return predictor

def autogluon_predict(predictor, test_df, model_name=None):
    if model_name == "TFT":
        return predictor.predict(test_df)
    else:
        return predictor.predict(test_df)
```

---

```python name=pipeline/backtest.py
import pandas as pd
from models.statsmodels_wrapper import fit_arima, arima_predict, fit_expsmooth, expsmooth_predict
from models.statsforecast_wrapper import fit_statsforecast, statsforecast_predict
from models.prophet_wrapper import fit_prophet, prophet_predict
from models.mlforecast_wrapper import fit_mlforecast, mlforecast_predict
from models.autogluon_wrapper import fit_autogluon, autogluon_predict
from utils.evaluation import evaluate_forecast

def backtest_models(df, config):
    scores = {}
    best_model_by_key = {}
    keys = df[config["key_column"]].unique()
    features = config["extra_features"] if config["mode"] == "multivariate" else []
    backtests = config["backtests"]

    for key in keys:
        key_df = df[df[config["key_column"]] == key]
        model_maes = {}

        for bt in backtests:
            train = key_df[key_df[config["date_column"]] <= bt["train_end"]]
            test = key_df[(key_df[config["date_column"]] > bt["train_end"])][:bt["forecast_horizon"]]
            y_train = train[config["target_column"]]
            y_test = test[config["target_column"]]
            # statsmodels
            for m in config["models"].get("statsmodels", []):
                tag = f"{m['model']}_{m['params']}_bt_{bt['train_end']}_{bt['forecast_horizon']}"
                try:
                    if m["model"].lower() == "arima":
                        model = fit_arima(y_train, **m["params"])
                        pred = arima_predict(model, len(test))
                    elif m["model"].lower() == "exponentialsmoothing":
                        model = fit_expsmooth(y_train, **m["params"])
                        pred = expsmooth_predict(model, len(test))
                    else:
                        continue
                    model_maes[tag] = evaluate_forecast(y_test, pred)
                except Exception:
                    model_maes[tag] = float("inf")
            # statsforecast
            for m in config["models"].get("statsforecast", []):
                tag = f"{m['model']}_{m['params']}_bt_{bt['train_end']}_{bt['forecast_horizon']}"
                try:
                    sf = fit_statsforecast(train, config["key_column"], config["date_column"], config["target_column"], m["model"], m["params"], freq=config["frequency"])
                    pred = statsforecast_predict(sf, len(test), m["model"], key)
                    model_maes[tag] = evaluate_forecast(y_test, pred)
                except Exception:
                    model_maes[tag] = float("inf")
            # prophet
            for m in config["models"].get("prophet", []):
                tag = f"prophet_{m['params']}_bt_{bt['train_end']}_{bt['forecast_horizon']}"
                try:
                    mp = fit_prophet(train, m["params"], config["target_column"], config["date_column"], features if features else None)
                    pred = prophet_predict(mp, test, len(test), config["date_column"], features if features else None)
                    model_maes[tag] = evaluate_forecast(y_test, pred)
                except Exception:
                    model_maes[tag] = float("inf")
            # mlforecast
            for m in config["models"].get("mlforecast", []):
                tag = f"{m['model']}_{m['params']}_bt_{bt['train_end']}_{bt['forecast_horizon']}"
                try:
                    fc = fit_mlforecast(train, config["key_column"], config["date_column"], config["target_column"], m["model"], m["params"], freq=config["frequency"], extra_features=features)
                    pred = mlforecast_predict(fc, len(test), config["key_column"], m["model"], key)
                    model_maes[tag] = evaluate_forecast(y_test, pred)
                except Exception:
                    model_maes[tag] = float("inf")
            # autogluon
            for m in config["models"].get("autogluon", []):
                tag = f"autogluon_{m['model']}_{m['params']}_bt_{bt['train_end']}_{bt['forecast_horizon']}"
                try:
                    ag = fit_autogluon(train, config["target_column"], model_name=m["model"], params=m["params"], freq=config["frequency"])
                    pred = autogluon_predict(ag, test.drop(columns=[config["target_column"]]), model_name=m["model"])
                    model_maes[tag] = evaluate_forecast(y_test, pred)
                except Exception:
                    model_maes[tag] = float("inf")
        avg_maes = {}
        for model_key in set(k.rsplit("_bt_", 1)[0] for k in model_maes):
            vals = [v for k, v in model_maes.items() if k.startswith(model_key)]
            avg_maes[model_key] = sum(vals) / len(vals)
        best_model = min(avg_maes, key=avg_maes.get)
        best_model_by_key[key] = best_model
        scores[key] = avg_maes
    return scores, best_model_by_key
```

---

```python name=pipeline/train.py
from models.statsmodels_wrapper import fit_arima, fit_expsmooth
from models.statsforecast_wrapper import fit_statsforecast
from models.prophet_wrapper import fit_prophet
from models.mlforecast_wrapper import fit_mlforecast
from models.autogluon_wrapper import fit_autogluon

def train_final_model(train_df, config, best_model_by_key):
    features = config["extra_features"] if config["mode"] == "multivariate" else []
    key_models = {}
    for key, model_tag in best_model_by_key.items():
        key_train = train_df[train_df[config["key_column"]] == key]
        if model_tag.startswith("arima_"):
            params = eval(model_tag[len("arima_"):].split("_bt_")[0])
            model = fit_arima(key_train[config["target_column"]], **params)
        elif model_tag.startswith("ExponentialSmoothing_"):
            params = eval(model_tag[len("ExponentialSmoothing_"):].split("_bt_")[0])
            model = fit_expsmooth(key_train[config["target_column"]], **params)
        elif model_tag.startswith("AutoARIMA_") or model_tag.startswith("ETS_") or model_tag.startswith("Theta_"):
            parts = model_tag.split("_", 1)
            model = fit_statsforecast(key_train, config["key_column"], config["date_column"], config["target_column"], parts[0], eval(parts[1].split("_bt_")[0]), freq=config["frequency"])
        elif model_tag.startswith("prophet_"):
            params = eval(model_tag[len("prophet_"):].split("_bt_")[0])
            model = fit_prophet(key_train, params, config["target_column"], config["date_column"], features if features else None)
        elif model_tag.startswith("LightGBM_") or model_tag.startswith("RandomForest_") or model_tag.startswith("CatBoost_"):
            msplit = model_tag.split("_", 1)
            mname = msplit[0]
            params = eval(msplit[1].split("_bt_")[0])
            model = fit_mlforecast(key_train, config["key_column"], config["date_column"], config["target_column"], mname, params, freq=config["frequency"], extra_features=features)
        elif model_tag.startswith("autogluon_"):
            msplit = model_tag.split("_", 2)
            mname = msplit[1]
            params = eval(msplit[2].split("_bt_")[0])
            model = fit_autogluon(key_train, config["target_column"], model_name=mname, params=params, freq=config["frequency"])
        else:
            model = None
        key_models[key] = (model_tag, model)
    return key_models
```

---

```python name=pipeline/infer.py
from models.statsmodels_wrapper import arima_predict, expsmooth_predict
from models.statsforecast_wrapper import statsforecast_predict
from models.prophet_wrapper import prophet_predict
from models.mlforecast_wrapper import mlforecast_predict
from models.autogluon_wrapper import autogluon_predict

def forecast_with_model(key_models, forecast_inputs, config):
    forecasts = {}
    features = config["extra_features"] if config["mode"] == "multivariate" else []
    for key, (model_tag, model) in key_models.items():
        if model_tag.startswith("arima_"):
            preds = arima_predict(model, config["backtests"][0]["forecast_horizon"])
        elif model_tag.startswith("ExponentialSmoothing_"):
            preds = expsmooth_predict(model, config["backtests"][0]["forecast_horizon"])
        elif model_tag.startswith("AutoARIMA_") or model_tag.startswith("ETS_") or model_tag.startswith("Theta_"):
            parts = model_tag.split("_", 1)
            preds = statsforecast_predict(model, config["backtests"][0]["forecast_horizon"], parts[0], key)
        elif model_tag.startswith("prophet_"):
            preds = prophet_predict(model, forecast_inputs[key], config["backtests"][0]["forecast_horizon"], config["date_column"], features if features else None)
        elif model_tag.startswith("LightGBM_") or model_tag.startswith("RandomForest_") or model_tag.startswith("CatBoost_"):
            msplit = model_tag.split("_", 1)
            preds = mlforecast_predict(model, config["backtests"][0]["forecast_horizon"], config["key_column"], msplit[0], key)
        elif model_tag.startswith("autogluon_"):
            msplit = model_tag.split("_", 2)
            preds = autogluon_predict(model, forecast_inputs[key].drop(columns=[config["target_column"]]), model_name=msplit[1])
        else:
            preds = None
        forecasts[key] = preds
    return forecasts
```

---

```python name=utils/evaluation.py
from sklearn.metrics import mean_absolute_error

def evaluate_forecast(y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    return mean_absolute_error(y_true, y_pred)
```

---

```python name=utils/plot.py
import matplotlib.pyplot as plt

def plot_forecasts(df, forecasts, config):
    for key in df[config["key_column"]].unique():
        plt.figure(figsize=(10, 4))
        key_df = df[df[config["key_column"]] == key]
        plt.plot(key_df[config["date_column"]], key_df[config["target_column"]], label="Actual")
        test_dates = key_df[key_df[config["date_column"]] >= config["backtests"][0]["train_end"]][config["date_column"]]
        preds = forecasts.get(key)
        if preds is not None:
            plt.plot(test_dates[:len(preds)], preds, label="Forecast")
        plt.title(f"{config['key_column'].capitalize()}: {key}")
        plt.xlabel("Date")
        plt.ylabel(config["target_column"].capitalize())
        plt.legend()
        plt.tight_layout()
        plt.show()
```

---

