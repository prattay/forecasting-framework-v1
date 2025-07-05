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
        # Select best model (lowest avg MAE across backtests)
        avg_maes = {}
        for model_key in set(k.rsplit("_bt_", 1)[0] for k in model_maes):
            vals = [v for k, v in model_maes.items() if k.startswith(model_key)]
            avg_maes[model_key] = sum(vals) / len(vals)
        best_model = min(avg_maes, key=avg_maes.get)
        best_model_by_key[key] = best_model
        scores[key] = avg_maes
    return scores, best_model_by_key