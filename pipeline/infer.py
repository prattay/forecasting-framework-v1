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