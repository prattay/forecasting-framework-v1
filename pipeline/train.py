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