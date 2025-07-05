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