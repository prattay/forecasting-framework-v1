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