from statsmodels.tsa.arima.model import ARIMA

def train_arima(train_series, order=(1,1,1)):
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps=30):
    return model_fit.forecast(steps=steps)