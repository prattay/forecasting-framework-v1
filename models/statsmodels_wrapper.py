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