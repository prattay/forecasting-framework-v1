from sklearn.metrics import mean_absolute_error

def evaluate_forecast(y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    return mean_absolute_error(y_true, y_pred)