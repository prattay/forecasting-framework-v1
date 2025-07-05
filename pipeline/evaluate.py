from sklearn.metrics import mean_absolute_error

def evaluate_models(df, forecasts):
    y_true = df['sales'][-len(list(forecasts.values())[0]):]
    metrics = {}
    for model, y_pred in forecasts.items():
        metrics[model] = mean_absolute_error(y_true, y_pred)
    return metrics