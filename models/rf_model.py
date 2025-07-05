from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_rf(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def forecast_rf(model, X_forecast):
    return model.predict(X_forecast)