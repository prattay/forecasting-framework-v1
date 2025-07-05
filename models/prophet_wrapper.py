from prophet import Prophet
import pandas as pd

def fit_prophet(df, params, target_col, date_col, extra_features=None):
    df_cp = df[[date_col, target_col] + (extra_features or [])].copy()
    df_cp.rename(columns={date_col: "ds", target_col: "y"}, inplace=True)
    m = Prophet(
        seasonality_mode=params.get("seasonality_mode", "additive"),
        yearly_seasonality=params.get("yearly_seasonality", 10),
        weekly_seasonality=params.get("weekly_seasonality", 3),
        daily_seasonality=params.get("daily_seasonality", 1)
    )
    if extra_features:
        for feat in extra_features:
            m.add_regressor(feat)
    m.fit(df_cp)
    return m

def prophet_predict(m, future_df, periods, date_col, extra_features=None):
    last_date = future_df[date_col].max()
    freq = pd.infer_freq(future_df[date_col])
    future = pd.date_range(last_date, periods=periods+1, freq=freq)[1:]
    future_df_new = pd.DataFrame({date_col: future})
    if extra_features:
        for feat in extra_features:
            future_df_new[feat] = future_df[feat].iloc[-1]
    future_df_new.rename(columns={date_col: "ds"}, inplace=True)
    forecast = m.predict(future_df_new)
    return forecast["yhat"].values