from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor

def fit_autogluon(train_df, target_col, problem_type="regression", model_name=None, params=None, freq="D"):
    if model_name == "TFT":
        predictor = TimeSeriesPredictor(
            target=target_col,
            prediction_length=params.get("prediction_length", 3),
            freq=freq,
            eval_metric="MAE"
        ).fit(train_df, presets="medium_quality", hyperparameters={"TFT": params})
        return predictor
    else:
        predictor = TabularPredictor(label=target_col, problem_type=problem_type).fit(
            train_df,
            presets="best_quality",
            hyperparameters={model_name: params or {}}
        )
        return predictor

def autogluon_predict(predictor, test_df, model_name=None):
    if model_name == "TFT":
        return predictor.predict(test_df)
    else:
        return predictor.predict(test_df)