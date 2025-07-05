import pandas as pd
from config import config
from pipeline.backtest import backtest_models
from pipeline.train import train_final_model
from pipeline.infer import forecast_with_model
from utils.plot import plot_forecasts

def run_testing(df, config):
    scores, best_model_by_key = backtest_models(df, config)
    print("Backtest scores (MAE) by key and model:")
    for key, model_scores in scores.items():
        print(f"{key}:")
        for k, v in model_scores.items():
            print(f"  {k}: {v}")
    print("Best model per key (by avg MAE):", best_model_by_key)
    # Optionally, write results to file for later reference
    pd.DataFrame(scores).to_csv("backtest_scores.csv")
    pd.Series(best_model_by_key).to_csv("best_models.csv")
    return best_model_by_key

def run_training_inferencing(df, config, best_model_by_key):
    # User override: if config["production_models"] is set, use that instead
    use_models = config["production_models"] if config["production_models"] else best_model_by_key
    print("Using the following models for final training/inferencing:")
    print(use_models)
    # Train final models on all available data up to latest date
    train_df = df
    key_models = train_final_model(train_df, config, use_models)
    # Prepare inference inputs for each key if needed (e.g., Prophet, AutoGluon)
    forecast_inputs = {}
    for key in df[config["key_column"]].unique():
        forecast_inputs[key] = df[df[config["key_column"]] == key]
    # Forecast
    forecasts = forecast_with_model(key_models, forecast_inputs, config)
    # Plot
    plot_forecasts(df, forecasts, config)

def main():
    df = pd.read_csv("data/sales.csv", parse_dates=[config["date_column"]])
    if config["pipeline_mode"] == "testing":
        best_model_by_key = run_testing(df, config)
    elif config["pipeline_mode"] == "training-inferencing":
        # Try to read best models from file if not set
        try:
            best_model_by_key = pd.read_csv("best_models.csv", index_col=0, squeeze=True).to_dict()
        except Exception:
            print("Warning: No best models found, running testing phase first.")
            best_model_by_key = run_testing(df, config)
        run_training_inferencing(df, config, best_model_by_key)
    else:
        raise ValueError("Unknown pipeline_mode in config.")

if __name__ == "__main__":
    main()