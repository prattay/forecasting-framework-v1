import matplotlib.pyplot as plt

def plot_forecasts(df, forecasts, config):
    for key in df[config["key_column"]].unique():
        plt.figure(figsize=(10, 4))
        key_df = df[df[config["key_column"]] == key]
        plt.plot(key_df[config["date_column"]], key_df[config["target_column"]], label="Actual")
        test_dates = key_df[key_df[config["date_column"]] >= config["backtests"][0]["train_end"]][config["date_column"]]
        preds = forecasts.get(key)
        if preds is not None:
            plt.plot(test_dates[:len(preds)], preds, label="Forecast")
        plt.title(f"{config['key_column'].capitalize()}: {key}")
        plt.xlabel("Date")
        plt.ylabel(config["target_column"].capitalize())
        plt.legend()
        plt.tight_layout()
        plt.show()