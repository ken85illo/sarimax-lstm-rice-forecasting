import numpy as np
import pandas as pd
from utils import print_tabulation
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from config import PathConfig


class SARIMAX:
    def __init__(self, model: SARIMAXResultsWrapper):
        self.model = model
        self.target = ""

    # == Loads the SARIMAX model that we trained sa Colab ==
    @classmethod
    def load(SARIMAX, target: str, paths: PathConfig | None = None) -> "SARIMAX":
        paths = paths or PathConfig()
        filename = paths.sarimax_model(target)

        try:
            # Wrapper is what allows the SARIMAX to be usable from its .pkl file
            # Warning pala, make sure na correct orders and residuals yung nandito
            model = SARIMAXResultsWrapper.load(filename)
            print(f"Loaded SARIMAX model from {filename}")

            return SARIMAX(model)

        except FileNotFoundError:
            raise FileNotFoundError(f"SARIMAX not found at '{filename}'.")

    # == Saves the newly updated SARIMAX model ==
    def save_model(self, target, paths: PathConfig | None = None):
        paths = paths or PathConfig()
        filename = paths.sarimax_model(target)
        
        # statsmodels wrapper objects have a built-in .save() method
        self.model.save(filename)
        print(f"Successfully saved updated SARIMAX model to {filename}")

    # == Walk forward forecast of SARIMAX ==
    def walk_forward(self, exog_window, steps=14):
        # Use the previous `steps` days of exogenous data as input
        # Executes actual prediction/forecasting
        forecast = self.model.get_forecast(steps=steps, exog=exog_window)

        # Basically creates formatted series of the forecast output
        forecast_df = forecast.summary_frame()["mean"]

        print_df = pd.DataFrame({
            "Input": exog_window.index.date,
            "Enso Index": exog_window.values,
            "Output": forecast_df.index.date,
            "Forecast": forecast_df.values
        })
        print()
        target = f"({self.target.capitalize()}) "
        print_tabulation(print_df, title=f"=== SARIMAX FORECAST {target}===")

        return forecast_df

    # == Update history of SARIMAX after walk forward ==
    def update_history(self, actual_window, exog_update):
        self.model = self.model.append(
            actual_window, exog=exog_update, refit=False
        )
        






    
    