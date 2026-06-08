import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from config import PathConfig


class SARIMAX:
    def __init__(self, model: SARIMAXResultsWrapper):
        self.model = model

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

    # == Walk forward forecast of SARIMAX ==
    def rolling_walk_forward(self, endog_dataset: pd.Series, exog_dataset, start_date, end_date, steps= 14,):
        forecasts = []
        fc_indices = []
        current_date = start_date

        while current_date <= end_date:
            forecast_end = min(current_date + pd.Timedelta(days=steps - 1), end_date)

            # Use the previous `steps` days of exogenous data as input
            exog_start = current_date - pd.Timedelta(days=steps)
            exog_window = exog_dataset.loc[exog_start: current_date - pd.Timedelta(days=1)]

            # Executes actual prediction/forecasting
            forecast = self.model.get_forecast(steps=steps, exog=exog_window)

            # Basically creates formatted dataframe of the forecast output
            forecast_df = forecast.summary_frame()

            # Pang-print lang to prove na walang data leakage
            print(
                f"\nForecast from {current_date.date()} "
                f"(using exog {exog_start.date()} to {(current_date - pd.Timedelta(days=1)).date()}):"
            )
            print(forecast_df.head(steps))

            # Collect results
            endog_window = endog_dataset.loc[current_date:forecast_end]
            forecasts.extend(forecast_df.values)
            fc_indices.extend(endog_window.index.tolist())

            # Refit on actuals (no parameter re-estimation)
            if len(endog_window) > 0:
                exog_update = exog_dataset.loc[current_date:forecast_end]
                self.model = self.model.append(endog_window, exog=exog_update, refit=False)

            current_date = forecast_end + pd.Timedelta(days=1)

        forecasts = np.array(forecasts)
        forecast_series = pd.Series(forecasts[: len(fc_indices), 0], index=fc_indices)
        residuals = endog_dataset[start_date:] - forecast_series
        residuals_df = pd.DataFrame({"residuals": residuals}, index=fc_indices)

        return forecast_series, residuals_df