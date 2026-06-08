import numpy as np
import pandas as pd
from forecasting import SARIMAX
from forecasting import LSTM
from utils import mae, mape, rmse

class ResidualLearning:
    def __init__(self, sarimax: SARIMAX, lstm: LSTM):
        self.sarimax = sarimax
        self.lstm = lstm

    def run(self, endog, exog, 
            trends_test, 
            sentiment_test,
            val_residuals_tail, 
            val_trends_tail, 
            val_sentiment_tail,
            start_date, 
            end_date,
            steps=14, output_csv = "final_forecast.csv",):
        
        # SARIMAX rolling forecast
        sarimax_forecasts, residuals_df = self.sarimax.rolling_walk_forward(
            endog_dataset=endog,
            exog_dataset=exog,
            start_date=start_date,
            end_date=end_date,
            steps=steps,
        )

        # LSTM rolling forecast from residuals
        lstm_corrections = self.lstm.predict(
            residuals=residuals_df["residuals"],
            trends=trends_test,
            sentiment=sentiment_test,
            val_residuals_tail=val_residuals_tail,
            val_trends_tail=val_trends_tail,
            val_sentiment_tail=val_sentiment_tail
        )

        # Additive fusionn
        actual = endog[start_date:]
        final_forecast = sarimax_forecasts + lstm_corrections

        # Pang check lang if match pa rin yung dates
        for s in [sarimax_forecasts, lstm_corrections, actual, final_forecast]:
            s.index = pd.to_datetime(s.index)

        # Single dataframe para madali nalang i-print
        comparison = pd.DataFrame({
            "actual": actual,
            "sarimax": np.round(sarimax_forecasts[: len(actual)]), # Ni-round ko para hindi decimal yung forecast
            "residuals": residuals_df["residuals"],
            "lstm_correction": lstm_corrections[: len(actual)],
            "final_forecast": np.round(final_forecast[: len(actual)]), # Same here naka round din
        })

        self._report(comparison)

        if output_csv:
            comparison.to_csv(output_csv)
            print(f"\nResults saved to {output_csv}")

        return comparison

    # == Just another method for printing sa terminal == 
    @staticmethod
    def _report(comparison: pd.DataFrame):
        # Prints the final forecast output 
        print(comparison)

        for label, col in [("SARIMAX", "sarimax"), ("RESIDUAL LEARNING", "final_forecast")]:
            pred = comparison[col]
            actual = comparison["actual"]

            # Pa-add nalang here if may kulang pa na metric
            print(f"\n=== {label} ===")
            print(f"RMSE:  {rmse(pred, actual):.4f}")
            print(f"MAE:   {mae(pred, actual):.4f}")
            print(f"MAPE:  {mape(pred, actual):.4f}%")