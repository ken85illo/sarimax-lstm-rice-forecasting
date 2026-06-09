import numpy as np
import pandas as pd
from forecasting import SARIMAX
from forecasting import LSTM
from utils import mae, mape, rmse
from utils.utils import print_tabulation

class ResidualLearning:
    def __init__(self, sarimax: SARIMAX, lstm: LSTM):
        self.sarimax = sarimax
        self.lstm = lstm

    def run_rolling(self, endog, exog, 
            trends_test, 
            sentiment_test,
            val_residuals_tail, 
            val_trends_tail, 
            val_sentiment_tail,
            start_date, 
            end_date,
            steps=14, target_csv=None,):
        
        
        all_fusion = []
        all_actuals = []
        all_sarimax = []
        all_lstm = []
        all_dates = []
        all_residuals = []

        current_date = start_date

        # Seed the LSTM input with the last `lookback` rows from validation
        lstm_residual_window = val_residuals_tail.copy()  # shape: (14,)
        lstm_trends_window = val_trends_tail.copy()        # shape: (14,)
        lstm_sentiment_window = val_sentiment_tail.copy()
        
        while current_date <= end_date:
            window_end = min(current_date + pd.Timedelta(days=steps - 1), end_date)

            # Create exog window for sarimax
            exog_start = current_date - pd.Timedelta(days=steps)
            exog_window = exog.loc[exog_start: current_date - pd.Timedelta(days=1)]
            sarimax_forecast = self.sarimax.walk_forward(exog_window, steps=14)

            # Feed residual, trends, and sentiment window to LSTM
            lstm_prediction = self.lstm.predict(
                lstm_residual_window, lstm_trends_window, lstm_sentiment_window, current_date
            )

            # Fusion of SARIMAX and LSTM residuals (Residual Learning)
            fusion_forecast = sarimax_forecast.values + lstm_prediction.flatten()

            # Get the actual data at window
            actual_window = endog.loc[current_date:window_end]
            window_dates = actual_window.index.date.tolist()
            n = len(window_dates)

            all_dates.extend(window_dates)
            all_actuals.extend(actual_window.values)
            all_sarimax.extend(sarimax_forecast[:n])
            all_lstm.extend(lstm_prediction[:n])
            all_fusion.extend(fusion_forecast[:n])
            all_residuals.extend(lstm_residual_window[:n])

            # Calculate the next residual by subtracting actual with SARIMAX forecast
            lstm_residual_window = (actual_window.values - sarimax_forecast[:n])
            
            # Get the next Trends and Sentiment window
            trends_window = trends_test.loc[current_date:window_end]
            lstm_trends_window = trends_window[:n]
            sentiment_window = sentiment_test.loc[current_date:window_end]
            lstm_sentiment_window = sentiment_window[:n]

            # Update the historical and exogenous data of SARIMAX
            exog_update = exog.loc[current_date:window_end]
            self.sarimax.update_history(actual_window, exog_update)
            current_date = window_end + pd.Timedelta(days=1)

        comparison = pd.DataFrame({
            "Date": all_dates,
            "Actual": all_actuals,
            "SARIMAX": np.round(all_sarimax), # Ni-round ko para hindi decimal yung forecast
            "LSTM Correction": np.array(all_lstm).flatten(),
            "Final Forecast": np.round(all_fusion), # Same here naka round din
        })

        self._report(comparison)

        test_residuals = pd.DataFrame({
            "Date": all_dates,
            "Residuals": all_residuals
        })

        self._save_csv(target_csv, comparison, test_residuals)


        return comparison

        
    def _save_csv(self,target_csv, comparison, test_residuals):
        if target_csv:
            output_csv = f"output/final_forecast_{target_csv}.csv"
            test_csv = f"output/test_residuals_{target_csv}.csv"

            comparison.to_csv(output_csv)
            test_residuals.to_csv(test_csv)
            print(f"\nResults saved to {output_csv}")

    # == Just another method for printing sa terminal == 
    @staticmethod
    def _report(comparison: pd.DataFrame):
        # Prints the final forecast output 
        print()
        print_tabulation(comparison.head(10), title="=== RESULTS ===")

        for label, col in [("SARIMAX", "SARIMAX"), ("RESIDUAL LEARNING", "Final Forecast")]:
            pred = comparison[col]
            actual = comparison["Actual"]

            # Pa-add nalang here if may kulang pa na metric
            print(f"\n=== {label} ===")
            print(f"RMSE:  {rmse(pred, actual):.4f}")
            print(f"MAE:   {mae(pred, actual):.4f}")
            print(f"MAPE:  {mape(pred, actual):.4f}%")