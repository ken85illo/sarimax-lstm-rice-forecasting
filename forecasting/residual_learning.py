import numpy as np
import pandas as pd
from forecasting import SARIMAX
from forecasting import LSTM
from utils import mae, mape, rmse, print_tabulation, truncate

class ResidualLearning:
    TRUNCATE_DECIMALS = 0

    def __init__(self, sarimax: SARIMAX, lstm: LSTM, target, steps):
        self.sarimax = sarimax
        self.lstm = lstm

        self.target = target
        sarimax.target = target
        lstm.target = target
        self.steps = steps

    def run_rolling(self, endog, exog, 
            trends_test, 
            sentiment_test,
            residuals_tail,
            trends_tail, 
            sentiment_tail,
            start_date, 
            end_date,
            target_csv=None, is_test_set=True, target='low'):
        
        
        all_fusion = []
        all_actuals = []
        all_sarimax = []
        all_lstm = []
        all_dates = []
        all_residuals = []

        current_date = start_date

        # Seed the LSTM input with the last `lookback` rows from validation
        lstm_residual_window = residuals_tail.copy()  # shape: (14,)
        lstm_trends_window = trends_tail.copy()        # shape: (14,)
        lstm_sentiment_window = sentiment_tail.copy()


        while current_date <= end_date:
            window_end = min(current_date + pd.Timedelta(days=self.steps - 1), end_date)

            # Get exogenous window (ENSO)
            exog_window = self._get_exog_window(exog,current_date)

            # Forecast one step
            sarimax_forecast, lstm_prediction, fusion_forecast = self._forecast_one_step(
                exog_window, lstm_residual_window, lstm_trends_window, 
                lstm_sentiment_window, current_date
            )

            # Get the actual data at window
            actual_window = endog.loc[current_date:window_end]
            window_dates = actual_window.index.date.tolist()
            n = len(window_dates)

            all_dates.extend(window_dates)
            all_actuals.extend(actual_window.values)
            all_sarimax.extend(sarimax_forecast[:n])
            all_lstm.extend(lstm_prediction[:n])
            all_fusion.extend(fusion_forecast[:n])


            # Calculate the next residual by subtracting actual with SARIMAX forecast
            lstm_residual_window = (actual_window.values - sarimax_forecast[:n])
            all_residuals.extend(lstm_residual_window[:n])

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
            "SARIMAX": truncate(all_sarimax, decimals=self.TRUNCATE_DECIMALS), # Ni-round ko para hindi decimal yung forecast
            "Residuals": all_residuals,
            "LSTM Correction": np.array(all_lstm).flatten(),
            "Final Forecast": np.round(all_fusion) # truncate(all_fusion, decimals=self.TRUNCATE_DECIMALS), # Same here naka round din
        })

        self._report(comparison, target, is_test_set)

        residuals = pd.DataFrame({
            "Date": all_dates,
            "Residuals": all_residuals
        })

        exog_window = self._get_exog_window(exog, current_date)

        # Final forecast step
        lstm_residual_window = all_residuals[-self.steps:]
        lstm_trends_window = trends_test.iloc[-self.steps:]
        lstm_sentiment_window = sentiment_test.iloc[-self.steps:]
        sarimax_forecast, _,  fusion_forecast = self._forecast_one_step(
            exog_window, lstm_residual_window, lstm_trends_window, 
            lstm_sentiment_window, current_date
        )

        forecast_dates = [current_date.date() + pd.Timedelta(days=t) for t in range(len(fusion_forecast)) ]
        forecast = pd.DataFrame({
            "Date": forecast_dates,
            "SARIMAX Forecast": truncate(sarimax_forecast.values, self.TRUNCATE_DECIMALS),
            "Residual Forecast": np.round(fusion_forecast.flatten()),
        })
        print()
        target = f"({self.target})"
        print_tabulation(forecast, title=f"=== FORECAST {target}===")
                
        self._save_csv(target_csv, comparison, residuals, forecast, is_test_set)

        return comparison

        
    def _forecast_one_step(
        self, exog_window, lstm_residual_window, lstm_trends_window,
        lstm_sentiment_window, current_date
    ):
        sarimax_forecast = self.sarimax.walk_forward(exog_window, self.steps)

        # Feed residual, trends, and sentiment window to LSTM
        lstm_prediction = self.lstm.predict(
            lstm_residual_window, lstm_trends_window, lstm_sentiment_window, current_date, self.steps
        )

        # Fusion of SARIMAX and LSTM residuals (Residual Learning)
        fusion_forecast = sarimax_forecast.values + lstm_prediction.flatten()

        return sarimax_forecast, lstm_prediction, fusion_forecast

    def _get_exog_window(self, exog, current_date):
        # Create exog window for sarimax
        exog_start = current_date - pd.Timedelta(days=self.steps)
        exog_window = exog.loc[exog_start: current_date - pd.Timedelta(days=1)]

        return exog_window

        
    def _save_csv(self,target_csv, comparison, residuals, forecast, is_test_set = True):
        prefix = "test" if is_test_set else "final"
        if target_csv:
            resid_csv = f"output/{prefix}_demo_residuals_{target_csv}.csv"
            forecast_csv = f"output/{prefix}_forecast_{target_csv}.csv"
            demo_csv = f"output/{prefix}_demo_prediction_{target_csv}.csv"

            if is_test_set:
                demo_json = f"output/{prefix}_demo_prediction_{target_csv}.json"
    
                # Create a temporary copy so we don't mutate your original DataFrame
                temp_comparison = comparison.copy()
                
                if temp_comparison.index.name == 'Date':
                    temp_comparison = temp_comparison.reset_index().rename(columns={'index': 'Date'})
                    
                if 'Date' in temp_comparison.columns:
                    temp_comparison['Date'] = pd.to_datetime(temp_comparison['Date']).dt.strftime('%Y-%m-%d')
                
                # Export the temporary dataframe to JSON
                temp_comparison.to_json(demo_json, orient='records', indent=4)

            comparison.to_csv(demo_csv)
            residuals.to_csv(resid_csv)
            forecast.to_csv(forecast_csv)
                
            print(f"\nDemo Comparison saved to {demo_csv}")
            print(f"Residuals saved to {resid_csv}")
            print(f"Forecast saved to {forecast_csv}")

    # == Just another method for printing sa terminal == 
    @staticmethod
    def _report(comparison: pd.DataFrame, target, is_test_set = False):
        errors_df = pd.DataFrame()
        
        # Prints the final forecast output 
        print()
        print_tabulation(comparison.head(10), title="=== RESULTS ===")
        
        for label, col in [("SARIMAX", "SARIMAX"), ("RESIDUAL LEARNING", "Final Forecast")]:
            pred = comparison[col]
            actual = comparison["Actual"]
            
            rmse_score = rmse(pred, actual)
            mae_score = mae(pred, actual)
            mape_score = mape(pred, actual)

            rmse_score = truncate(rmse_score, 2)
            mae_score = truncate(mae_score, 2)
            mape_score = truncate(mape_score, 2)

            df_row = pd.DataFrame([{'model': label, 'rmse': rmse_score, 'mae': mae_score, 'mape': mape_score}])
            errors_df = pd.concat([errors_df, df_row], ignore_index=True)
            
            # Pa-add nalang here if may kulang pa na metric
            print(f"\n=== {label} ===")
            print(f"RMSE:  {rmse_score:.4f}")
            print(f"MAE:   {mae_score:.4f}")
            print(f"MAPE:  {mape_score:.4f}%")
        
        prefix = "test_demo" if is_test_set else "final_demo"
        errors_json = F"output/{prefix}_{target}_errors.json"

        errors_df.to_json(errors_json, orient='records', indent=4)
