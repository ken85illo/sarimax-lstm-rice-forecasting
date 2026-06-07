import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
import numpy as np
from min_max_scaler import MinMaxScaler
from utils import create_sequences, create_sequences_multistep, split_by_chunks
from lstm_network import LSTMNetwork

def train_on_residual_high():
    train_csv = pd.read_csv("datasets/sarimax_train_residuals_Well-Milled_High.csv")
    val_csv = pd.read_csv("datasets/sarimax_val_residuals_Well-Milled_High.csv")

    df_train = train_csv["Well-Milled High_train_residual"]
    df_val = val_csv["Well-Milled High_val_residual"]
    df_train_val = pd.concat([df_train, df_val], ignore_index=True)

    scaler = MinMaxScaler()
    scaled_df_train = scaler.fit_transform(df_train.values.reshape(-1, 1)) # Fit on training data
    scaled_df_val = scaler.transform(df_val.values.reshape(-1, 1))     # Transform validation data using the fitted scaler

    # Create sequences
    lookback = 14
    horizon = 14
    X_train, y_train = create_sequences_multistep(scaled_df_train, lookback, horizon)
    X_val, y_val = create_sequences_multistep(scaled_df_val, lookback, horizon)

    # print(f"y_train min: {y_train.min():.4f}, max: {y_train.max():.4f}, mean: {y_train.mean():.4f}")
    # print(f"X_train min: {X_train.min():.4f}, max: {X_train.max():.4f}")
    #
    # print(f"y_val min: {y_val.min():.4f}, max: {y_val.max():.4f}, mean: {y_val.mean():.4f}")
    # print(f"X_val min: {X_val.min():.4f}, max: {X_val.max():.4f}")
    # TODO: Instantiate and train your network
    network = LSTMNetwork(input_size=1, hidden_size=64, output_size=horizon, learning_rate=0.001, epochs=300)
    network.train( X_train, y_train, X_val, y_val, patience=20)
    network.save_model(target="high")

def train_on_residual_low():
    train_csv = pd.read_csv("datasets/sarimax_train_residuals_Well-Milled_Low.csv")
    val_csv = pd.read_csv("datasets/sarimax_val_residuals_Well-Milled_Low.csv")

    df_train = train_csv["Well-Milled Low_train_residual"]
    df_val = val_csv["Well-Milled Low_val_residual"]
    df_train_val = pd.concat([df_train, df_val], ignore_index=True)

    scaler = MinMaxScaler()
    scaled_df_train = scaler.fit_transform(df_train.values.reshape(-1, 1)) # Fit on training data
    scaled_df_val = scaler.transform(df_val.values.reshape(-1, 1))     # Transform validation data using the fitted scaler

    # Create sequences
    lookback = 14
    X_train, y_train = create_sequences(scaled_df_train, lookback)
    X_val, y_val = create_sequences(scaled_df_val, lookback)

    # TODO: Instantiate and train your network
    network = LSTMNetwork(input_size=1, hidden_size=64, output_size=1, learning_rate=0.001, epochs=150)
    network.train(X_train, y_train, X_val, y_val, patience=20)
    network.save_model(target="low")

def load_sarimax_model(model_name: str):
    filename = f"sarimax_model_Well-Milled_{model_name.capitalize()}.pkl"
    try:
        model = SARIMAXResultsWrapper.load(filename)
        print(f"Loaded SARIMAX model from {filename}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {filename}. Please ensure the 'models' directory exists and contains the model files.")
        return None

def predict_sarimax(model, exog_data=None):
    if model is None:
        return None

    if exog_data is None:
        print("Warning: No exogenous data provided for SARIMAX prediction. Using dummy data and forecasting for 10 days from 2026-05-01.")
        future_dates = pd.date_range(start='2026-05-01', periods=15, freq='D')
        exog_data = pd.DataFrame(np.zeros((10, model.exog.shape[1])), columns=[f'exog_{i}' for i in range(model.exog.shape[1])], index=future_dates)

    forecast = model.get_forecast(steps=exog_data.shape[0], exog=exog_data)
    forecast_df = forecast.summary_frame()
    return forecast_df

def example_sarimax_usage():
    # Load the models
    model_high = load_sarimax_model('high')
    model_low = load_sarimax_model('low')

    if model_high is None or model_low is None:
        print("Could not load one or more SARIMAX models. Exiting example.")
        return

    df_enso = pd.read_csv("datasets/enso_daily.csv", parse_dates=['date'], index_col='date')
    df_test_enso = df_enso["ONI value"]
    df_test_enso = df_test_enso[df_test_enso.index >= '2025-08-28'] 

    df_rice = pd.read_csv("datasets/well_milled_rice_daily_preprocessed.csv", parse_dates=['Date'], index_col='Date')
    df_test_rice_high = df_rice["Well-Milled High"][(df_rice["Well-Milled High"].index >= '2025-08-28')]
    df_test_rice_low = df_rice["Well-Milled Low"][(df_rice["Well-Milled Low"].index >= '2025-08-28')]
    
    current_date = pd.to_datetime('2025-09-11')
    end_date = df_test_rice_high.index.max()

    model_high, resid_high = forecast_rolling_walk_forward(model_high, df_test_rice_high, df_test_enso, current_date, end_date)
    lstm_predict("high", 14, resid_high)

    # model_low, resid_low = forecast_rolling_walk_forward(model_low, df_test_rice_low, df_test_enso, current_date, end_date)

def lstm_predict(target, lookback, resid):
    network = LSTMNetwork( input_size=1, hidden_size=64, output_size=14, learning_rate=0.001, epochs=150)
    network.load_model(target)

    train_csv = pd.read_csv("datasets/sarimax_train_residuals_Well-Milled_High.csv")
    val_csv = pd.read_csv("datasets/sarimax_val_residuals_Well-Milled_High.csv")

    df_train = train_csv["Well-Milled High_train_residual"]

    scaler = MinMaxScaler()
    scaler.fit(df_train.values.reshape(-1, 1))
    print(f"{scaler.min} {scaler.max}")
    
    val_csv = val_csv.rename(columns={"Well-Milled High_val_residual": "residuals"})
    if "Date" in val_csv.columns:
        val_csv = val_csv.set_index("Date")

    val_csv.index = pd.to_datetime(val_csv.index)
    df_test_resid = pd.concat([val_csv[len(val_csv) - lookback:], resid])["residuals"]
    scaled_test_resid = scaler.transform(df_test_resid.values.reshape(-1, 1))

    X_test = split_by_chunks(scaled_test_resid, 14)
    resid_dates = split_by_chunks(df_test_resid.index, 14)

    for X_seq, date in zip(X_test, resid_dates): 
        input_seq = X_seq
        input_inverse_seq = scaler.inverse_transform(input_seq)
        predicted = scaler.inverse_transform(network.predict(input_seq).reshape(-1, 1))

        print(f"Date: {date[0].date()}\nInput:")
        for j in range(len(input_inverse_seq)):
            last_date = date[j].date()
            print(f"{input_inverse_seq[j]} => {last_date}")

        print(f"Predicted:")
        for j in range(1, len(predicted) + 1):
            forecast_date = last_date + pd.Timedelta(days=j)            
            print(f"{predicted[j - 1]} => {forecast_date}")

        print()

def forecast_rolling_walk_forward(model, endog_dataset, exog_dataset, current_date, end_date, steps=14):
    forecasts = []
    fc_indices = []
    start_date = current_date
    while current_date <= end_date:
        forecast_end = min(current_date + pd.Timedelta(days=steps - 1), end_date)
        
        # Use the last 14 days of ENSO as exog
        exog_start = current_date - pd.Timedelta(days=steps)
        exog_forecast = exog_dataset.loc[exog_start:current_date - pd.Timedelta(days=1)]
        
        forecast = predict_sarimax(model, exog_data=exog_forecast)
        # if forecast is not None:
        #     print(f"\nForecast from {current_date.date()} (using ENSO {exog_start.date()} to {(current_date - pd.Timedelta(days=1)).date()}):")
        #     print(forecast.head(steps))

        # Refit on the forecasted 14 days
        endog_update = endog_dataset.loc[current_date:forecast_end]
        exog_update = exog_dataset.loc[current_date:forecast_end]
        
        forecasts.extend(forecast.values)
        fc_indices.extend(endog_dataset.loc[current_date: forecast_end].index.tolist())
    
        if len(endog_update) > 0:
            model = model.append(endog_update, exog=exog_update, refit=True)
        
        current_date = forecast_end + pd.Timedelta(days=1)
    
    forecasts = np.array(forecasts)
    forecast_test = pd.Series(forecasts[:len(fc_indices), 0], index=fc_indices)

    resid = endog_dataset[start_date:] - forecast_test
    resid = pd.DataFrame(resid, columns=["residuals"], index=fc_indices)

    return model, resid

if __name__ == "__main__":
    # train_on_residual_high()
    example_sarimax_usage()
