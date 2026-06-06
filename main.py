import numpy as np
import pandas as pd
from lstm_network import LSTMNetwork
from min_max_scaler import MinMaxScaler
from utils import create_sequences
from min_max_scaler import MinMaxScaler

def test_overfitting():
    # 1. Setup: 3 inputs, 2 hidden neurons, 1 output
    network = LSTMNetwork(input_size=3, hidden_size=64, output_size=1, learning_rate=0.1, epochs=200)

    # 2. Dummy data: A single sequence of 5 time steps
    X_sample = np.random.randn(5, 3) # 5 steps, 3 features
    y_sample = np.array([0.8])       # Target

    # Wrap in list so it matches the expected iterable structure
    X_train = [X_sample]
    y_train = [y_sample]

    X_val = [X_sample]
    y_val = [y_sample]

    print(X_train)
    print(y_train)

    #3. Train
    print("Starting sanity check (loss should decrease)...")
    network.train( X_train, y_train, X_val, y_val, patience=20)
    print("\n\n===== FINISH TRAINING =====\n\n")

    #4. Predict
    final_pred = network.predict(X_sample)
    print(f"Target: {y_sample}, Prediction: {final_pred}")

def test_prediction():
    df_raw_rice = pd.read_csv("datasets/well_milled_rice_daily_preprocessed.csv")
    df_raw_rice['Date'] = pd.to_datetime(df_raw_rice['Date'])
    df_raw_rice = df_raw_rice.sort_values('Date').reset_index(drop=True)

    # 70-20-10 split
    train_split_count = 0.7
    validation_split_count = 0.2

    # Row split calculation for training - validation - testing
    total_rows = len(df_raw_rice)

    train_size = int(total_rows * train_split_count)
    val_size = int(total_rows * validation_split_count)

    # Split rice price data
    scaler = MinMaxScaler()
    train_val_rice_df = df_raw_rice.iloc[:train_size + val_size].copy()
    print(f"Train + Val Start: {train_val_rice_df['Date'].iloc[0]}\nTrain + Val End: {train_val_rice_df['Date'].iloc[-1]}")
    high_train_val_df = train_val_rice_df["Well-Milled High"]
    scaler.fit(high_train_val_df.values.reshape(-1, 1)) # Fit on training data

    df_train = high_train_val_df.iloc[:train_size]
    df_val = high_train_val_df.iloc[train_size:train_size + val_size]

    scaled_df_train = scaler.transform(df_train.values.reshape(-1, 1)) # Fit on training data
    scaled_df_val = scaler.transform(df_val.values.reshape(-1, 1))     # Transform validation data using the fitted scaler

    # Create sequences (adjust lookback as needed)
    lookback = 14
    X_train, y_train = create_sequences(scaled_df_train, lookback)
    X_val, y_val = create_sequences(scaled_df_val, lookback)

    # Test Dataset and min-max transform
    test_rice_df = df_raw_rice.iloc[train_size + val_size - lookback:].copy()
    high_test_df = test_rice_df["Well-Milled High"]
    scaled_df_test = scaler.transform(high_test_df.values.reshape(-1, 1)) # Fit on training data
    
    # Parameters and sequence creation
    lookback = 14
    X_test, y_test = create_sequences(scaled_df_test, lookback)
    test_dates = test_rice_df['Date'].reset_index(drop=True)

    network = LSTMNetwork(input_size=1, hidden_size=64, output_size=1, learning_rate=0.001, epochs=200)
    # network.train(X_train, y_train, X_val, y_val, patience=10)
    network.load_model("high-price")

    test_set = list(zip(X_test, y_test))


    for i in range(0, len(test_set), lookback): 
        X_seq, y_actual = test_set[i]
        date = test_dates.iloc[i]

        input_seq = X_seq
        input_inverse_seq = scaler.inverse_transform(input_seq)
        actual = scaler.inverse_transform(y_actual)
        predicted = scaler.inverse_transform(network.predict(input_seq, lookback))

        print(f"Date: {date.date()}\nInput:")
        for j in range(len(input_inverse_seq)):
            last_date = test_dates.iloc[i + j]
            print(f"{input_inverse_seq[j]} => {last_date}")

        print(f"Actual: {actual}\n")
        print(f"Predicted:")
        for j in range(1, len(predicted) + 1):
            forecast_date = last_date + pd.Timedelta(days=j)            
            print(f"{predicted[j - 1]} => {forecast_date}")

        print()




            


if __name__ == "__main__":
    test_prediction()

    

