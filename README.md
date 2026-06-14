# SARIMAX-LSTM Rice Forecasting

Rice price forecasting using a hybrid residual learning approach. This repository combines SARIMAX for baseline price modeling with an LSTM neural network trained on SARIMAX residuals and Google Trends data.

## Project Overview

- **Objective**: Forecast daily well-milled rice prices using both statistical time series modeling and deep learning residual correction.
- **Approach**:
  - Use SARIMAX to model rice price dynamics with ENSO as an exogenous input.
  - Compute residuals from the SARIMAX model.
  - Train an LSTM on residuals and Google Trends features to generate corrective forecasts.
  - Combine SARIMAX forecasts and LSTM corrections for final predictions.
- **Data sources**:
  - Rice price dataset
  - Google Trends dataset
  - ENSO index dataset
  - SARIMAX residual training/validation datasets

## Repository Structure

- `main.py` - Entry point for training and evaluation workflows.
- `config.py` - Configuration dataclasses for model hyperparameters and file paths.
- `requirements.txt` - Python dependencies required to run the project.
- `frontend/` - submodule/external user-interface repository.
- `data/`
  - `data_loader.py` - Loads rice prices, Google Trends, ENSO index, and SARIMAX residual CSV files.
  - `preprocessor.py` - Preprocesses time series, performs dataset splitting, and generates LSTM sequences.

- `datasets/`
  - `well_milled_rice_daily_preprocessed.csv`
  - `sarimax_train_residuals_Well-Milled_High.csv`
  - `sarimax_train_residuals_Well-Milled_Low.csv`
  - `sarimax_val_residuals_Well-Milled_High.csv`
  - `sarimax_val_residuals_Well-Milled_Low.csv`
  - `enso_daily.csv`
  - `combined_google_trends_dataset.csv`
  - `daily_sentiment_scores.csv`

- `colab-notebooks/`
  - `SARIMAX.ipynb` - Colab notebook on how the SARIMAX model used in the forecasting was trained and exported.
  - `XLM-RoBERTa.ipynb` - Colab notebook that provided the sentiment scores used in LSTM.

- `forecasting/`
  - `sarimax.py` - SARIMAX model implementation and rolling forecast logic.
  - `lstm.py` - LSTM wrapper for loading/saving and making predictions on residuals.
  - `residual_learning.py` - Combines SARIMAX and LSTM predictions into a final residual learning forecast.

- `lstm/`
  - `lstm_cell.py` - LSTM cell implementation.
  - `lstm_network.py` - LSTM network architecture.
  - `output_layer.py` - Output layer and loss computation.

- `training/`
  - `trainer.py` - Training loop, validation, early stopping, and checkpointing.
  - `checkpoint.py` - Model checkpoint utility.

- `utils/`
  - `min_max_scaler.py` - Custom min-max scaler for feature normalization.
  - `utils.py` - Utility helpers for sequence generation and evaluation metrics.
  - `backend_utils.py` - Utility helpers for backend logic.

- `checkpoint/`
  - Pretrained model checkpoints and scaler files used by evaluation workflows.

- `output/`
  - `final_forecast_high.csv`
  - `final_forecast_low.csv`

## Installation

1. Create and activate a Python virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```

2. Install dependencies.

   ```powershell
   python -m pip install -r requirements.txt
   ```

## Running the Project

### Evaluate residual learning forecast

Use the user interface provided in the `frontend/` submodule. Open the user interface by running the `index.html` file via live server, and provide it with the necessary details. 

To run the backend server that accepts the inputs, execute the script in your terminal to run the backend server.
```powershell
uvicorn server:app --reload
```

### Train LSTM residual model

Uncomment or call `train_lstm_residuals(target="low")` or `train_lstm_residuals(target="high")` in `main.py`.
Then execute:

```powershell
python main.py
```