from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import os
import json
import io
import zipfile

from main import run_residual_forecast, run_residual_learning_test_set
from utils import validate_input_dataframe, extract_lstm_inputs, extract_sarimax_inputs

app = FastAPI()

# Allow your JavaScript frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"message": "test endpoint"}

@app.post("/process-csv")
async def process_csv(sarimax_input_file: UploadFile = File(...), lstm_input_file: UploadFile = File(...)):
    # Read SARIMAX Input File: Well-Milled (Low & High) and ENSO Index
    sarimax_contents = await sarimax_input_file.read()
    sarimax_input_df = pd.read_csv(io.BytesIO(sarimax_contents))

    # Read LSTM Input File: SVI and Sentiment Scores
    lstm_contents = await lstm_input_file.read()
    lstm_input_df = pd.read_csv(io.BytesIO(lstm_contents))

    valid, validation_errors = validate_input_dataframe(sarimax_input_df, lstm_input_df)

    if not valid:
        return {
            "status": "Input Validation Error",
            "errors": validation_errors,
        }

    rice_low_df, rice_high_df, enso_df = extract_sarimax_inputs(sarimax_input_df)
    google_trends_df, sentiment_df = extract_lstm_inputs(lstm_input_df)

    # run_residual_learning_test_set("high")
    # run_residual_learning_test_set("low")
    
    run_residual_forecast(rice_low_df, rice_high_df, enso_df, google_trends_df, sentiment_df)
    
    df_high_forecast = pd.read_csv("output/final_demo_prediction_high.csv")
    df_low_forecast = pd.read_csv("output/final_demo_prediction_low.csv")

    high_errors_path = "output/final_demo_high_errors.json"
    low_errors_path = "output/final_demo_low_errors.json"

    with open(high_errors_path, "r", encoding="utf-8") as f:
        high_errors_data = json.load(f)
        
    # Load the low errors JSON file safely back into a Python list/dict
    with open(low_errors_path, "r", encoding="utf-8") as f:
        low_errors_data = json.load(f)

    return {
        "high_forecast": df_high_forecast.to_dict(orient="records"),
        "high_forecast_errors": high_errors_data,
        "low_forecast": df_low_forecast.to_dict(orient="records"),
        "low_forecast_errors": low_errors_data
    }