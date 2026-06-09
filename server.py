from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import os
import io
import zipfile

from main import run_residual_learning_evaluation
from utils.backendutils import validate_input_dataframe, extractLSTMInputs, extractSARIMAXInputs

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

    rice_low_df, rice_high_df, enso_df = extractSARIMAXInputs(sarimax_input_df)
    google_trends_df, sentiment_df = extractLSTMInputs(lstm_input_df)
    
    

    # Temporarily return muna
    # return 

    # The logic here moving forward is formatting of return file
    # To be continued..
    # run_residual_learning_evaluation("high")
    
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, "output")
    csv_files = ["final_forecast_high.csv", "final_forecast_low.csv"]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for csv_file in csv_files:
            path = os.path.join(output_dir, csv_file)
            if not os.path.exists(path):
                return {"error": f"Output file not found: {csv_file}"}

            archive.write(path, arcname=csv_file)

    buffer.seek(0)

    print("RETURNING FINAL FORECAST CSV")
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=output_csv.zip"},
    )