from datetime import datetime

import pandas as pd

MIN_INPUT_DATE = datetime(2026, 4, 30)

REQUIRED_SARIMAX_COLUMNS = ["Date", "Well-Milled Low", "Well-Milled High", "ONI value"]

REQUIRED_LSTM_COLUMNS = ["Date", "Google Trends", "score_positive", "score_neutral", "score_negative"]

def check_missing_columns(df, required_columns):
    return [col for col in required_columns if col not in df.columns]

def validate_dataframe(df, required_columns, min_date, label):
    errors = []
    
    missing_columns = check_missing_columns(df, required_columns)
    if missing_columns:
        errors.append(f"Missing required columns for {label}: {', '.join(missing_columns)}.")

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        if dates.isna().any():
            errors.append(f"Some Date values in {label} are invalid or could not be parsed.")

        else:
            invalid_dates = dates[dates <= min_date]
            if not invalid_dates.empty:
                errors.append(
                    f"All dates in {label} must be after {min_date.strftime('%Y-%m-%d')}. "
                )

    for col in required_columns:
        if col in df.columns and df[col].isna().any():
            errors.append(f"Column '{col}' in {label} contains missing values.")

    return errors

def validate_input_dataframe(df_sarimax_input, df_lstm_input, min_date=MIN_INPUT_DATE):
    errors = []

    sarimax_input_errors = validate_dataframe(df_sarimax_input, REQUIRED_SARIMAX_COLUMNS, min_date, "SARIMAX input")
    errors.extend(sarimax_input_errors)

    lstm_input_errors = validate_dataframe(df_lstm_input, REQUIRED_LSTM_COLUMNS, min_date, "LSTM input")
    errors.extend(lstm_input_errors)

    return len(errors) == 0, errors

def extractSARIMAXInputs(df):
    df = df.set_index("Date")
    df_prepared = df.sort_index()

    # Rice Prices (Low and High) 
    rice_low_df = df_prepared[["Well-Milled Low"]].copy()
    rice_high_df = df_prepared[["Well-Milled High"]].copy()

    # ENSO Index
    enso_df = df_prepared[["ONI value"]].copy()

    return rice_low_df, rice_high_df, enso_df


def extractLSTMInputs(df):
    """Return google trends and sentiment series (as DataFrames) indexed by Date."""
    df = df.set_index("Date")
    df_prepared = df.sort_index()
    
    # Google Trends SVI
    google_trends_df = df_prepared[["Google Trends"]].copy()

    # Sentiment Scores
    sent_pos_df = df_prepared[["score_positive"]].copy()
    sent_neg_df = df_prepared[["score_negative"]].copy()
    sent_neut_df = df_prepared[["score_neutral"]].copy()
    
    return google_trends_df, sent_pos_df, sent_neg_df, sent_neut_df
