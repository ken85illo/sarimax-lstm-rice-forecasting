from dataclasses import dataclass, field

# This file contains mga configuration ng LSTM and yung paths ng models/datasets
# Note: Pag magtetest kayo please dito nyo nalang baguhin yung params

@dataclass
class ModelConfig:
    lookback: int = 14
    horizon: int = 14
    hidden_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 300
    patience: int = 20
    input_size: int = 2        # 2 for multivariate (price + trends), 1 for univariate
    output_size: int = 14      # matches horizon for multi-step forecasting


@dataclass
class PathConfig:
    datasets_dir: str = "datasets"
    combined_trends_csv: str = "datasets/combined_google_trends_dataset.csv"
    rice_csv: str = "datasets/well_milled_rice_daily_preprocessed.csv"
    enso_csv: str = "datasets/enso_daily.csv"
    sentiments_csv: str = "datasets/daily_sentiment_scores.csv"

    def sarimax_model(self, target: str) -> str:
        return f"checkpoint/sarimax_model_Well-Milled_{target.capitalize()}.pkl"

    def train_residuals(self, label: str) -> str:
        return f"{self.datasets_dir}/sarimax_train_residuals_Well-Milled_{label.capitalize()}.csv"

    def val_residuals(self, label: str) -> str:
        return f"{self.datasets_dir}/sarimax_val_residuals_Well-Milled_{label.capitalize()}.csv"