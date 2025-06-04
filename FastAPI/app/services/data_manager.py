import pandas as pd


class DataManager:
    def __init__(self, csv_path: str, tickers: list[str]):
        self.data = pd.read_csv(csv_path)
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.tickers = tickers

    def get_ticker_history(self, ticker: str, start_date=None, end_date=None) -> dict:
        temp_df = self.data[["date", ticker]]
        if start_date:
            temp_df = temp_df[temp_df["date"] >= start_date]
        if end_date:
            temp_df = temp_df[temp_df["date"] <= end_date]

        return {
            "ticker": ticker,
            "dates": temp_df["date"].tolist(),
            "values": temp_df[ticker].tolist(),
        }

    def filter_data_for_training(self, ticker: str, base_date: pd.Timestamp, window: int = 60):
        border = base_date - pd.Timedelta(days=window)
        return self.data[(self.data["date"] >= border) & (self.data["date"] <= base_date)][["date", ticker]]
