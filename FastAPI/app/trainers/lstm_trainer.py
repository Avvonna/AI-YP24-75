import logging

import numpy as np
import pandas as pd
import torch
from app.configs import LSTMConfig
from app.core.data_manager import DataManager
from app.features import create_time_features, update_extended_features_lastrow
from app.models.nn import LSTMForecaster, TimeSeriesDataset, fit_model
from app.schemas import ExperimentMetrics
from app.trainers import BaseModelTrainer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMTrainer(BaseModelTrainer):
    config_class = LSTMConfig

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.model = None
        self.df = None
        self.feature_columns = []
        self.target_column = "target"

    def train(self, ticker: str, base_date: pd.Timestamp, config: config_class):
        self.window = config.window_size
        df = self.data_manager.get_features(ticker, base_date)

        if df.empty or df[self.target_column].dropna().empty:
            raise ValueError(f"Недостаточно данных для обучения по тикеру '{ticker}'")

        self.df = df.dropna().copy()
        self.feature_columns = [col for col in df.columns if col != self.target_column]
        X = df[self.feature_columns].values
        y = df[self.target_column].values

        self.scaler_X = StandardScaler().fit(X)
        self.scaler_y = StandardScaler().fit(y.reshape(-1, 1))

        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).squeeze()

        # Преобразуем в 3D (samples, timesteps, features)
        if len(X_scaled) < config.window_size + 1:
            raise ValueError(
                f"Недостаточно данных: {len(X_scaled)} строк, требуется минимум {config.window_size + 1} "
                f"для окна ({config.window_size}) и хотя бы одного шага обучения"
            )

        X_seq = np.array([X_scaled[i - self.window:i] for i in range(self.window, len(X_scaled))])
        y_seq = y_scaled[self.window:]

        dataset = TimeSeriesDataset(X_seq, y_seq)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        model = LSTMForecaster(
            input_dim=X_seq.shape[2],
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_dim=1,
            dropout=config.dropout
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_fn = nn.MSELoss()

        model, train_losses, val_losses = fit_model(
            model=model,
            train_loader=dataloader,
            val_loader=dataloader,  # TODO: сделать train-test split?
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=config.epochs,
            patience=config.patience
        )

        self.model = model
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
        preds_scaled = model(X_tensor).detach().cpu().numpy().squeeze()

        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).squeeze()
        y_true = self.scaler_y.inverse_transform(y_seq.reshape(-1, 1)).squeeze()

        metrics = ExperimentMetrics(
            mae=float(np.mean(np.abs(y_true - preds))),
            mse=float(np.mean((y_true - preds) ** 2))
        )

        return model, metrics

    @torch.inference_mode()
    def predict(self, steps: int):
        if self.model is None or self.df is None:
            raise ValueError("Сначала необходимо обучить модель")

        scaled_preds = []
        df = self.df.copy()

        try:
            for _ in range(steps):
                df[self.feature_columns] = df[self.feature_columns].astype(np.float32)
                X = df[self.feature_columns].values

                if len(X) < self.window:
                    raise ValueError(f"Недостаточно данных для окна ({len(X)} < {self.window})")

                X_scaled = self.scaler_X.transform(X[-self.window:])
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                y_pred_scaled = self.model(X_tensor).item()
                scaled_preds.append(y_pred_scaled)

                y_pred = self.scaler_y.inverse_transform([[y_pred_scaled]])[0][0]

                next_index = df.index[-1] + pd.Timedelta(days=1)
                new_row = pd.DataFrame(index=[next_index], columns=df.columns)
                df = pd.concat([df, new_row])

                time_features = create_time_features(new_row, return_new_colnames=False)
                for col in time_features.columns:
                    df.loc[next_index, col] = time_features.loc[next_index, col]

                df = update_extended_features_lastrow(df, self.target_column, y_pred)

        except Exception:
            logger.exception("Ошибка при прогнозировании LSTM")
            raise

        final_preds = self.scaler_y.inverse_transform(np.array(scaled_preds).reshape(-1, 1)).squeeze()
        return final_preds.tolist(), None
