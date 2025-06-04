from typing import Optional

import torch
from torch import nn
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeSeriesDataset(Dataset):
    """
    Класс датасета для временных рядов.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class LSTMForecaster(nn.Module):
    """
    LSTM модель для прогнозирования временных рядов.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def fit_model(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    epochs: int = 100,
    patience: int = 15,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    save_path: str = "best_lstm_model.pt"
):
    """
    Функция обучения модели с ранней остановкой и сохранением лучшей версии по валидации.
    """
    model.to(DEVICE)
    train_losses, val_losses = [], []
    best_val = float("inf")
    stop_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_loader))

        # Валидация
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = loss_fn(predictions, y_batch.unsqueeze(1))
                epoch_val_loss += loss.item()
        val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            stop_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            stop_count += 1
            if stop_count >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}")

    # Загрузка лучшей модели
    model.load_state_dict(torch.load(save_path))
    return model, train_losses, val_losses
