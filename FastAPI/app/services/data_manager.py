import logging
from pathlib import Path
from typing import Any, List
import pandas as pd
from app.models.schemas import SingleTickerData

logger = logging.getLogger(__name__)


class DataManager:
    """В классе реализованы методы для работы с данными котировок"""
    def __init__(self, data_dir: str = "app/tickers-data"):
        """
        Инициализирует DataManager

        Attributes:
            data_dir (Path): Путь к директории с данными доступных тикеров
            tickers (List): Список доступных тикеров
            data (pd.DataFrame): DataFrame с данными тикеров
        """
        self.data_dir = Path(data_dir)
        self.tickers: list[str] = self._search_available_tickers()
        self.data: pd.DataFrame = self._load_all_data()
        logger.info(f"Загружено {len(self.data)} строк данных из {data_dir}")

    def _search_available_tickers(self) -> List[str]:
        """
        Проверяет наличие доступных данных
        и возвращает список доступных тикеров
        """
        tickers = []
        for file in self.data_dir.glob('*.csv'):
            ticker = file.stem.upper()
            tickers.append(ticker)
        return tickers

    def _load_all_data(self) -> dict:
        """Загружает данные доступных тикеров"""
        data = {}
        for ticker in self.tickers:
            file_path = self.data_dir / f"{ticker}.csv"
            try:
                df = pd.read_csv(file_path, parse_dates=["date"])
                df.sort_index(inplace=True)
                data[ticker] = df
                logger.debug(f"Загружены данные для {ticker}")
            except Exception as e:
                logger.error(f"Ошибка загрузки {file_path}: {str(e)}")
                data[ticker] = pd.DataFrame()
        return data

    def get_available_tickers(self) -> list[str]:
        """Возвращает список доступных тикеров"""
        return self.tickers

    def add_ticker_data(self, ticker: str, data: pd.DataFrame) -> None:
        """
        Добавляет данные нового тикера или
        обновляет данные уже имеющегося тикера
        Args:
            ticker: Название добавляемого тикера
            data: DataFrame с данными добавляемого тикера
        """
        ticker = ticker.upper()
        data.columns = [
            col.upper() if col != "date" else col
            for col in data.columns
            ]

        try:
            validated_df = SingleTickerData.validate_dataframe(data, ticker)
            if ticker not in self.tickers:
                self.tickers.append(ticker)
                logger.info(f"Добавлен новый тикер: {ticker}")
            if ticker in self.data:
                existing_data = self.data[ticker]
                updated_data = (
                    pd.concat([existing_data, validated_df])
                    .drop_duplicates("date")
                    .sort_values("date")
                )
                self.data[ticker] = updated_data
                logger.info(f"Данные для тикера {ticker} обновлены")
            else:
                self.data[ticker] = validated_df
                logger.info(f"Добавлены новые данные для тикера {ticker}")

        except Exception as e:
            logger.error(f"Ошибка добавления данных для {ticker}: {str(e)}")
            raise
        file_path = self.data_dir / f"{ticker}.csv"

        try:
            data.to_csv(file_path, index=False)
            if ticker not in self.tickers:
                self.tickers.append(ticker)
                logger.info(f"Добавлен новый тикер: {ticker}")
            logger.info(f"Данные для {ticker} сохранены в {file_path}")

        except Exception as e:
            logger.error(f"Ошибка сохранения данных для {ticker}: {str(e)}")
            raise

    def remove_ticker(self, ticker: str) -> None:
        """
        Удаляет тикер из списка доступных тикеров,
        а так же удаляет данные тикера
        Args:
            ticker: Название удаляемого тикера
        """
        ticker = ticker.upper()

        if ticker not in self.tickers:
            logger.warning(f"Тикер {ticker} не найден")
            return
        file_path = self.data_dir / f"{ticker}.csv"
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Данные тикера {file_path} удален")
        except Exception as e:
            logger.error(
                f"Ошибка удаления данных тикера {file_path}: {str(e)}"
                )
            raise

        self.tickers.remove(ticker)

        del self.data[ticker]
        logger.info(f"Тикер {ticker} полностью удален")

    def get_ticker_history(
            self,
            ticker: str,
            start_date: str = None,
            end_date: str = None
            ) -> dict[str, Any]:
        """
        Возвращает отфильтрованные по дате данные тикера,
        если временные рамки не указаны, то возвращает все данные
        Args:
            ticker: Название тикера
            start_date: Дата начала периода
            end_date: Дата окончания периода
        """

        ticker = ticker.upper()

        if ticker not in self.tickers:
            raise ValueError(f'{ticker} не найден')

        try:
            temp_df = self.data[ticker].copy()
            valid_data = temp_df.dropna(subset=[ticker])
            min_date = valid_data["date"].min()
            max_date = valid_data["date"].max()

            if start_date is None:
                start_date = min_date
            else:
                start_date = pd.to_datetime(start_date)

            if end_date is None:
                end_date = max_date
            else:
                end_date = pd.to_datetime(end_date)
            mask = (
                (temp_df["date"] >= start_date)
                & (temp_df["date"] <= end_date)
            )
            filtered_df = temp_df.loc[mask]

            return {
                "ticker": ticker,
                "dates": filtered_df["date"].dt.strftime('%Y-%m-%d').tolist(),
                "values": filtered_df[ticker].tolist(),
            }
        except Exception as e:
            raise Exception(
                f"Ошибка при получении данных для {ticker}: {str(e)}"
                ) from e

    def filter_data_for_training(
            self,
            ticker: str,
            base_date: pd.Timestamp,
            window: int = 60
            ):
        """
        Возвращает заданное количество (по умолчанию - 60) timestamps
        до заданной даты
        Args:
            ticker: Название тикера
            base_date: Конечная дата
            window: Количество timestamps до конечной даты
        """
        min_date = self.data[ticker]["date"].min()
        max_date = self.data[ticker]["date"].max()

        original_base_date = base_date
        base_date = min(base_date, max_date)
        border = max(base_date - pd.Timedelta(days=window), min_date)

        df = self.data[ticker][
            (self.data[ticker]["date"] >= border)
            & (self.data[ticker]["date"] <= base_date)
            ]

        if df.empty:
            logger.warning(
                f"Пустой DataFrame: тикер='{ticker}', окно={window} дней, "
                f"дата={original_base_date.date()},"
                f"обрезано до {base_date.date()}"
            )
            raise ValueError(
                f"Нет данных для тикера '{ticker}'"
                f"в интервале {border.date()} — {base_date.date()}"
            )

        if original_base_date != base_date:
            logger.info(
                f"Дата base_date обрезана с {original_base_date.date()}"
                f"до {base_date.date()} "
                f"(тикер: {ticker})"
            )

        logger.debug(
            f"Отобрано {len(df)} строк"
            f"для обучения по тикеру {ticker}"
            )

        return df[["date", ticker]]
