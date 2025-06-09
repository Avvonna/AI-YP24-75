from datetime import datetime
from typing import Optional

import pandas as pd


def validate_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Проверяет корректность датафрейма для добавления нового тикера:
    наличие нужных колонок, правильный тип данных и сортировку по дате.

    Args:
        df (pd.DataFrame): Датафрейм с колонками "date" и <ticker>
        ticker (str): Название тикера

    Returns:
        pd.DataFrame: Валидированный и отсортированный датафрейм
    """

    if "date" not in df.columns:
        raise ValueError("DataFrame должен содержать колонку 'date'")

    if ticker not in df.columns:
        raise ValueError(f"DataFrame должен содержать колонку '{ticker}'")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            raise ValueError(f"Не удалось преобразовать колонку 'date' в datetime: {str(e)}") from e

    if not pd.api.types.is_numeric_dtype(df[ticker]):
        raise ValueError(f"Колонка '{ticker}' должна содержать числовые значения")

    return (
        df[["date", ticker]]
        .rename(columns={ticker: ticker})
        .sort_values("date")
    )

def validate_iso_date(v: Optional[str]) -> Optional[str]:
    if not v or v.strip() == "":
        return None
    try:
        datetime.fromisoformat(v)
        return v
    except ValueError as e:
        raise ValueError("Неверный формат даты. Используйте YYYY-MM-DD") from e
