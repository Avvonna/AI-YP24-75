import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

def create_time_features(
    df: pd.DataFrame,
    return_new_colnames: bool = True,
    exclude: str | list | None = "dayofweek"
):
    df_cols = df.columns
    df = df.copy()

    # Основные временные признаки
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["days_since_start"] = (df.index - df.index.min()).days

    # Синус и косинус для цикличных признаков (месяцы, дни недели)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    if exclude is not None:
        df = df.drop(exclude, axis=1)

    if return_new_colnames:
        return df, df.columns.difference(df_cols).to_list()
    return df


N_LAGS = 30
MA_WINDOWS = (7, 14, 30)
EWM_WINDOWS = (7, 14, 30)
VOL_WINDOWS = (7, 14, 30)

def create_extended_features(
    df: pd.DataFrame,
    target_column: str,
    n_lags: int = N_LAGS,
    ma_windows: tuple | list = MA_WINDOWS,
    ewm_windows: tuple | list = EWM_WINDOWS,
    vol_windows: tuple | list = VOL_WINDOWS,
    return_new_colnames: bool = True
) -> pd.DataFrame | tuple[pd.DataFrame, list]:
    """
    Создает расширенные финансовые признаки для целевого столбца.
    """
    if target_column not in df.columns:
        raise ValueError(f"Колонка '{target_column}' не найдена в DataFrame")
    if df.empty:
        raise ValueError("DataFrame не может быть пустым")

    original_cols = df.columns.tolist()
    base_df = df.copy()
    all_features = {}

    # Lag features
    lag_dfs = {
        f"{target_column}_lag_{lag}": base_df[target_column].shift(lag)
        for lag in range(1, n_lags + 1)
    }
    all_features.update(lag_dfs)

    # Moving averages
    ma_dfs = {
        f"{target_column}_ma_{window}": base_df[target_column].rolling(window=window, closed="left").mean()
        for window in ma_windows
    }
    all_features.update(ma_dfs)

    # Exponential moving averages
    ewm_dfs = {
        f"{target_column}_ema_{window}": base_df[target_column].ewm(span=window, adjust=False).mean()
        for window in ewm_windows
    }
    all_features.update(ewm_dfs)

    # Volatility
    vol_dfs = {
        f"{target_column}_vol_{window}": base_df[target_column].rolling(window=window, closed="left").std()
        for window in vol_windows
    }
    all_features.update(vol_dfs)

    # Percent changes
    pct_change_dfs = {
        f"{target_column}_pct_change_{lag}": base_df[target_column].pct_change(periods=lag)
        for lag in range(1, 6)
    }
    all_features.update(pct_change_dfs)

    # Rate of change
    for window in ma_windows:
        window_value = base_df[target_column].shift(window-1)
        mask = window_value != 0
        roc_series = pd.Series(index=base_df.index, dtype=float)
        roc_series[mask] = (
            (base_df[target_column][mask] - window_value[mask]) / window_value[mask] * 100
        )
        all_features[f"{target_column}_roc_{window}"] = roc_series

    # Momentum
    for window in ma_windows:
        all_features[f"{target_column}_momentum_{window}"] = (
            base_df[target_column] - base_df[target_column].shift(window-1)
        )

    # RSI
    delta = base_df[target_column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for window in (14, 21):
        avg_gain = gain.rolling(window=window, closed="left").mean()
        avg_loss = loss.rolling(window=window, closed="left").mean()
        rs = avg_gain / avg_loss
        all_features[f"{target_column}_rsi_{window}"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = base_df[target_column].ewm(span=12, adjust=False).mean()
    ema_26 = base_df[target_column].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    all_features[f"{target_column}_macd"] = macd
    all_features[f"{target_column}_macd_signal"] = macd_signal
    all_features[f"{target_column}_macd_hist"] = macd_hist

    # Bollinger Bands
    for window in ma_windows:
        ma_series = ma_dfs[f"{target_column}_ma_{window}"]
        vol_series = vol_dfs[f"{target_column}_vol_{window}"]
        bb_upper = ma_series + 2 * vol_series
        bb_lower = ma_series - 2 * vol_series
        all_features[f"{target_column}_bb_upper_{window}"] = bb_upper
        all_features[f"{target_column}_bb_lower_{window}"] = bb_lower
        all_features[f"{target_column}_bb_width_{window}"] = (bb_upper - bb_lower) / ma_series
        all_features[f"{target_column}_bb_pct_{window}"] = (base_df[target_column] - bb_lower) / (bb_upper - bb_lower)

    # Statistical features
    def safe_skew(x):
        try:
            return stats.skew(x, nan_policy="omit") if len(x) >= 3 else np.nan
        except Exception:
            return np.nan

    def safe_kurtosis(x):
        try:
            return stats.kurtosis(x, nan_policy="omit") if len(x) >= 4 else np.nan
        except Exception:
            return np.nan

    for window in [30]:
        rolling = base_df[target_column].rolling(window=window, closed="left")
        all_features[f"{target_column}_skew_{window}"] = rolling.apply(safe_skew)
        all_features[f"{target_column}_kurt_{window}"] = rolling.apply(safe_kurtosis)
        rolling_max = rolling.max()
        rolling_min = rolling.min()
        all_features[f"{target_column}_max_{window}"] = rolling_max
        all_features[f"{target_column}_min_{window}"] = rolling_min
        all_features[f"{target_column}_range_{window}"] = rolling_max - rolling_min
        q25 = rolling.quantile(0.25)
        q75 = rolling.quantile(0.75)
        all_features[f"{target_column}_q25_{window}"] = q25
        all_features[f"{target_column}_q75_{window}"] = q75
        all_features[f"{target_column}_iqr_{window}"] = q75 - q25

    # Z-score
    for window in ma_windows:
        ma = ma_dfs[f"{target_column}_ma_{window}"]
        vol = vol_dfs[f"{target_column}_vol_{window}"]
        all_features[f"{target_column}_zscore_{window}"] = np.where(vol != 0, (base_df[target_column] - ma) / vol, 0)

    # Distance from MA
    for window in ma_windows:
        ma_series = ma_dfs[f"{target_column}_ma_{window}"]
        all_features[f"{target_column}_dist_ma_{window}"] = base_df[target_column] - ma_series
        all_features[f"{target_column}_dist_pct_ma_{window}"] = np.where(
            ma_series != 0, (base_df[target_column] / ma_series - 1) * 100, 0
        )

    # Crossovers
    for fast_window, slow_window in [(7, 14), (14, 30)]:
        fast_col = f"{target_column}_ma_{fast_window}"
        slow_col = f"{target_column}_ma_{slow_window}"
        if fast_col in ma_dfs and slow_col in ma_dfs:
            all_features[f"{target_column}_cross_{fast_window}_{slow_window}"] = (
                ma_dfs[fast_col] - ma_dfs[slow_col]
            )

    # Williams %R
    for window in ma_windows:
        high = base_df[target_column].rolling(window=window, closed="left").max()
        low = base_df[target_column].rolling(window=window, closed="left").min()
        range_hl = high - low
        all_features[f"{target_column}_williams_{window}"] = np.where(
            range_hl != 0, -100 * (high - base_df[target_column]) / range_hl, -50
        )

    # Smoothed features
    for window in ma_windows:
        ma_series = ma_dfs[f"{target_column}_ma_{window}"]
        ema_series = ewm_dfs[f"{target_column}_ema_{window}"]
        all_features[f"{target_column}_ma_of_ma_{window}"] = ma_series.rolling(window=window, closed="left").mean()
        all_features[f"{target_column}_ema_of_ema_{window}"] = ema_series.ewm(span=window, adjust=False).mean()

    # Trend strength
    for window in ma_windows:
        ma_series = ma_dfs[f"{target_column}_ma_{window}"]
        vol_series = vol_dfs[f"{target_column}_vol_{window}"]
        ma_diff = ma_series.diff(periods=window)
        all_features[f"{target_column}_trend_strength_{window}"] = np.where(
            vol_series != 0, np.abs(ma_diff) / vol_series, 0
        )

    # Fourier features
    for window in [30, 60]:
        if len(base_df) >= window:
            rolling = base_df[target_column].rolling(window=window, closed="left")
            all_features[f"{target_column}_fft_dominant_{window}"] = rolling.apply(
                lambda x, w=window: (np.abs(np.fft.fft(x - np.mean(x)))[1:w//2].argmax() + 1) if len(x) == w else np.nan
            )

    features_df = pd.DataFrame(all_features, index=base_df.index)
    result_df = pd.concat([base_df, features_df], axis=1)

    logger.info(f"Добавлено признаков: {len(features_df.columns)}")

    if return_new_colnames:
        new_cols = [col for col in result_df.columns if col not in original_cols]
        return result_df, new_cols
    else:
        return result_df


def preprocess_for_model(
    df: pd.DataFrame,
    target_column: str,
    return_new_colnames: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    df = create_time_features(df, return_new_colnames=False)
    df, new_cols = create_extended_features(
        df,
        target_column=target_column,
        return_new_colnames=True
    )
    df = df.dropna()
    return (df, new_cols) if return_new_colnames else df


def update_extended_features_lastrow(
    df: pd.DataFrame,
    target_column: str,
    new_target_value: float,
    n_lags: int = N_LAGS,
    ma_windows: tuple | list = MA_WINDOWS,
    ewm_windows: tuple | list = EWM_WINDOWS,
    vol_windows: tuple | list = VOL_WINDOWS,
) -> pd.DataFrame:
    """
    Обновляет последнюю строку датасета с расширенными признаками, заменяя значение целевой
    переменной на заданное и пересчитывая все признаки, созданные функцией create_extended_features.
    """

    if df.empty:
        raise ValueError("DataFrame не может быть пустым")
    if len(df) < 2:
        raise ValueError("DataFrame должен содержать минимум 2 строки")
    if target_column not in df.columns:
        raise ValueError(f"Колонка '{target_column}' не найдена в DataFrame")
    if not isinstance(new_target_value, (int, float)) or np.isnan(new_target_value):
        raise ValueError("new_target_value должно быть числом")

    result = df.copy()
    last_index = result.index[-1]
    prev_row_index = result.index[-2]

    # Обновляем значение целевой переменной
    result.loc[last_index, target_column] = new_target_value

    # Lag features
    for lag in range(1, n_lags + 1):
        if lag == 1:
            result.loc[last_index, f"{target_column}_lag_{lag}"] = (
                result.loc[prev_row_index, target_column]
            )
        else:
            result.loc[last_index, f"{target_column}_lag_{lag}"] = (
                result.loc[prev_row_index, f"{target_column}_lag_{lag-1}"]
            )

    # Moving averages
    for window in ma_windows:
        values = result.loc[:prev_row_index, target_column].iloc[-(window-1):].values
        if len(values) == window-1:
            ma_value = (np.sum(values) + new_target_value) / window
            result.loc[last_index, f"{target_column}_ma_{window}"] = ma_value

    # Exponential moving averages
    for window in ewm_windows:
        alpha = 2 / (window + 1)
        prev_ema = result.loc[prev_row_index, f"{target_column}_ema_{window}"]
        new_ema = alpha * new_target_value + (1 - alpha) * prev_ema
        result.loc[last_index, f"{target_column}_ema_{window}"] = new_ema

    # Volatility
    for window in vol_windows:
        values = result.loc[:prev_row_index, target_column].iloc[-(window-1):].values
        if len(values) == window-1:
            full_window = np.append(values, new_target_value)
            vol_value = np.std(full_window, ddof=1)
            result.loc[last_index, f"{target_column}_vol_{window}"] = vol_value

    # Percent changes
    for lag in range(1, 6):
        if lag == 1:
            prev_value = result.loc[prev_row_index, target_column]
            if prev_value != 0:
                pct_change = (new_target_value - prev_value) / prev_value
                result.loc[last_index, f"{target_column}_pct_change_{lag}"] = pct_change
        else:
            # Для lag > 1 используем значение из соответствующего lag
            lag_col = f"{target_column}_lag_{lag}"
            if lag_col in result.columns:
                prev_value = result.loc[last_index, lag_col]  # Используем обновленное значение lag
                if prev_value != 0:
                    pct_change = (new_target_value - prev_value) / prev_value
                    result.loc[last_index, f"{target_column}_pct_change_{lag}"] = pct_change

    # Rate of change
    for window in ma_windows:
        if window <= n_lags:
            window_value = result.loc[prev_row_index, f"{target_column}_lag_{window-1}"]
        else:
            window_value = (
                result.loc[:prev_row_index, target_column]
                .shift(window-1)
                .iloc[-1]
            )
        if window_value != 0:
            roc = (new_target_value - window_value) / window_value * 100
            result.loc[last_index, f"{target_column}_roc_{window}"] = roc

    # Momentum indicators
    for window in ma_windows:
        if window <= n_lags:
            window_value = result.loc[prev_row_index, f"{target_column}_lag_{window-1}"]
        else:
            window_value = (
                result.loc[:prev_row_index, target_column]
                .shift(window-1)
                .iloc[-1]
            )
        momentum = new_target_value - window_value
        result.loc[last_index, f"{target_column}_momentum_{window}"] = momentum

    # RSI (Relative Strength Index)
    for window in (14, 21):
        delta = new_target_value - result.loc[prev_row_index, target_column]
        gain = max(0, delta)
        loss = max(0, -delta)

        # Получаем предыдущие средние приросты и убытки
        prev_rows = result.iloc[-(window+1):-1]  # Берем window предыдущих строк
        prev_deltas = prev_rows[target_column].diff().iloc[1:]  # Пропускаем первую NaN
        prev_gains = prev_deltas.clip(lower=0)
        prev_losses = -prev_deltas.clip(upper=0)

        if len(prev_rows) >= window:
            # Если есть достаточно предыдущих строк
            avgain_col = f"{target_column}_rsi_{window}_avgain"
            if avgain_col in result.columns:
                prev_avg_gain = result.loc[prev_row_index, avgain_col]
            else:
                prev_avg_gain = prev_gains.mean()

            avloss_col = f"{target_column}_rsi_{window}_avloss"
            if avloss_col in result.columns:
                prev_avg_loss = result.loc[prev_row_index, avloss_col]
            else:
                prev_avg_loss = prev_losses.mean()

            avg_gain = (prev_avg_gain * (window - 1) + gain) / window
            avg_loss = (prev_avg_loss * (window - 1) + loss) / window

            result.loc[last_index, avgain_col] = avg_gain
            result.loc[last_index, avloss_col] = avg_loss
        else:
            # Если недостаточно данных, используем простое среднее
            avg_gain = (prev_gains.sum() + gain) / max(1, len(prev_gains))
            avg_loss = (prev_losses.sum() + loss) / max(1, len(prev_losses))

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        result.loc[last_index, f"{target_column}_rsi_{window}"] = rsi

    # MACD (Moving Average Convergence Divergence)
    ema12_col = f"{target_column}_ema_12"
    if ema12_col in result.columns:
        ema_12 = result.loc[last_index, ema12_col]
    else:
        ema_12 = (
            new_target_value * 0.15 +
            result.loc[prev_row_index, target_column] * 0.85
        )

    ema26_col = f"{target_column}_ema_26"
    if ema26_col in result.columns:
        ema_26 = result.loc[last_index, ema26_col]
    else:
        ema_26 = (
            new_target_value * 0.07 +
            result.loc[prev_row_index, target_column] * 0.93
        )

    macd = ema_12 - ema_26
    result.loc[last_index, f"{target_column}_macd"] = macd

    prev_macd_signal = result.loc[prev_row_index, f"{target_column}_macd_signal"]
    macd_signal = macd * 0.2 + prev_macd_signal * 0.8
    result.loc[last_index, f"{target_column}_macd_signal"] = macd_signal

    macd_hist = macd - macd_signal
    result.loc[last_index, f"{target_column}_macd_hist"] = macd_hist

    # Bollinger Bands
    for window in ma_windows:
        ma_value = result.loc[last_index, f"{target_column}_ma_{window}"]
        vol_value = result.loc[last_index, f"{target_column}_vol_{window}"]

        bb_upper = ma_value + 2 * vol_value
        bb_lower = ma_value - 2 * vol_value
        bb_width = (bb_upper - bb_lower) / ma_value if ma_value != 0 else 0
        bb_pct = (new_target_value - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5

        result.loc[last_index, f"{target_column}_bb_upper_{window}"] = bb_upper
        result.loc[last_index, f"{target_column}_bb_lower_{window}"] = bb_lower
        result.loc[last_index, f"{target_column}_bb_width_{window}"] = bb_width
        result.loc[last_index, f"{target_column}_bb_pct_{window}"] = bb_pct

    # Statistical features
    for window in [30]:
        values = np.append(result.loc[:prev_row_index, target_column].iloc[-(window-1):].values, new_target_value)

        if len(values) == window:
            # Скошенность (Skewness)
            skew = stats.skew(values)
            result.loc[last_index, f"{target_column}_skew_{window}"] = skew

            # Эксцесс (Kurtosis)
            kurt = stats.kurtosis(values)
            result.loc[last_index, f"{target_column}_kurt_{window}"] = kurt

            # Максимум, минимум, размах
            rolling_max = np.max(values)
            rolling_min = np.min(values)

            result.loc[last_index, f"{target_column}_max_{window}"] = rolling_max
            result.loc[last_index, f"{target_column}_min_{window}"] = rolling_min
            result.loc[last_index, f"{target_column}_range_{window}"] = rolling_max - rolling_min

            # Квантили
            q25 = np.percentile(values, 25)
            q75 = np.percentile(values, 75)

            result.loc[last_index, f"{target_column}_q25_{window}"] = q25
            result.loc[last_index, f"{target_column}_q75_{window}"] = q75
            result.loc[last_index, f"{target_column}_iqr_{window}"] = q75 - q25

    # Z-score
    for window in ma_windows:
        ma_value = result.loc[last_index, f"{target_column}_ma_{window}"]
        vol_value = result.loc[last_index, f"{target_column}_vol_{window}"]

        if vol_value != 0:
            zscore = (new_target_value - ma_value) / vol_value
            result.loc[last_index, f"{target_column}_zscore_{window}"] = zscore

    # Distance from MA
    for window in ma_windows:
        ma_value = result.loc[last_index, f"{target_column}_ma_{window}"]

        dist = new_target_value - ma_value
        dist_pct = (new_target_value / ma_value - 1) * 100 if ma_value != 0 else 0

        result.loc[last_index, f"{target_column}_dist_ma_{window}"] = dist
        result.loc[last_index, f"{target_column}_dist_pct_ma_{window}"] = dist_pct

    # Crossovers
    for fast_window, slow_window in [(7, 14), (14, 30)]:
        fast_ma_col = f"{target_column}_ma_{fast_window}"
        slow_ma_col = f"{target_column}_ma_{slow_window}"
        if fast_ma_col in result.columns and slow_ma_col in result.columns:
            fast_ma = result.loc[last_index, fast_ma_col]
            slow_ma = result.loc[last_index, slow_ma_col]
            cross = fast_ma - slow_ma
            result.loc[last_index, f"{target_column}_cross_{fast_window}_{slow_window}"] = cross

    # Williams %R
    for window in ma_windows:
        values = np.append(result.loc[:prev_row_index, target_column].iloc[-(window-1):].values, new_target_value)

        if len(values) == window:
            highest_high = np.max(values)
            lowest_low = np.min(values)

            if highest_high != lowest_low:
                williams_r = -100 * (highest_high - new_target_value) / (highest_high - lowest_low)
                result.loc[last_index, f"{target_column}_williams_{window}"] = williams_r

    # Smoothed features
    for window in ma_windows:
        ma_values = result.loc[:prev_row_index, f"{target_column}_ma_{window}"].iloc[-(window-1):].values
        if len(ma_values) == window-1:
            current_ma = result.loc[last_index, f"{target_column}_ma_{window}"]
            ma_of_ma = (np.sum(ma_values) + current_ma) / window
            result.loc[last_index, f"{target_column}_ma_of_ma_{window}"] = ma_of_ma

        prev_ema_of_ema = result.loc[prev_row_index, f"{target_column}_ema_of_ema_{window}"]
        current_ema = result.loc[last_index, f"{target_column}_ema_{window}"]
        alpha = 2 / (window + 1)
        ema_of_ema = alpha * current_ema + (1 - alpha) * prev_ema_of_ema
        result.loc[last_index, f"{target_column}_ema_of_ema_{window}"] = ema_of_ema

    # Trend strength
    for window in ma_windows:
        if f"{target_column}_ma_{window}" in result.columns:
            window_index = None
            try:
                current_pos = result.index.get_loc(last_index)
                if current_pos >= window:
                    window_index = result.index[current_pos - window]
            except Exception as e:
                print(f"Trend strength error: {e}")

            if window_index is not None and window_index in result.index:
                previous_ma = result.loc[window_index, f"{target_column}_ma_{window}"]
                current_ma = result.loc[last_index, f"{target_column}_ma_{window}"]
                ma_diff = current_ma - previous_ma

                vol_value = result.loc[last_index, f"{target_column}_vol_{window}"]
                if vol_value != 0:
                    trend_strength = abs(ma_diff) / vol_value
                    result.loc[last_index, f"{target_column}_trend_strength_{window}"] = trend_strength
                else:
                    result.loc[last_index, f"{target_column}_trend_strength_{window}"] = np.nan
            else:
                result.loc[last_index, f"{target_column}_trend_strength_{window}"] = np.nan
        else:
            result.loc[last_index, f"{target_column}_trend_strength_{window}"] = np.nan

    # Fourier Transform features
    for window in [30, 60]:
        if len(result) >= window:
            values = np.append(result.loc[:prev_row_index, target_column].iloc[-(window-1):].values, new_target_value)
            if len(values) == window:
                # Удаляем среднее значение для лучшего анализа частот
                values_centered = values - np.mean(values)
                # Вычисляем FFT
                fft_values = np.abs(np.fft.fft(values_centered))
                # Определяем доминирующую частоту (первую половину спектра без постоянной составляющей)
                dominant_freq = fft_values[1:window//2].argmax() + 1
                result.loc[last_index, f"{target_column}_fft_dominant_{window}"] = dominant_freq

    return result
