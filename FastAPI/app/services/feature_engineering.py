import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats


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


n_lags = 30
ma_windows = (7, 14, 30)
ewm_windows = (7, 14, 30)
vol_windows = (7, 14, 30)

def create_extended_features(
    df: pd.DataFrame,
    target_column: str,
    n_lags: int = n_lags,
    ma_windows: tuple | list = ma_windows,
    ewm_windows: tuple | list = ewm_windows,
    vol_windows: tuple | list = vol_windows,
    return_new_colnames: bool = True,
    verbose: bool = True
) -> pd.DataFrame | tuple[pd.DataFrame, list]:
    """
    Создает расширенные финансовые признаки для целевого столбца.
    """
    # Словарь для хранения времени выполнения каждой секции
    timings = defaultdict(float)

    # Вспомогательная функция для замера времени
    def time_section(section_name, func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        timings[section_name] += elapsed
        return result

    total_start_time = time.time()

    original_cols = df.columns.tolist()
    base_df = df.copy()

    all_features = {}

    # Lag features
    def create_lag_features():
        lag_dfs = {}
        for lag in range(1, n_lags + 1):
            lag_dfs[f"{target_column}_lag_{lag}"] = base_df[target_column].shift(lag)
        return lag_dfs
    all_features.update(time_section("Lag features", create_lag_features))

    # Moving averages
    def create_ma_features():
        ma_dfs = {}
        for window in ma_windows:
            ma_dfs[f"{target_column}_ma_{window}"] = base_df[target_column].rolling(window=window, closed="left").mean()
        return ma_dfs
    ma_dfs = time_section("Moving averages (MA)", create_ma_features)
    all_features.update(ma_dfs)

    # Exponential moving averages
    def create_ewm_features():
        ewm_dfs = {}
        for window in ewm_windows:
            ewm_dfs[f"{target_column}_ema_{window}"] = base_df[target_column].ewm(span=window, adjust=False).mean()
        return ewm_dfs
    ewm_dfs = time_section("Exponential moving averages (EMA)", create_ewm_features)
    all_features.update(ewm_dfs)

    # Volatility
    def create_vol_features():
        vol_dfs = {}
        for window in vol_windows:
            vol_dfs[f"{target_column}_vol_{window}"] = base_df[target_column].rolling(window=window, closed="left").std()
        return vol_dfs
    vol_dfs = time_section("Volatility", create_vol_features)
    all_features.update(vol_dfs)

    # Percent changes
    def create_pct_change_features():
        pct_change_dfs = {}
        for lag in range(1, 6):
            pct_change_dfs[f"{target_column}_pct_change_{lag}"] = base_df[target_column].pct_change(periods=lag)
        return pct_change_dfs
    all_features.update(time_section("Percent changes", create_pct_change_features))

    # Rate of change
    def create_roc_features():
        roc_dfs = {}
        for window in ma_windows:
            roc_dfs[f"{target_column}_roc_{window}"] = (base_df[target_column] - base_df[target_column].shift(window)) / base_df[target_column].shift(window) * 100
        return roc_dfs
    all_features.update(time_section("Rate of change", create_roc_features))

    # Momentum indicators
    def create_momentum_features():
        momentum_dfs = {}
        for window in ma_windows:
            momentum_dfs[f"{target_column}_momentum_{window}"] = base_df[target_column] - base_df[target_column].shift(window)
        return momentum_dfs
    all_features.update(time_section("Momentum indicators", create_momentum_features))

    # RSI (Relative Strength Index)
    def create_rsi_features():
        rsi_dfs = {}
        delta = base_df[target_column].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        for window in (14, 21):
            avg_gain = gain.rolling(window=window, closed="left").mean()
            avg_loss = loss.rolling(window=window, closed="left").mean()
            rs = avg_gain / avg_loss
            rsi_dfs[f"{target_column}_rsi_{window}"] = 100 - (100 / (1 + rs))
        return rsi_dfs
    all_features.update(time_section("RSI", create_rsi_features))

    # MACD (Moving Average Convergence Divergence)
    def create_macd_features():
        macd_dfs = {}
        ema_12 = base_df[target_column].ewm(span=12, adjust=False).mean()
        ema_26 = base_df[target_column].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        macd_dfs[f"{target_column}_macd"] = macd
        macd_dfs[f"{target_column}_macd_signal"] = macd_signal
        macd_dfs[f"{target_column}_macd_hist"] = macd - macd_signal
        return macd_dfs
    all_features.update(time_section("MACD", create_macd_features))

    # Bollinger Bands
    def create_bb_features():
        bb_dfs = {}
        for window in ma_windows:
            ma_series = ma_dfs[f"{target_column}_ma_{window}"]
            vol_series = vol_dfs[f"{target_column}_vol_{window}"]

            bb_upper = ma_series + 2 * vol_series
            bb_lower = ma_series - 2 * vol_series
            bb_width = (bb_upper - bb_lower) / ma_series
            bb_pct = (base_df[target_column] - bb_lower) / (bb_upper - bb_lower)

            bb_dfs[f"{target_column}_bb_upper_{window}"] = bb_upper
            bb_dfs[f"{target_column}_bb_lower_{window}"] = bb_lower
            bb_dfs[f"{target_column}_bb_width_{window}"] = bb_width
            bb_dfs[f"{target_column}_bb_pct_{window}"] = bb_pct
        return bb_dfs
    all_features.update(time_section("Bollinger Bands", create_bb_features))

    # Statistical features
    def create_stat_features():
        stat_dfs = {}
        for window in [30]:
            # Skewness
            stat_dfs[f"{target_column}_skew_{window}"] = base_df[target_column].rolling(window=window, closed="left").apply(lambda x: stats.skew(x))
            # Kurtosis
            stat_dfs[f"{target_column}_kurt_{window}"] = base_df[target_column].rolling(window=window, closed="left").apply(lambda x: stats.kurtosis(x))
            # Max, Min, Range
            rolling_max = base_df[target_column].rolling(window=window, closed="left").max()
            rolling_min = base_df[target_column].rolling(window=window, closed="left").min()

            stat_dfs[f"{target_column}_max_{window}"] = rolling_max
            stat_dfs[f"{target_column}_min_{window}"] = rolling_min
            stat_dfs[f"{target_column}_range_{window}"] = rolling_max - rolling_min

            # Quantiles
            q25 = base_df[target_column].rolling(window=window, closed="left").quantile(0.25)
            q75 = base_df[target_column].rolling(window=window, closed="left").quantile(0.75)

            stat_dfs[f"{target_column}_q25_{window}"] = q25
            stat_dfs[f"{target_column}_q75_{window}"] = q75
            stat_dfs[f"{target_column}_iqr_{window}"] = q75 - q25
        return stat_dfs
    all_features.update(time_section("Statistical features", create_stat_features))

    # Z-score
    def create_zscore_features():
        zscore_dfs = {}
        for window in ma_windows:
            zscore_dfs[f"{target_column}_zscore_{window}"] = (base_df[target_column] - ma_dfs[f"{target_column}_ma_{window}"]) / vol_dfs[f"{target_column}_vol_{window}"]
        return zscore_dfs
    all_features.update(time_section("Z-score", create_zscore_features))

    # Distance from MA
    def create_dist_features():
        dist_dfs = {}
        for window in ma_windows:
            ma_series = ma_dfs[f"{target_column}_ma_{window}"]
            dist_dfs[f"{target_column}_dist_ma_{window}"] = base_df[target_column] - ma_series
            dist_dfs[f"{target_column}_dist_pct_ma_{window}"] = (base_df[target_column] / ma_series - 1) * 100
        return dist_dfs
    all_features.update(time_section("Distance from MA", create_dist_features))

    # Crossovers
    def create_cross_features():
        cross_dfs = {}
        for fast_window, slow_window in [(7, 14), (14, 30)]:
            if f"{target_column}_ma_{fast_window}" in ma_dfs and f"{target_column}_ma_{slow_window}" in ma_dfs:
                cross_dfs[f"{target_column}_cross_{fast_window}_{slow_window}"] = ma_dfs[f"{target_column}_ma_{fast_window}"] - ma_dfs[f"{target_column}_ma_{slow_window}"]
        return cross_dfs
    all_features.update(time_section("Crossovers", create_cross_features))

    # Williams %R
    def create_williams_features():
        williams_dfs = {}
        for window in ma_windows:
            highest_high = base_df[target_column].rolling(window=window, closed="left").max()
            lowest_low = base_df[target_column].rolling(window=window, closed="left").min()
            williams_dfs[f"{target_column}_williams_{window}"] = -100 * (highest_high - base_df[target_column]) / (highest_high - lowest_low)
        return williams_dfs
    all_features.update(time_section("Williams %R", create_williams_features))

    # Smoothed features
    def create_smooth_features():
        smooth_dfs = {}
        for window in ma_windows:
            smooth_dfs[f"{target_column}_ma_of_ma_{window}"] = ma_dfs[f"{target_column}_ma_{window}"].rolling(window=window, closed="left").mean()
            smooth_dfs[f"{target_column}_ema_of_ema_{window}"] = ewm_dfs[f"{target_column}_ema_{window}"].ewm(span=window, adjust=False).mean()
        return smooth_dfs
    all_features.update(time_section("Smoothed features", create_smooth_features))

    # Trend strength
    def create_trend_features():
        trend_dfs = {}
        for window in ma_windows:
            trend_dfs[f"{target_column}_trend_strength_{window}"] = np.abs(ma_dfs[f"{target_column}_ma_{window}"].diff(periods=window)) / vol_dfs[f"{target_column}_vol_{window}"]
        return trend_dfs
    all_features.update(time_section("Trend strength", create_trend_features))

    # Fourier Transform features
    def create_fft_features():
        fft_dfs = {}
        for window in [30, 60]:
            if len(base_df) >= window:
                fft_dfs[f"{target_column}_fft_dominant_{window}"] = base_df[target_column].rolling(window=window, closed="left").apply(
                    lambda x, w=window: np.abs(np.fft.fft(x - np.mean(x)))[1:w//2].argmax() + 1 if len(x) == window else np.nan
                )
        return fft_dfs
    all_features.update(time_section("Fourier Transform features", create_fft_features))

    def create_result_df():
        features_df = pd.DataFrame(all_features, index=base_df.index)
        result_df = pd.concat([base_df, features_df], axis=1)
        return result_df, features_df

    result_df, features_df = time_section("Create and concatenate DataFrame", create_result_df)

    total_time = time.time() - total_start_time
    if verbose:
        print(f"Total execution time: {total_time:.4f} сек")
        print("\nОперации, отсортированные по времени выполнения:")
        for section, elapsed in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            print(f"  {section}: {elapsed:.4f} сек ({elapsed/total_time*100:.1f}%)")

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
    df, _ = create_time_features(df, return_new_colnames=False)
    df, new_cols = create_extended_features(
        df,
        target_column=target_column,
        return_new_colnames=True,
        verbose=False
    )
    df = df.dropna()
    return (df, new_cols) if return_new_colnames else df

