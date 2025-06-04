import numpy as np
import pandas as pd
from app.services.feature_engineering import create_extended_features, update_extended_features_lastrow


def generate_test_df(n=100):
    index = pd.date_range(start="2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({"target": np.arange(n)}, index=index)
    df_ext = create_extended_features(df, "target", return_new_colnames=False)
    return df_ext

def test_no_nans_in_last_row():
    df = generate_test_df()
    next_day = df.index[-1] + pd.Timedelta(days=1)
    df.loc[next_day] = np.nan

    updated = update_extended_features_lastrow(df, "target", 123.0)
    nan_cols = updated.loc[next_day].isna()
    assert not nan_cols.any(), f"NaNs in columns: {nan_cols[nan_cols].index.tolist()}"

def test_lag_features_correctness():
    df = generate_test_df()
    next_day = df.index[-1] + pd.Timedelta(days=1)
    df.loc[next_day] = np.nan
    last_value = df["target"].iloc[-2]

    updated = update_extended_features_lastrow(df, "target", new_target_value=1000.0)

    assert updated.loc[next_day, "target_lag_1"] == last_value
    assert updated.loc[next_day, "target_lag_2"] == df.loc[df.index[-2], "target_lag_1"]

def test_macd_consistency():
    df = generate_test_df()
    next_day = df.index[-1] + pd.Timedelta(days=1)
    df.loc[next_day] = np.nan

    updated = update_extended_features_lastrow(df, "target", 1000.0)

    macd = updated.loc[next_day, "target_macd"]
    signal = updated.loc[next_day, "target_macd_signal"]
    hist = updated.loc[next_day, "target_macd_hist"]

    assert np.isclose(macd - signal, hist, atol=1e-5), "MACD histogram mismatch"

def test_bollinger_bands_ordering():
    df = generate_test_df()
    next_day = df.index[-1] + pd.Timedelta(days=1)
    df.loc[next_day] = np.nan

    updated = update_extended_features_lastrow(df, "target", 1000.0)

    for window in (7, 14, 30):
        upper = updated.loc[next_day, f"target_bb_upper_{window}"]
        lower = updated.loc[next_day, f"target_bb_lower_{window}"]
        assert upper >= lower, f"Bollinger Bands invalid order for window {window}"

def test_rsi_range():
    df = generate_test_df()
    next_day = df.index[-1] + pd.Timedelta(days=1)
    df.loc[next_day] = np.nan

    updated = update_extended_features_lastrow(df, "target", 1000.0)

    for window in (14, 21):
        rsi = updated.loc[next_day, f"target_rsi_{window}"]
        assert 0 <= rsi <= 100, f"RSI out of bounds: {rsi}"

def test_trend_strength_finite():
    df = generate_test_df()
    next_day = df.index[-1] + pd.Timedelta(days=1)
    df.loc[next_day] = np.nan

    updated = update_extended_features_lastrow(df, "target", 1000.0)

    for window in (7, 14, 30):
        val = updated.loc[next_day, f"target_trend_strength_{window}"]
        assert np.isfinite(val) or np.isnan(val), "Trend strength must be finite or NaN"

def test_fft_dominant_is_positive_integer():
    df = generate_test_df()
    next_day = df.index[-1] + pd.Timedelta(days=1)
    df.loc[next_day] = np.nan

    updated = update_extended_features_lastrow(df, "target", 1000.0)
    for window in (30, 60):
        col = f"target_fft_dominant_{window}"
        if col in updated.columns:
            val = updated.loc[next_day, col]
            if not np.isnan(val):
                assert isinstance(val, (int, float, np.integer, np.floating)), f"FFT value is not numeric: {val}"
                assert val > 0, f"FFT value is not positive: {val}"
                assert float(val).is_integer(), f"FFT value is not close to integer: {val}"

def test_pct_change_consistency():
    df = generate_test_df()
    next_day = df.index[-1] + pd.Timedelta(days=1)
    df.loc[next_day] = np.nan
    prev_value = df["target"].iloc[-2]

    updated = update_extended_features_lastrow(df, "target", 1000.0)

    pct1 = updated.loc[next_day, "target_pct_change_1"]
    expected = (1000.0 - prev_value) / prev_value
    assert np.isclose(pct1, expected, rtol=1e-5), f"pct_change_1 incorrect: {pct1} != {expected}"


