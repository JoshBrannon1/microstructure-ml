import pytest
import polars as pl
from src.microstructure_ml.feature_pipeline import rolling_spread, depth_imbalance, mid_price, rolling_spread, spread, imbalance, microprice

def test_mid_price_calculation():
    df = pl.DataFrame({
        "bid_price_1": [100.0, 101.0],
        "bid_size_1": [1.0, 2.0],
        "ask_price_1": [102.0, 103.0],
        "ask_size_1": [1.5, 3.0]
    })
    result = df.with_columns([mid_price()])
    assert result["mid_price"].to_list() == [101.0, 102.0]

def test_spread_calculation():
    df = pl.DataFrame({
        "bid_price_1": [100.0, 101.0],
        "bid_size_1": [1.0, 2.0],
        "ask_price_1": [102.0, 103.0],
        "ask_size_1": [1.5, 3.0]
    })
    result = df.with_columns([spread()])
    assert result["spread"].to_list() == [2.0, 2.0]

def test_imbalance_calculation():
    df = pl.DataFrame({
        "bid_price_1": [100.0, 101.0],
        "bid_size_1": [1.0, 2.0],
        "ask_price_1": [102.0, 103.0],
        "ask_size_1": [1.5, 3.0]
    })
    result = df.with_columns([imbalance()])
    assert result["imbalance"].to_list() == [-0.2, -0.2]

def test_microprice_calculation():
    df = pl.DataFrame({
        "bid_price_1": [100.0, 101.0],
        "bid_size_1": [1.0, 2.0],
        "ask_price_1": [102.0, 103.0],
        "ask_size_1": [1.5, 3.0]
    })
    intermediate = df.with_columns([mid_price(), imbalance()])
    df = intermediate.with_columns([microprice()])
    assert df["microprice"].to_list() == [100.8, 101.8]

def test_depth_imbalance_calculation():
    df = pl.DataFrame({
        "bid_size_1": [1.0, 2.0],
        "bid_size_2": [0.5, 1.0],
        "bid_size_3": [0.2, 0.5],
        "bid_size_4": [0.1, 0.2],
        "bid_size_5": [0.05, 0.1],
        "bid_size_6": [0.02, 0.05],
        "bid_size_7": [0.01, 0.02],
        "bid_size_8": [0.005, 0.01],
        "bid_size_9": [0.002, 0.005],
        "bid_size_10": [0.001, 0.002],
        "ask_size_1": [1.5, 3.0],
        "ask_size_2": [0.75, 1.5],
        "ask_size_3": [0.3, 0.75],
        "ask_size_4": [0.15, 0.3],
        "ask_size_5": [0.075, 0.15],
        "ask_size_6": [0.03, 0.075],
        "ask_size_7": [0.015, 0.03],
        "ask_size_8": [0.0075, 0.015],
        "ask_size_9": [0.003, 0.0075],
        "ask_size_10": [0.0015, 0.003]
    })
    result = df.with_columns([depth_imbalance()])
    assert result["depth_imbalance"].to_list() == pytest.approx([-0.2,-0.2], rel=1e-4)

def test_rolling_spread_calculation():
    df = pl.DataFrame({
        "bid_price_1": [99.0, 99.5, 100.0, 100.5],
        "ask_price_1": [100.0, 102.0, 103.0, 104.0],
    })
    result = df.with_columns([rolling_spread(3)])
    assert result["rolling_spread_3"].to_list() == pytest.approx([1.0, 1.75, 2.166666667, 3.0], rel=1e-4)

def test_none_handling_in_feature_calculations():
    df = pl.DataFrame({
        "bid_price_1": [100.0, None],
        "bid_size_1": [1.0, None],
        "ask_price_1": [102.0, None],
        "ask_size_1": [1.5, None],
        "bid_size_2": [0.5, None],
        "bid_size_3": [0.2, None],
        "bid_size_4": [0.1, None],
        "bid_size_5": [0.05, None],
        "bid_size_6": [0.02, None],
        "bid_size_7": [0.01, None],
        "bid_size_8": [0.005, None],
        "bid_size_9": [0.002, None],
        "bid_size_10": [0.001, None],
        "ask_size_2": [0.75, None],
        "ask_size_3": [0.3, None],
        "ask_size_4": [0.15, None],
        "ask_size_5": [0.075, None],
        "ask_size_6": [0.03, None],
        "ask_size_7": [0.015, None],
        "ask_size_8": [0.0075, None],
        "ask_size_9": [0.003, None],
        "ask_size_10": [0.0015, None]
    })
    intermediate = df.with_columns([mid_price(), spread(), imbalance()])
    df = intermediate.with_columns([microprice(), depth_imbalance()])
    assert df["mid_price"].to_list() == [101.0, None]
    assert df["spread"].to_list() == [2.0, None]
    assert df["imbalance"].to_list() == [-0.2, None]
    assert df["microprice"].to_list() == [100.8, None]
    assert df["depth_imbalance"].to_list()[0] == pytest.approx(-0.2, rel=1e-4)
    assert df["depth_imbalance"].to_list()[1] is None