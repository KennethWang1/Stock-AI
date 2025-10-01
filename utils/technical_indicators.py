"""
Technical Indicators Calculation Module
Contains functions for calculating various technical analysis indicators
"""

import pandas as pd
import numpy as np
from config import RSI_PERIOD, SMA_SHORT_PERIOD, SMA_LONG_PERIOD, VOLATILITY_PERIOD


def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: Series of prices
        period: Period for RSI calculation (default 14)
    
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Fill NaN with neutral RSI


def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA)
    
    Args:
        prices: Series of prices
        window: Window size for SMA
    
    Returns:
        Series of SMA values
    """
    return prices.rolling(window=window, min_periods=1).mean()


def calculate_volatility(prices: pd.Series, window: int = VOLATILITY_PERIOD) -> pd.Series:
    """
    Calculate price volatility using rolling standard deviation
    
    Args:
        prices: Series of prices
        window: Window size for volatility calculation
    
    Returns:
        Series of volatility values
    """
    return prices.rolling(window=window, min_periods=1).std()


def calculate_price_change(df: pd.DataFrame) -> pd.Series:
    """
    Calculate price change (close - open)
    
    Args:
        df: DataFrame with 'close' and 'open' columns
    
    Returns:
        Series of price changes
    """
    return df['close'] - df['open']


def calculate_price_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate price range (high - low)
    
    Args:
        df: DataFrame with 'high' and 'low' columns
    
    Returns:
        Series of price ranges
    """
    return df['high'] - df['low']


def calculate_volume_normalized(df: pd.DataFrame) -> pd.Series:
    """
    Calculate normalized volume (volume / mean volume)
    
    Args:
        df: DataFrame with 'volume' column
    
    Returns:
        Series of normalized volume values
    """
    return df['volume'] / df['volume'].mean()


def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the DataFrame
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Basic price calculations
    df['price_change'] = calculate_price_change(df)
    df['price_range'] = calculate_price_range(df)
    df['volume_normalized'] = calculate_volume_normalized(df)
    
    # Moving averages
    df['sma_5'] = calculate_sma(df['close'], SMA_SHORT_PERIOD)
    df['sma_20'] = calculate_sma(df['close'], SMA_LONG_PERIOD)
    
    # Technical indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['volatility'] = calculate_volatility(df['close'])
    
    return df
