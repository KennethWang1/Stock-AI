import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple

from config import (
    DATA_DIR, STOCK_HISTORY_FILE, STOCK_FUNDAMENTALS_FILE, NEWS_DATA_FILE,
    MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY, STOCK_FEATURES, NEWS_FEATURES
)
from utils.technical_indicators import add_all_technical_indicators
from utils.memory_optimizer import optimize_data_types, optimize_arrays, optimize_state_dict


def load_json_data() -> Tuple[List[Dict], Dict, List[Dict]]:
    with open(os.path.join(DATA_DIR, STOCK_HISTORY_FILE), 'r') as f:
        stock_history = json.load(f)
    
    with open(os.path.join(DATA_DIR, STOCK_FUNDAMENTALS_FILE), 'r') as f:
        stock_fundamentals = json.load(f)
    
    with open(os.path.join(DATA_DIR, NEWS_DATA_FILE), 'r') as f:
        news_data = json.load(f)
    
    return stock_history, stock_fundamentals, news_data


def prepare_rl_data(stock_history: List[Dict], stock_fundamentals: Dict, 
                   news_data: List[Dict]) -> Tuple[pd.DataFrame, List[float], Dict, np.ndarray]:
    df_history = pd.DataFrame(stock_history)
    
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        df_history[col] = pd.to_numeric(df_history[col], errors='coerce').astype('float32')
    df_history['volume'] = pd.to_numeric(df_history['volume'], errors='coerce').astype('float32')
    
    df_history = df_history.sort_values('days_ago', ascending=True).reset_index(drop=True)
    
    if len(df_history) > MAX_DAYS_HISTORY:
        df_history = df_history.iloc[-MAX_DAYS_HISTORY:].reset_index(drop=True)
    
    df_history = add_all_technical_indicators(df_history)
    df_history = optimize_data_types(df_history)
    
    current_days = len(df_history)
    if current_days < MAX_DAYS_HISTORY:
        padding_days = MAX_DAYS_HISTORY - current_days
        padding_data = []
        for i in range(padding_days):
            padding_row = {col: np.float32(0.0) for col in STOCK_FEATURES}
            padding_row['days_ago'] = current_days + i + 1
            padding_data.append(padding_row)
        
        padding_df = pd.DataFrame(padding_data)
        padding_df = optimize_data_types(padding_df)
        df_history = pd.concat([padding_df, df_history], ignore_index=True)
    
    stock_mask = np.zeros(MAX_DAYS_HISTORY)
    stock_mask[-current_days:] = 1.0
    
    fundamental_features = []
    for key, value in stock_fundamentals.items():
        if key in ['Symbol', 'Country', 'Sector']:
            continue
        try:
            fundamental_features.append(float(value))
        except (ValueError, TypeError):
            fundamental_features.append(0.0)
    
    df_news = pd.DataFrame(news_data)
    daily_news = prepare_daily_news(df_news)
    
    return df_history, fundamental_features, daily_news, stock_mask


def prepare_daily_news(df_news: pd.DataFrame) -> Dict[str, np.ndarray]:
    daily_news = {}
    
    for _, article in df_news.iterrows():
        day = article.get('days_ago', 0)
        if day not in daily_news:
            daily_news[day] = []
        
        news_features = [
            float(article.get('overall_sentiment_score', 0.0)),
            float(article.get('ticker_relevance_score', 0.0)),
            float(article.get('ticker_sentiment_score', 0.0))
        ]
        daily_news[day].append(news_features)
    
    news_array = np.zeros((MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY, 3))
    news_mask = np.zeros((MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY))
    
    for day in range(MAX_DAYS_HISTORY):
        if day in daily_news:
            articles = daily_news[day][:MAX_NEWS_PER_DAY]
            for i, article_features in enumerate(articles):
                news_array[day, i] = article_features
                news_mask[day, i] = 1.0
    
    return {'articles': news_array, 'mask': news_mask}


def create_state_representation(df_history: pd.DataFrame, fundamental_features: List[float], 
                              daily_news: Dict, stock_mask: np.ndarray, 
                              portfolio_cash: float, portfolio_shares: int, 
                              current_price: float) -> Dict[str, np.ndarray]:
    stock_data = df_history[STOCK_FEATURES].values.astype(np.float32)
    
    current_data_length = len(stock_data)
    current_mask = stock_mask[-current_data_length:] if len(stock_mask) >= current_data_length else np.ones(current_data_length, dtype=np.float32)
    
    scaler_stock = MinMaxScaler(feature_range=(0, 1))
    non_zero_mask = current_mask == 1
    if np.any(non_zero_mask):
        stock_data_nonzero = stock_data[non_zero_mask]
        if len(stock_data_nonzero) > 0:
            scaler_stock.fit(stock_data_nonzero)
            stock_data = scaler_stock.transform(stock_data).astype(np.float32)
    
    if len(stock_data) < MAX_DAYS_HISTORY:
        padding = np.zeros((MAX_DAYS_HISTORY - len(stock_data), len(STOCK_FEATURES)), dtype=np.float32)
        stock_data = np.vstack([padding, stock_data])
        mask_padding = np.zeros(MAX_DAYS_HISTORY - len(current_mask), dtype=np.float32)
        current_mask = np.concatenate([mask_padding, current_mask])
    elif len(stock_data) > MAX_DAYS_HISTORY:
        stock_data = stock_data[-MAX_DAYS_HISTORY:]
        current_mask = current_mask[-MAX_DAYS_HISTORY:]
    
    fundamental_array = np.array(fundamental_features, dtype=np.float32)
    if len(fundamental_array) > 0:
        scaler_fundamental = MinMaxScaler(feature_range=(0, 1))
        fundamental_array = scaler_fundamental.fit_transform(fundamental_array.reshape(1, -1)).flatten().astype(np.float32)
    
    portfolio_total_value = portfolio_cash + portfolio_shares * current_price
    normalized_cash = np.float32(portfolio_cash / max(portfolio_total_value, 1.0))
    normalized_shares = np.float32((portfolio_shares * current_price) / max(portfolio_total_value, 1.0))
    normalized_price = np.float32(current_price / max(current_price, 1.0))
    
    state = {
        'stock_history': stock_data,
        'stock_mask': current_mask,
        'fundamentals': fundamental_array,
        'news_articles': daily_news['articles'].astype(np.float32),
        'news_mask': daily_news['mask'].astype(np.float32),
        'portfolio_cash': normalized_cash,
        'portfolio_shares': normalized_shares,
        'current_price': normalized_price
    }
    
    return optimize_state_dict(state)
