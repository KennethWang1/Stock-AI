import os

STOCK_SYMBOL = 'OCTO'
STOCK_EXCHANGE = None

BUFFER_SIZE = 50
MAX_DAYS_HISTORY = 180
MAX_NEWS_PER_DAY = 5
ACTION_BOUND = 1.0

DEFAULT_INITIAL_CASH = 1500
DEFAULT_INITIAL_SHARES = 0

BATCH_SIZE = 16
EPOCHS_PER_DAY = 10
LEARNING_RATE = 0.001
GAMMA = 0.95

RSI_PERIOD = 14
SMA_SHORT_PERIOD = 5
SMA_LONG_PERIOD = 20
VOLATILITY_PERIOD = 10

DATA_DIR = './data'
STOCK_HISTORY_FILE = 'stock_history.json'
STOCK_FUNDAMENTALS_FILE = 'stock_data_filtered.json'
NEWS_DATA_FILE = 'news_feed.json'
PORTFOLIO_FILE = 'stock.json'
EXPERIENCE_BUFFER_FILE = './rl_experience_buffer.pkl'
MODEL_SAVE_PATH = 'rl_stock_trading_model.keras'

STOCK_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'price_change', 
    'price_range', 'volume_normalized', 'sma_5', 'sma_20', 'rsi', 'volatility'
]

NEWS_FEATURES = [
    'overall_sentiment_score',
    'ticker_relevance_score', 
    'ticker_sentiment_score'
]

MIN_HISTORICAL_DATA_POINTS = 50