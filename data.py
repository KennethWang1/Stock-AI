import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime, timedelta
from config import STOCK_SYMBOL, STOCK_EXCHANGE
from dotenv import load_dotenv

load_dotenv()

def load_data(ticker, exchange=None):
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    r = requests.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}")
    data = r.json()
    
    if 'Information' in data and 'rate limit' in data['Information']:
        print(f"API Rate limit exceeded: {data['Information']}")
        return

    wanted_keys = ['Symbol', 'Country', 'Sector', 'DividendYield', 'EPS', 'RevenuePerShareTTM', 'ProfitMargin','AnalystTargetPrice', 'AnalystRatingStrongBuy'
                , 'AnalystRatingBuy', 'AnalystRatingHold', 'AnalystRatingSell', 'AnalystRatingStrongSell', 'PercentInstitutions', 'MarketCapitalization', 'PERatio'
                , 'PEGRatio', 'BookValue', 'OperatingMarginTTM', 'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'DilutedEPSTTM', 'QuarterlyEarningsGrowthYOY', 'QuarterlyRevenueGrowthYOY'
                , 'PercentInsiders', 'DividendDate', 'ExDividendDate']
    filtered_data_method = {k: v for k, v in data.items() if k in wanted_keys}
    
    dividend_date = data.get('DividendDate')
    if dividend_date and dividend_date != 'None':
        try:
            filtered_data_method['DividendDate'] = (datetime.strptime(dividend_date, '%Y-%m-%d') - datetime.now()).days
        except ValueError:
            filtered_data_method['DividendDate'] = 0
    else:
        filtered_data_method['DividendDate'] = 0
    
    ex_dividend_date = data.get('ExDividendDate')
    if ex_dividend_date and ex_dividend_date != 'None':
        try:
            filtered_data_method['ExDividendDate'] = (datetime.now() - datetime.strptime(ex_dividend_date, '%Y-%m-%d')).days
        except ValueError:
            filtered_data_method['ExDividendDate'] = 0
    else:
        filtered_data_method['ExDividendDate'] = 0

    r = requests.get(f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={(ticker + '.' + exchange) if exchange else ticker}&apikey={api_key}")
    data = r.json()
    
    if 'Information' in data and 'rate limit' in data['Information']:
        print(f"API Rate limit exceeded: {data['Information']}")
        return

    minimal_feed_data = []
    i = 0
    for article in data.get('feed', []):
        published_time = datetime.strptime(article.get('time_published'), '%Y%m%dT%H%M%S')
        if((datetime.now() - published_time).days == 0 and published_time.hour > 9 and datetime.now().hour < 17):
            continue
        minimal_article = {
            'days_ago': (datetime.now() - published_time).days,
            'overall_sentiment_score': article.get('overall_sentiment_score'),
            'ticker_relevance_score' : article.get('ticker_sentiment')[0].get('relevance_score'),
            'ticker_sentiment_score' : article.get('ticker_sentiment')[0].get('ticker_sentiment_score')
        }
        
        minimal_feed_data.append(minimal_article)
        i += 1
        if i >= 10:
            break

    r = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={(ticker + '.' + exchange) if exchange else ticker}&outputsize=full&apikey={api_key}")
    data = r.json()
    
    if 'Information' in data and 'rate limit' in data['Information']:
        print(f"API Rate limit exceeded: {data['Information']}")
        return

    time_series = data.get('Time Series (Daily)', {})
    daily_data = []
    for date, metrics in time_series.items():
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        daily_data.append({
            'days_ago': (datetime.now() - date_obj).days,
            'open': metrics.get('1. open'),
            'high': metrics.get('2. high'),
            'low': metrics.get('3. low'),
            'close': metrics.get('4. close'),
            'volume': metrics.get('5. volume')
        })
        thirty_days_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if date < thirty_days_ago:
            break

    while len(minimal_feed_data) < 10:
        minimal_feed_data.append({
            'time': None,
            'overall_sentiment_score': 0.5,
            'ticker_relevance_score': 0,
            'ticker_sentiment_score': 0.5
        })
    with open('./data/news_feed.json', 'w') as f:
        json.dump(minimal_feed_data, f, indent=4)

    with open('./data/stock_data_filtered.json', 'w') as f:
        json.dump(filtered_data_method, f, indent=4)

    with open('./data/stock_history.json', 'w') as f:
        json.dump(daily_data, f, indent=4)

if __name__ == "__main__":
    load_data(STOCK_SYMBOL, STOCK_EXCHANGE)