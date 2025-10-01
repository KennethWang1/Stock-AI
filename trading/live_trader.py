import numpy as np
import pandas as pd
import json
import random
from typing import Tuple, Dict, List
from trading_tracker import has_traded_today, record_trading_session

from config import (
    MAX_DAYS_HISTORY, EXPERIENCE_BUFFER_FILE, PORTFOLIO_FILE, STOCK_SYMBOL,
    DEFAULT_INITIAL_CASH, DEFAULT_INITIAL_SHARES
)
from trading.environment import PortfolioEnvironment
from trading.buffer import ExperienceReplayBuffer
from models.rl_model import train_rl_model
from utils.data_preprocessing import create_state_representation


def make_live_trading_decision(model, df_history: pd.DataFrame, fundamental_features: List[float], 
                              daily_news: Dict, stock_mask: np.ndarray, 
                              initial_cash: float = DEFAULT_INITIAL_CASH, 
                              initial_shares: int = DEFAULT_INITIAL_SHARES) -> Tuple[float, float, ExperienceReplayBuffer]:
    
    env = PortfolioEnvironment(initial_cash, initial_shares)
    experience_buffer = ExperienceReplayBuffer()
    
    if has_traded_today():
        print(f"Already traded today! Skipping to prevent duplicate experiences.")
        print(f"Current buffer size: {experience_buffer.size() if experience_buffer else 0}")
        
        current_price = df_history.iloc[-1]['close']
        portfolio_value = initial_cash + initial_shares * current_price
        
        portfolio_data = {
            'total_value': portfolio_value,
            'cash': initial_cash,
            'shares': initial_shares, 
            'current_price': current_price,
            'timestamp': pd.Timestamp.now().isoformat(),
            'status': 'no_trade_today_already_completed'
        }
        
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        
        print(f"Portfolio timestamp updated (no trade needed)")
        
        return portfolio_value, 0.0, experience_buffer
    
    try:
        experience_buffer.load(EXPERIENCE_BUFFER_FILE)
        print(f"Loaded existing experience buffer with {experience_buffer.size()} experiences")
    except:
        print("Starting with empty experience buffer")
    
    today_data = df_history.iloc[-1]
    open_price = today_data['open']
    close_price = today_data['close']
    
    print(f"\n=== LIVE TRADING DECISION ===")
    print(f"Market Open Price: ${open_price:.4f}")
    print(f"Starting Cash: ${env.cash:.2f}")
    print(f"Starting Shares: {env.shares}")
    print(f"Experience Buffer Size: {experience_buffer.size()}")
    print("-" * 50)
    
    current_state = create_state_representation(
        df_history.tail(MAX_DAYS_HISTORY),
        fundamental_features, daily_news, stock_mask,
        env.cash, env.shares, open_price
    )
    
    model_inputs = [
        np.expand_dims(current_state['stock_history'], 0),
        np.expand_dims(current_state['fundamentals'], 0),
        np.expand_dims(current_state['news_articles'], 0),
        np.expand_dims([current_state['portfolio_cash'], 
                       current_state['portfolio_shares'], 
                       current_state['current_price']], 0)
    ]
    
    predictions = model.predict(model_inputs, verbose=0)
    base_action = predictions[0][0][0]
    
    epsilon = max(0.2, 1.0 - (experience_buffer.size() / 100.0))
    
    if random.random() < epsilon:
        noise_action = random.uniform(-1.0, 1.0)
        exploration_weight = 0.4
        action = (1 - exploration_weight) * base_action + exploration_weight * noise_action
        print(f"Exploration: ε={epsilon:.2f}, base={base_action:+.3f}, noise={noise_action:+.3f}, final={action:+.3f}")
    else:
        action = base_action
        print(f"Model Decision: {action:+.3f}")
    
    action = np.clip(action, -1.0, 1.0)
    
    reward, trade_info = env.execute_action(action, open_price, close_price)
    
    next_state = current_state.copy()
    next_state['portfolio_cash'] = trade_info['cash']
    next_state['portfolio_shares'] = trade_info['shares']
    next_state['current_price'] = close_price
    
    experience_buffer.add_experience(
        current_state, action, reward, next_state, done=False
    )
    
    if experience_buffer.size() >= 10:
        print(f"\nTraining model with {experience_buffer.size()} experiences...")
        train_rl_model(model, experience_buffer)
    else:
        print(f"Need {10 - experience_buffer.size()} more experiences before training")
    
    experience_buffer.save(EXPERIENCE_BUFFER_FILE)
    
    action_type = "BUY" if action > 0.1 else "SELL" if action < -0.1 else "HOLD"
    print(f"\nTRADING RESULT:")
    print(f"Action Type: {action_type}")
    print(f"Action Value: {action:+.3f}")
    print(f"Market Close Price: ${close_price:.4f}")
    print(f"Reward: {reward:+.2f}%")
    print(f"Portfolio Value: ${trade_info['portfolio_value']:.2f}")
    print(f"Final Cash: ${trade_info['cash']:.0f}")
    print(f"Final Shares: {trade_info['shares']}")
    
    if 'action_bonus' in trade_info and trade_info['action_bonus'] > 0:
        print(f"Risk-taking bonus: +{trade_info['action_bonus']:.2f}")
    
    final_portfolio_value = trade_info['portfolio_value']
    
    record_trading_session(action, reward, final_portfolio_value)
    print(f"Trading session recorded for today")
    
    return final_portfolio_value, reward, experience_buffer


def save_final_portfolio(portfolio_value: float, shares: int, current_price: float, cash: float):
    portfolio_data = {
        'total_value': portfolio_value,
        'cash': cash,
        'shares': shares,
        'current_price': current_price,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio_data, f, indent=2)
    
    print(f"Portfolio saved to {PORTFOLIO_FILE}")


def load_initial_portfolio() -> Tuple[float, int]:
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            portfolio_data = json.load(f)
        
        cash = portfolio_data.get('cash', DEFAULT_INITIAL_CASH)
        shares = portfolio_data.get('shares', DEFAULT_INITIAL_SHARES)
        
        if cash > 1_000_000 or shares > 1000:
            print(f"Detected inflated values (cash: ${cash:,.0f}, shares: {shares})")
            print(f"Resetting to defaults: cash=${DEFAULT_INITIAL_CASH}, shares={DEFAULT_INITIAL_SHARES}")
            return DEFAULT_INITIAL_CASH, DEFAULT_INITIAL_SHARES
        
        return cash, shares
    except FileNotFoundError:
        return DEFAULT_INITIAL_CASH, DEFAULT_INITIAL_SHARES
