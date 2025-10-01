import numpy as np
import pandas as pd
import json
import random
from typing import Tuple, Dict, List

from config import (
    MAX_DAYS_HISTORY, EXPERIENCE_BUFFER_FILE, PORTFOLIO_FILE, STOCK_SYMBOL,
    DEFAULT_INITIAL_CASH, DEFAULT_INITIAL_SHARES
)
from trading.environment import PortfolioEnvironment
from trading.buffer import ExperienceReplayBuffer
from models.rl_model import train_rl_model
from utils.data_preprocessing import create_state_representation


def simulate_rl_trading(model, df_history: pd.DataFrame, fundamental_features: List[float], 
                       daily_news: Dict, stock_mask: np.ndarray, 
                       initial_cash: float = DEFAULT_INITIAL_CASH, 
                       initial_shares: int = DEFAULT_INITIAL_SHARES) -> Tuple[float, float, ExperienceReplayBuffer]:
    
    env = PortfolioEnvironment(initial_cash, initial_shares)
    experience_buffer = ExperienceReplayBuffer()
    
    experience_buffer.load(EXPERIENCE_BUFFER_FILE)
    
    total_experiences = len(df_history) - 1
    print(f"\n=== RL TRADING SIMULATION ===")
    print(f"Starting Cash: ${initial_cash:.2f}")
    print(f"Starting Shares: {initial_shares}")
    print(f"Total Trading Days: {total_experiences}")
    print("-" * 60)
    
    for day_idx in range(total_experiences):
        open_price = df_history.iloc[day_idx]['open']
        close_price = df_history.iloc[day_idx]['close']
        next_day_open = df_history.iloc[day_idx + 1]['open']
        
        current_state = create_state_representation(
            df_history.iloc[:day_idx+1].tail(MAX_DAYS_HISTORY),
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
        action = predictions[0][0][0]
        
        epsilon = max(0.1, 1.0 - (day_idx / 100.0))
        if random.random() < epsilon:
            noise_action = random.uniform(-1.0, 1.0)
            exploration_weight = 0.3
            action = (1 - exploration_weight) * action + exploration_weight * noise_action
            print(f"Exploration: ε={epsilon:.2f}, noise={noise_action:+.3f}, final={action:+.3f}")
        
        action = np.clip(action, -1.0, 1.0)
        
        reward, trade_info = env.execute_action(action, open_price, close_price)
        
        next_state = create_state_representation(
            df_history.iloc[:day_idx+2].tail(MAX_DAYS_HISTORY),
            fundamental_features, daily_news, stock_mask,
            env.cash, env.shares, next_day_open
        )
        
        done = (day_idx == total_experiences - 1)
        experience_buffer.add_experience(
            current_state, action, reward, next_state, done
        )
        
        action_type = "BUY" if action > 0.1 else "SELL" if action < -0.1 else "HOLD"
        print(f"Day {day_idx+1:3d}: {action_type:4s} | Action: {action:+.3f} | "
              f"Reward: {reward:+.2f}% | Portfolio: ${trade_info['portfolio_value']:.2f} | "
              f"Cash: ${trade_info['cash']:.0f} | Shares: {trade_info['shares']}")
        
        if experience_buffer.size() >= 10:
            train_rl_model(model, experience_buffer)
    
    experience_buffer.save(EXPERIENCE_BUFFER_FILE)
    
    final_portfolio_value = env.get_portfolio_value(df_history.iloc[-1]['close'])

    total_return = ((final_portfolio_value - initial_cash - initial_shares * df_history.iloc[-1]['close']) / (initial_cash + initial_shares * df_history.iloc[-1]['close'])) * 100

    print("-" * 60)
    print(f"FINAL RL TRADING RESULTS:")
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Experiences Collected: {experience_buffer.size()}")
    
    return final_portfolio_value, total_return, experience_buffer


def save_final_portfolio(final_portfolio_value: float, final_shares: int, 
                        current_price: float, final_cash: float):
    try:
        final_portfolio = {
            "cash": final_cash,
            "stocks": []
        }
        
        if final_shares > 0:
            final_portfolio["stocks"].append({
                "symbol": STOCK_SYMBOL,
                "shares": final_shares,
                "price": current_price
            })
        
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(final_portfolio, f, indent=4)
        print(f"Updated portfolio saved to {PORTFOLIO_FILE}")
    except Exception as e:
        print(f"Warning: Could not save portfolio to {PORTFOLIO_FILE}: {e}")


def load_initial_portfolio() -> Tuple[float, int]:
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            portfolio_data = json.load(f)
        initial_cash = portfolio_data.get('cash', DEFAULT_INITIAL_CASH)
        initial_stocks = portfolio_data.get('stocks', [])
        initial_shares = sum(stock.get('shares', 0) for stock in initial_stocks 
                           if stock.get('symbol') == STOCK_SYMBOL)
        return initial_cash, initial_shares
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing portfolio found. Using default values.")
        return DEFAULT_INITIAL_CASH, DEFAULT_INITIAL_SHARES
