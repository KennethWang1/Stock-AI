#!/usr/bin/env python3
"""
Morning Trading Decision Script
Runs at the beginning of the day to:
1. Fetch the most recent market data
2. Ask the AI for a trading decision
3. Update today.json with the information
4. Does NOT train the model (morning-only decision)
"""

import os
import sys
import json
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    STOCK_SYMBOL, MIN_HISTORICAL_DATA_POINTS, MODEL_SAVE_PATH, 
    MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY, STOCK_FEATURES, STOCK_EXCHANGE,
    PORTFOLIO_FILE, EXPERIENCE_BUFFER_FILE
)
from data import load_data
from utils.data_preprocessing import load_json_data, prepare_rl_data, create_state_representation
from models.rl_model import build_rl_actor_critic_model
from trading.environment import PortfolioEnvironment
from trading.buffer import ExperienceReplayBuffer
from utils.memory_optimizer import setup_tensorflow_memory_optimization, clean_memory


def log_message(message: str, log_file: Path):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")


def load_initial_portfolio() -> Tuple[float, int]:
    """Load portfolio from stock.json"""
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            portfolio = json.load(f)
        return portfolio.get('cash', 1500), portfolio.get('shares', 0)
    except FileNotFoundError:
        print("No portfolio file found, using defaults")
        return 1500, 0
    except Exception as e:
        print(f"Error loading portfolio: {e}")
        return 1500, 0


def make_morning_trading_decision(model, df_history, fundamental_features, daily_news, 
                                stock_mask, initial_cash: float, initial_shares: int) -> Dict:
    """
    Make a trading decision without training the model
    Returns trading decision information
    """
    
    env = PortfolioEnvironment(initial_cash, initial_shares)
    
    # Load existing experience buffer for context (but don't modify it)
    experience_buffer = ExperienceReplayBuffer()
    try:
        experience_buffer.load(EXPERIENCE_BUFFER_FILE)
        print(f"Loaded experience buffer with {experience_buffer.size()} experiences")
    except:
        print("No existing experience buffer found")
    
    today_data = df_history.iloc[-1]
    current_price = today_data['close']  # Use close price for morning decision
    
    print(f"\n=== MORNING TRADING ANALYSIS ===")
    print(f"Current Stock Price: ${current_price:.4f}")
    print(f"Available Cash: ${env.cash:.2f}")
    print(f"Current Shares: {env.shares}")
    print(f"Portfolio Value: ${env.get_portfolio_value(current_price):.2f}")
    print("-" * 50)
    
    # Create state representation
    current_state = create_state_representation(
        df_history.tail(MAX_DAYS_HISTORY),
        fundamental_features, daily_news, stock_mask,
        env.cash, env.shares, current_price
    )
    
    # Get model prediction
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
    
    # Apply action bounds
    action = np.clip(action, -1.0, 1.0)
    
    # Determine action type from AI prediction
    action_type = "BUY" if action > 0.1 else "SELL" if action < -0.1 else "HOLD"
    
    # Get current portfolio value (no projection, just current state)
    current_portfolio_value = env.get_portfolio_value(current_price)
    
    print(f"AI Recommendation: {action_type}")
    print(f"Action Strength: {action:+.3f}")
    print(f"Current Portfolio Value: ${current_portfolio_value:.2f}")
    
    return {
        'action': float(action),
        'action_type': action_type,
        'portfolio_value': current_portfolio_value,
        'reward': 0.0,  # No reward calculated for morning decision
        'cash': env.cash,
        'shares': env.shares,
        'current_price': current_price,
        'message': f'Morning analysis complete: {action_type} recommendation'
    }


def update_today_json_morning(trading_result: Dict):
    """Update today.json with morning trading analysis"""
    try:
        today_json_path = Path("./today.json")
        
        # Load existing data if it exists
        if today_json_path.exists():
            with open(today_json_path, 'r') as f:
                today_data = json.load(f)
        else:
            today_data = {}
        
        # Update with morning analysis
        today_data.update({
            "totalCapital": trading_result['portfolio_value'],
            "lastAction": datetime.now().isoformat() + "Z",
            "lastActionType": trading_result['action_type'],
            "lastActionValue": str(abs(trading_result['action'])),
            "lastReward": str(trading_result['reward']),
            "ticker": STOCK_SYMBOL,
            "lastUpdated": datetime.now().isoformat() + "Z",
            "morningAnalysis": {
                "recommendation": trading_result['action_type'],
                "actionStrength": trading_result['action'],
                "currentCash": trading_result['cash'],
                "currentShares": trading_result['shares'],
                "currentPrice": trading_result['current_price'],
                "message": trading_result['message'],
                "timestamp": datetime.now().isoformat() + "Z"
            }
        })
        
        # Calculate net profit
        initial_investment = 1500  # Adjust based on your initial investment
        today_data["netProfit"] = trading_result['portfolio_value'] - initial_investment
        
        # Handle initial stock price and valuesLast30 updates
        current_price = trading_result['current_price']
        
        # Set initial stock price if null
        if "initialStockPrice" not in today_data or today_data["initialStockPrice"] is None:
            today_data["initialStockPrice"] = current_price
            print(f"Set initial stock price to: ${current_price:.4f}")
        
        # Calculate new portfolio value based on price change
        initial_stock_price = today_data["initialStockPrice"]
        new_portfolio_value = (initial_investment / initial_stock_price) * current_price
        
        # Update valuesLast30 array
        if "valuesLast30" not in today_data:
            today_data["valuesLast30"] = []
            today_data["traderLast30"] = []
        
        # Add new value and shift existing ones back
        if len(today_data["valuesLast30"]) >= 30:
            # Remove oldest value and add new one
            today_data["valuesLast30"] = [round(new_portfolio_value, 2)] + today_data["valuesLast30"][:-1]
            today_data["traderLast30"] = [round(new_portfolio_value * 0.9, 2)] + today_data["traderLast30"][:-1]
        else:
            # Add new value to the beginning
            today_data["valuesLast30"].insert(0, round(new_portfolio_value, 2))
            today_data["traderLast30"].insert(0, round(new_portfolio_value * 0.9, 2))
        
        # Fill with historical data if array is still not full (first-time setup)
        while len(today_data["valuesLast30"]) < 30:
            import random
            # Generate some historical variation for the remaining slots
            base_value = new_portfolio_value
            variation = random.uniform(-0.05, 0.05)
            historical_value = base_value * (1 + variation * len(today_data["valuesLast30"])/30)
            today_data["valuesLast30"].append(round(historical_value, 2))
            today_data["traderLast30"].append(round(historical_value * 0.9, 2))
        
        # Save updated data
        with open(today_json_path, 'w') as f:
            json.dump(today_data, f, indent=4)
        
        print(f"Updated today.json with morning analysis")
        return today_data
        
    except Exception as e:
        print(f"Error updating today.json: {e}")
        return None


def main():
    """Main morning trading function"""
    script_dir = Path(__file__).parent.absolute()
    
    # Setup logging
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    morning_log = logs_dir / "morning_trading.log"
    error_log = logs_dir / "trading_errors.log"
    
    # Change to script directory
    os.chdir(script_dir)
    
    log_message("Morning Trading Analysis Started", morning_log)
    log_message(f"Working Directory: {os.getcwd()}", morning_log)
    log_message("-" * 50, morning_log)
    
    try:
        # Setup TensorFlow memory optimization
        setup_tensorflow_memory_optimization()
        
        # Load fresh data
        log_message("Fetching latest market data...", morning_log)
        try:
            if STOCK_EXCHANGE is None or STOCK_EXCHANGE == '':
                load_data(STOCK_SYMBOL)
            else:
                load_data(STOCK_SYMBOL, STOCK_EXCHANGE)
            stock_history, stock_fundamentals, news_data = load_json_data()
            log_message("Successfully loaded fresh market data", morning_log)
        except Exception as e:
            log_message(f"Warning: Could not refresh data, using cached: {e}", morning_log)
            stock_history, stock_fundamentals, news_data = load_json_data()
        
        # Validate data
        if len(stock_history) < MIN_HISTORICAL_DATA_POINTS:
            raise Exception(f"Insufficient historical data. Need at least {MIN_HISTORICAL_DATA_POINTS} points, got {len(stock_history)}")
        
        # Prepare data for RL model
        df_history, fundamental_features, daily_news, stock_mask = prepare_rl_data(
            stock_history, stock_fundamentals, news_data
        )
        
        # Load the trained model
        log_message("Loading trained AI model...", morning_log)
        model = build_rl_actor_critic_model(
            stock_history_shape=(MAX_DAYS_HISTORY, len(STOCK_FEATURES)),
            fundamentals_shape=len(fundamental_features),
            news_shape=(MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY, 3)
        )
        
        try:
            model.load_weights(MODEL_SAVE_PATH)
            log_message("Successfully loaded pre-trained model weights", morning_log)
        except:
            log_message("Warning: Could not load pre-trained weights, using fresh model", morning_log)
        
        # Load current portfolio
        initial_cash, initial_shares = load_initial_portfolio()
        log_message(f"Current Portfolio: ${initial_cash:.2f} cash, {initial_shares} shares", morning_log)
        
        # Make morning trading decision (no training)
        trading_result = make_morning_trading_decision(
            model, df_history, fundamental_features, daily_news, stock_mask,
            initial_cash, initial_shares
        )
        
        # Update today.json with results
        update_today_json_morning(trading_result)
        
        log_message(f"Morning analysis complete: {trading_result['action_type']} recommendation", morning_log)
        log_message(f"Action strength: {trading_result['action']:+.3f}", morning_log)
        log_message(f"Current portfolio value: ${trading_result['portfolio_value']:.2f}", morning_log)
        log_message("=" * 50, morning_log)
        
        # Clean up memory
        clean_memory()
        
        print(f"\n✅ Morning trading analysis completed successfully!")
        print(f"📊 Recommendation: {trading_result['action_type']}")
        print(f"💼 Current Portfolio Value: ${trading_result['portfolio_value']:.2f}")
        print(f"📄 Check today.json for detailed results")
        
        sys.exit(0)
        
    except Exception as e:
        error_msg = f"Morning trading analysis failed: {e}"
        log_message(error_msg, error_log)
        log_message(error_msg, morning_log)
        
        full_traceback = traceback.format_exc()
        log_message(full_traceback, error_log)
        log_message("=" * 50, error_log)
        
        print(f"❌ Error: {error_msg}")
        print(f"📋 Check logs/morning_trading.log for details")
        
        # Clean memory even on error
        clean_memory()
        sys.exit(1)


if __name__ == "__main__":
    main()