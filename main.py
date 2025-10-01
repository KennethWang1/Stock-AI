import traceback

from config import (
    STOCK_SYMBOL, MIN_HISTORICAL_DATA_POINTS, MODEL_SAVE_PATH, 
    MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY, STOCK_FEATURES, STOCK_EXCHANGE
)
from data import load_data
from utils.data_preprocessing import load_json_data, prepare_rl_data
from models.rl_model import build_rl_actor_critic_model
from trading.live_trader import (
    make_live_trading_decision, save_final_portfolio, load_initial_portfolio
)
from utils.memory_optimizer import setup_tensorflow_memory_optimization, clean_memory


def main():
    try:
        setup_tensorflow_memory_optimization()
        
        try:
            if(STOCK_EXCHANGE is None or STOCK_EXCHANGE == ''):
                load_data(STOCK_SYMBOL)
            else:
                load_data(STOCK_SYMBOL, STOCK_EXCHANGE)
            stock_history, stock_fundamentals, news_data = load_json_data()
        except Exception as e:
            print(f"Warning: Could not refresh data: {e}")
            stock_history, stock_fundamentals, news_data = load_json_data()

        if len(stock_history) < MIN_HISTORICAL_DATA_POINTS:
            raise Exception(f"Insufficient historical data. Need at least {MIN_HISTORICAL_DATA_POINTS} points, got {len(stock_history)}")
        
        df_history, fundamental_features, daily_news, stock_mask = prepare_rl_data(
            stock_history, stock_fundamentals, news_data
        )
        
        model = build_rl_actor_critic_model(
            stock_history_shape=(MAX_DAYS_HISTORY, len(STOCK_FEATURES)),
            fundamentals_shape=len(fundamental_features),
            news_shape=(MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY, 3)
        )
        
        initial_cash, initial_shares = load_initial_portfolio()
        
        final_portfolio_value, reward, experience_buffer = make_live_trading_decision(
            model, df_history, fundamental_features, daily_news, stock_mask,
            initial_cash=initial_cash, initial_shares=initial_shares
        )
        
        model.save(MODEL_SAVE_PATH)
        
        current_price = df_history.iloc[-1]['close']
        if experience_buffer.size() > 0:
            last_experience = experience_buffer.buffer[-1]
            final_cash = last_experience['next_state']['portfolio_cash']
            final_shares = last_experience['next_state']['portfolio_shares']
        else:
            final_cash = initial_cash
            final_shares = initial_shares
        
        initial_value = initial_cash + initial_shares * current_price
        total_return = ((final_portfolio_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
        
        save_final_portfolio(final_portfolio_value, final_shares, current_price, final_cash)
        
        print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        print(f"Today's Reward: {reward:.2f}%")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Experiences: {experience_buffer.size()}")
        
        clean_memory()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()
        # Clean memory even on error
        clean_memory()


if __name__ == "__main__":
    main()
