from typing import Tuple, Dict
from config import DEFAULT_INITIAL_CASH, DEFAULT_INITIAL_SHARES
import random


class PortfolioEnvironment:
    def __init__(self, initial_cash: float = random.uniform(1500,10000), initial_shares: int = random.randint(0,10)):
        self.initial_cash = initial_cash
        self.initial_shares = initial_shares
        self.reset()
        
    def reset(self):
        self.cash = self.initial_cash
        self.shares = self.initial_shares
        self.current_price = 0.0
        self.portfolio_value_start_day = 0.0
        
    def get_portfolio_value(self, price: float) -> float:
        return self.cash + self.shares * price
        
    def execute_action(self, action: float, open_price: float, close_price: float) -> Tuple[float, Dict]:
        self.current_price = open_price
        self.portfolio_value_start_day = self.get_portfolio_value(open_price)
        self.current_portfolio_value_end_day = self.get_portfolio_value(close_price)
        
        max_affordable_shares = int(self.cash / open_price) if open_price > 0 else 0
        max_sellable_shares = self.shares
        
        if action > 0.1:
            shares_to_buy = int(action * max_affordable_shares)
            shares_to_buy = min(shares_to_buy, max_affordable_shares)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * open_price
                self.cash -= cost
                self.shares += shares_to_buy
                
        elif action < -0.1:
            shares_to_sell = int(abs(action) * max_sellable_shares)
            shares_to_sell = min(shares_to_sell, max_sellable_shares)
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * open_price
                self.cash += revenue
                self.shares -= shares_to_sell
        
        portfolio_value_end_day = self.get_portfolio_value(close_price)
        reward = (portfolio_value_end_day - self.current_portfolio_value_end_day) / self.current_portfolio_value_end_day * 100

        action_bonus = 0
        if abs(action) > 0.1:
            action_bonus = abs(action) * 0.5
            reward += action_bonus
            
        if action<= 0.1 and action>= -0.1:
            hold_penalty = -0.1
            reward += hold_penalty
        
        trade_info = {
            'action': action,
            'cash': self.cash,
            'shares': self.shares,
            'portfolio_value': portfolio_value_end_day,
            'reward': reward,
            'action_bonus': action_bonus if abs(action) > 0.1 else 0,
            'base_reward': reward - (action_bonus if abs(action) > 0.1 else 0) - (hold_penalty if action<= 0.1 and action>= -0.1 else 0)
        }
        
        return reward, trade_info
