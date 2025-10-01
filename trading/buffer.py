import collections
import pickle
import numpy as np
from typing import List, Dict
from config import BUFFER_SIZE, MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY


class ExperienceReplayBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        
    def _validate_state_shape(self, state: Dict[str, np.ndarray]) -> bool:
        try:
            expected_stock_shape = (MAX_DAYS_HISTORY, 12)
            if state['stock_history'].shape != expected_stock_shape:
                return False
                
            expected_news_shape = (MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY, 3)
            if state['news_articles'].shape != expected_news_shape:
                return False
                
            return True
        except (KeyError, AttributeError):
            return False
        
    def add_experience(self, state: Dict[str, np.ndarray], action: float, 
                      reward: float, next_state: Dict[str, np.ndarray], done: bool):
        if not self._validate_state_shape(state) or not self._validate_state_shape(next_state):
            print("Skipping experience due to incompatible shapes")
            return
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.buffer.append(experience)
        
    def sample_batch(self, batch_size: int) -> List[Dict]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        return len(self.buffer)
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                experiences = pickle.load(f)
                
            compatible_experiences = []
            skipped_count = 0
            
            for exp in experiences:
                if (self._validate_state_shape(exp['state']) and 
                    self._validate_state_shape(exp['next_state'])):
                    compatible_experiences.append(exp)
                else:
                    skipped_count += 1
            
            self.buffer = collections.deque(compatible_experiences, maxlen=self.capacity)
            
            if skipped_count > 0:
                print(f"Loaded {len(compatible_experiences)} experiences, skipped {skipped_count} incompatible ones")
            else:
                print(f"Loaded {len(compatible_experiences)} compatible experiences")
                
        except FileNotFoundError:
            print(f"No existing buffer found at {filepath}")
        except Exception as e:
            print(f"Error loading buffer: {e}")
            self.buffer = collections.deque(maxlen=self.capacity)
    
    def clear(self):
        self.buffer.clear()
