import psutil
import os
import numpy as np
from datetime import datetime
from typing import Dict

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.checkpoints = []
    
    def get_memory_usage(self) -> Dict[str, float]:
        memory_info = self.process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,
            'vms': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'available': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def checkpoint(self, label: str):
        current_memory = self.get_memory_usage()
        
        if current_memory['rss'] > self.peak_memory['rss']:
            self.peak_memory = current_memory
        
        checkpoint = {
            'label': label,
            'timestamp': datetime.now(),
            'memory': current_memory,
            'growth_mb': current_memory['rss'] - self.initial_memory['rss']
        }
        
        self.checkpoints.append(checkpoint)
        
        print(f"{label}: {current_memory['rss']:.1f}MB RAM "
              f"({checkpoint['growth_mb']:+.1f}MB from start)")
        
        return checkpoint
    
    def report(self):
        print("\nMEMORY USAGE REPORT")
        print("=" * 50)
        
        current = self.get_memory_usage()
        total_growth = current['rss'] - self.initial_memory['rss']
        
        print(f"Initial Memory: {self.initial_memory['rss']:.1f}MB")
        print(f"Current Memory: {current['rss']:.1f}MB")
        print(f"Peak Memory: {self.peak_memory['rss']:.1f}MB")
        print(f"Total Growth: {total_growth:+.1f}MB")
        print(f"System Available: {current['available']:.1f}MB")
        
        print(f"\nMemory Checkpoints:")
        for cp in self.checkpoints:
            print(f"   {cp['label']}: {cp['memory']['rss']:.1f}MB "
                  f"({cp['growth_mb']:+.1f}MB)")
        
        if current['rss'] > 3000:
            print(f"\nHIGH MEMORY USAGE WARNING!")
            print(f"   Current usage: {current['rss']:.1f}MB")
            print(f"   Consider reducing BUFFER_SIZE or MAX_DAYS_HISTORY")
        
        if current['available'] < 1000:
            print(f"\nLOW SYSTEM MEMORY WARNING!")
            print(f"   Available: {current['available']:.1f}MB")
            print(f"   System may start swapping to disk")

def estimate_memory_requirements():
    from config import BUFFER_SIZE, MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY, BATCH_SIZE
    
    print("MEMORY REQUIREMENT ESTIMATION")
    print("=" * 40)
    
    stock_features = 12
    fundamentals_features = 24
    news_features = MAX_DAYS_HISTORY * MAX_NEWS_PER_DAY * 3
    portfolio_features = 3
    
    state_size_floats = (MAX_DAYS_HISTORY * stock_features + 
                        fundamentals_features + 
                        news_features + 
                        portfolio_features)
    
    single_state_mb = (state_size_floats * 4) / (1024 * 1024)
    buffer_memory_mb = single_state_mb * BUFFER_SIZE * 2
    batch_memory_mb = single_state_mb * BATCH_SIZE * 2
    
    model_memory_mb = 10
    
    total_estimated_mb = buffer_memory_mb + batch_memory_mb + model_memory_mb + 200
    
    print(f"Single State Size: {single_state_mb:.2f}MB")
    print(f"Experience Buffer: {buffer_memory_mb:.2f}MB")
    print(f"Training Batch: {batch_memory_mb:.2f}MB")
    print(f"Model + Overhead: {model_memory_mb + 200:.2f}MB")
    print(f"Total Estimated: {total_estimated_mb:.2f}MB")
    
    if total_estimated_mb > 3500:
        print(f"\nWARNING: Estimated memory usage is high!")
        print(f"   Consider reducing:")
        print(f"   - BUFFER_SIZE (current: {BUFFER_SIZE})")
        print(f"   - MAX_DAYS_HISTORY (current: {MAX_DAYS_HISTORY})")
        print(f"   - MAX_NEWS_PER_DAY (current: {MAX_NEWS_PER_DAY})")
    else:
        print(f"\nMemory usage looks manageable for 4GB+ systems")
    
    return total_estimated_mb

if __name__ == "__main__":
    estimate_memory_requirements()