import numpy as np
import pandas as pd
import gc
import tensorflow as tf
from typing import Dict, Any

def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df_optimized = df.copy()
    
    for col in df_optimized.select_dtypes(include=['float64']):
        df_optimized[col] = df_optimized[col].astype('float32')
    
    for col in df_optimized.select_dtypes(include=['int64']):
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()
        if col_min >= -2147483648 and col_max <= 2147483647:
            df_optimized[col] = df_optimized[col].astype('int32')
    
    return df_optimized

def optimize_arrays(*arrays) -> tuple:
    return tuple(np.array(arr, dtype=np.float32) for arr in arrays)

def setup_tensorflow_memory_optimization():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory setup: {e}")
    
    tf.config.experimental_run_functions_eagerly(False)

def clean_memory():
    tf.keras.backend.clear_session()
    gc.collect()

def optimize_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    optimized_state = {}
    
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            if value.dtype in [np.float64, 'float64']:
                optimized_state[key] = value.astype(np.float32)
            else:
                optimized_state[key] = value
        else:
            optimized_state[key] = np.float32(value) if isinstance(value, (float, int)) else value
    
    return optimized_state