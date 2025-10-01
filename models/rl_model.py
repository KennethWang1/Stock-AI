import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict

from config import (
    MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY, LEARNING_RATE, BATCH_SIZE, 
    EPOCHS_PER_DAY, GAMMA, STOCK_FEATURES
)
from utils.memory_optimizer import setup_tensorflow_memory_optimization, clean_memory


def train_rl_model(model, experience_buffer, epochs=EPOCHS_PER_DAY):
    if experience_buffer.size() < 10:
        print("Not enough experiences for training")
        return
    
    buffer_size = experience_buffer.size()
    if buffer_size < 30:
        boosted_epochs = epochs * 3
        
        original_lr = model.optimizer.learning_rate.numpy()
        boost_lr = original_lr * 2.0
        model.optimizer.learning_rate.assign(boost_lr)
        epochs = boosted_epochs
    
    batch_size = min(BATCH_SIZE, experience_buffer.size())
    experiences = experience_buffer.sample_batch(batch_size)
    
    if not experiences:
        print("No experiences to train on")
        return
        
    first_state = experiences[0]['state']
    expected_stock_shape = (MAX_DAYS_HISTORY, len(STOCK_FEATURES))
    expected_news_shape = (MAX_DAYS_HISTORY, MAX_NEWS_PER_DAY, 3)
    
    if (first_state['stock_history'].shape != expected_stock_shape or 
        first_state['news_articles'].shape != expected_news_shape):
        print("Experience shapes don't match current config - clearing buffer")
        experience_buffer.clear()
        return
    
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    for exp in experiences:
        states.append(exp['state'])
        actions.append(exp['action'])
        rewards.append(exp['reward'])
        next_states.append(exp['next_state'])
        dones.append(exp['done'])
    
    batch_stock_history = np.array([s['stock_history'] for s in states], dtype=np.float32)
    batch_fundamentals = np.array([s['fundamentals'] for s in states], dtype=np.float32)
    batch_news_articles = np.array([s['news_articles'] for s in states], dtype=np.float32)
    batch_portfolio = np.array([[s['portfolio_cash'], s['portfolio_shares'], s['current_price']] for s in states], dtype=np.float32)
    
    batch_actions = np.array(actions, dtype=np.float32)
    batch_rewards = np.array(rewards, dtype=np.float32)
    
    next_batch_stock_history = np.array([s['stock_history'] for s in next_states], dtype=np.float32)
    next_batch_fundamentals = np.array([s['fundamentals'] for s in next_states], dtype=np.float32)
    next_batch_news_articles = np.array([s['news_articles'] for s in next_states], dtype=np.float32)
    next_batch_portfolio = np.array([[s['portfolio_cash'], s['portfolio_shares'], s['current_price']] for s in next_states], dtype=np.float32)
    
    next_predictions = model.predict([
        next_batch_stock_history, next_batch_fundamentals,
        next_batch_news_articles, next_batch_portfolio
    ], verbose=0)
    
    next_values = next_predictions[1].flatten()
    
    target_values = batch_rewards + GAMMA * next_values * (1 - np.array(dones, dtype=np.float32))
    
    if epochs > 10:
        chunk_size = max(1, epochs // 5)
        for chunk in range(0, epochs, chunk_size):
            current_epochs = min(chunk_size, epochs - chunk)
            
            history = model.fit(
                [batch_stock_history, batch_fundamentals,
                 batch_news_articles, batch_portfolio],
                {
                    'action': batch_actions.reshape(-1, 1).astype(np.float32),
                    'value': target_values.reshape(-1, 1).astype(np.float32)
                },
                epochs=current_epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            clean_memory()
    else:
        history = model.fit(
            [batch_stock_history, batch_fundamentals,
             batch_news_articles, batch_portfolio],
            {
                'action': batch_actions.reshape(-1, 1).astype(np.float32),
                'value': target_values.reshape(-1, 1).astype(np.float32)
            },
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
    
    if buffer_size < 30:
        model.optimizer.learning_rate.assign(original_lr)
    
    clean_memory()
    
    return history


class AttentionMaskLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionMaskLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        mask = inputs
        return tf.expand_dims(mask, axis=1)


def build_rl_actor_critic_model(stock_history_shape: Tuple[int, int], 
                               fundamentals_shape: int, 
                               news_shape: Tuple[int, int, int]) -> tf.keras.Model:
    stock_input = tf.keras.layers.Input(shape=stock_history_shape, name='stock_history')
    
    stock_lstm = tf.keras.layers.LSTM(128, return_sequences=True)(stock_input)
    stock_lstm = tf.keras.layers.Dropout(0.3)(stock_lstm)
    
    stock_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=8, key_dim=64, name='stock_attention'
    )(stock_lstm, stock_lstm)
    
    stock_lstm_final = tf.keras.layers.LSTM(64, return_sequences=False)(stock_attention)
    stock_output = tf.keras.layers.Dense(64, activation='relu')(stock_lstm_final)
    
    fundamentals_input = tf.keras.layers.Input(shape=(fundamentals_shape,), name='fundamentals')
    fundamentals_branch = tf.keras.layers.Dense(64, activation='relu')(fundamentals_input)
    fundamentals_branch = tf.keras.layers.Dropout(0.3)(fundamentals_branch)
    fundamentals_output = tf.keras.layers.Dense(32, activation='relu')(fundamentals_branch)
    
    news_input = tf.keras.layers.Input(shape=news_shape, name='news_articles')
    
    news_reshaped = tf.keras.layers.Reshape((MAX_DAYS_HISTORY * MAX_NEWS_PER_DAY, 3))(news_input)
    
    news_embed = tf.keras.layers.Dense(32, activation='relu')(news_reshaped)
    
    news_pooled = tf.keras.layers.GlobalAveragePooling1D()(news_embed)
    news_output = tf.keras.layers.Dense(32, activation='relu')(news_pooled)
    
    portfolio_input = tf.keras.layers.Input(shape=(3,), name='portfolio_state')
    portfolio_branch = tf.keras.layers.Dense(16, activation='relu')(portfolio_input)
    
    combined = tf.keras.layers.concatenate([
        stock_output, fundamentals_output, news_output, portfolio_branch
    ])
    
    shared_dense = tf.keras.layers.Dense(128, activation='relu')(combined)
    shared_dense = tf.keras.layers.Dropout(0.4)(shared_dense)
    shared_dense = tf.keras.layers.Dense(64, activation='relu')(shared_dense)
    
    actor_branch = tf.keras.layers.Dense(32, activation='relu')(shared_dense)
    actor_output = tf.keras.layers.Dense(1, activation='tanh', name='action')(actor_branch)
    
    critic_branch = tf.keras.layers.Dense(32, activation='relu')(shared_dense)
    critic_output = tf.keras.layers.Dense(1, activation='linear', name='value')(critic_branch)
    
    model = tf.keras.Model(
        inputs=[stock_input, fundamentals_input, news_input, portfolio_input],
        outputs=[actor_output, critic_output]
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            'action': 'mse',
            'value': 'mse'
        },
        metrics={
            'action': ['mae'],
            'value': ['mae']
        }
    )
    
    return model
