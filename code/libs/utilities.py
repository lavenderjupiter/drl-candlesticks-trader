from statistics import stdev
import numpy as np
import pandas as pd
import os
from collections import deque


HYPERPARAMS = {
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'batch_size': 64,
    'sync_target_steps': 1000, #DQN
    'replay_buffer_size': 10000, #DQN
    'epsilon_decay_last_frame': 10**5, #DQN
    'epsilon_start': 1.0, #DQN
    'epsilon_final': 0.1, #DQN
    'actor_learning_rate': 1e-4, #PPO
    'critic_learning_rate': 1e-3, #PPO
    'clip_epsilon': 0.2, #PPO
    'gae_lambda': 0.95, #PPO
    'entropy_beta': 1e-3, #PPO
    'ppo_epochs': 10, #PPO
    'critic_discount': 0.5, #PPO
    # 'update_iters': 1000, #PPO
    'tau': 1e-2, #DDPG
    'folder_counts': 9,
    'train_days': 180,
    'valid_days': 80,
    'test_days': 30, 
    'image_per_day': 96,
    'roll_offset': 80,
    'total_test_step': 96*30,
    'horizion_steps': 96*180 #PPO
}

STATE_FEATURES = {
    'ratio': 1,
    'gain' : 2,
    'position': 3, 
    'fundrate': 4
}

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

INIT_PATHES = {
    'image': './images/candlesticks',
    'weight': './weights/bitcoin', 
    'train': './results/train/',
    'valid': './results/valid/',
    'test': './results/test/',
    'video': './results/test/video/'
}

def get_account_features(state, account_features, init_balance):
    assert isinstance(account_features, dict)
    account_feats = []
    for name, index in account_features.items():
        if name == 'ratio':
            account_feats.append(state[index])
        elif name == 'gain':
            account_feats.append(state[index] / init_balance * 100)
        elif name == 'position':
            account_feats.append(state[index])
        elif name == 'fundrate': 
            account_feats.append(state[index])
    return account_feats

def get_funding_rate(fundrate_df, date_time):
    assert isinstance(fundrate_df, pd.DataFrame)
    sub_df = fundrate_df.truncate(after=date_time)
    return float(sub_df.iloc[0]['Funding Rate'][:-1])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)
    
    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return states, actions, rewards, dones, next_states

    def get_batch_indices(self, batch_size): # PPO
        batch_start = np.arange(0, len(self.buffer), batch_size)
        indices = np.arange(len(self.buffer), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i: i+ batch_size] for i in batch_start]
        return batches 

class Metrics:
    def __init__(self, init_balance, returns_df):
        assert isinstance(returns_df, pd.DataFrame), "returns should be a dateframe"
        self.returns = returns_df
        self.balance = init_balance
    
    def sortino_ratio(self, rf=0):
        profits = self.returns['profit']
        mu = profits.mean()
        downside = profits.loc[profits < mu].to_numpy()
        sigma = np.sqrt(((downside - mu) ** 2).sum()/(len(profits) - 1))
        return (mu - rf) / sigma

    def sharpe_ratio(self, rf=0):
        profits = self.returns['profit']
        mu = profits.mean()
        sigma = np.sqrt(((profits - mu) ** 2).sum()/(len(profits) - 1))
        return (mu - rf) / sigma
        
    def max_drawdown(self):
        profits = self.returns['profit'].to_numpy()
        peak_values = np.maximum.accumulate(profits)
        down = peak_values - profits
        trough = np.argmax(down)
        if trough > 0:
            peak = np.argmax(profits[:trough])
            return (profits[trough] - profits[peak])/(profits[peak])
        else:
            return 0

    def volatility(self):
        profits = self.returns['profit']
        return profits.std()

    def cumulative_return(self):
        cumulative_return = np.cumsum(self.returns['profit'])
        return cumulative_return / self.balance

    def coverage(self):
        action_names = {0:'Idle', 1:'Long', 2:'Short', 3:'Close'}
        actions = self.returns['action']
        total_count = len(actions)
        action_counts = actions.value_counts().to_dict()
        action_coverage = {k: v / total_count for k, v in action_counts.items()}
        if isinstance(list(action_coverage.keys())[0], int):
            action_coverage = {action_names[k]: v for k, v in action_coverage.items()}
        for a in action_names.values():
            if a not in action_coverage.keys():
                action_coverage[a] = 0
        return action_coverage

    def action_gain(self):
        length = len(self.returns)
        gain = np.zeros((length, 1))
        actions = self.returns['action']
        profits = self.returns['profit']   

        action_gain = 0
        current_gain = 0
        for i, a in enumerate(actions):
            gain[i][0] = current_gain
            action_gain += profits[i]
            if a == 'Close':
                current_gain += action_gain
                gain[i][0] = current_gain
                action_gain = 0  
            elif a == 3:
                current_gain += action_gain
                gain[i][0] = current_gain
                action_gain = 0  
        current_gain += action_gain
        gain[length-1][0] = current_gain
        return gain


        