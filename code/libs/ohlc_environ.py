from cmath import inf
import enum
import gym
import numpy as np
from gym import spaces
import enum
import random

from libs.utilities import HYPERPARAMS
    
class Actions(enum.Enum):
    IDLE = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3

def get_legal_actions(position):
    if position == 0:
        legal_actions = [Actions.LONG.value, Actions.SHORT.value, Actions.IDLE.value]
    elif position == 1 or position == -1:
        legal_actions = [Actions.IDLE.value, Actions.CLOSE.value]
    return legal_actions

class BitcoinEnv(gym.Env):
    metadata = {'render.mode': ['human']}

    def __init__(self, dataset, balance, threshold=0.2, commission_perc=0):
        self.dataset = dataset
        self.init_balance = balance
        self.threshold = threshold
        self.commission_rate = commission_perc / 100
        self.action_space = spaces.Discrete(n=len(Actions))
        self.legal_actions = list(range(len(Actions)))
        self.observation_space = spaces.Box(low=0, high=1, shape=dataset[0][0].shape, dtype=np.float32)

    def reset(self, is_buff_emp=False, ensemble=False):
        self.balance = self.init_balance
        self.action = 'None'
        self.amount = 0.0
        self.reward = 0.0
        self.step_gain = 0.0
        self.gain = 0.0
        self.current_position = 0 # have no position
        self.legal_actions = get_legal_actions(self.current_position)
        self.ensemble = ensemble
        self.actions = []
        if is_buff_emp == False:
            end_index = int(len(self.dataset) * (1 - self.threshold))
        else:
            end_index = len(self.dataset) - HYPERPARAMS['replay_buffer_size'] # data size should more than replay_buffer_size
        self.current_step = random.randint(0, end_index)
  
        self.ohlc, self.prices, self.fund_rate, self.date_time = self._get_observation()
        return [self.ohlc, None, self.gain, self.current_position, self.fund_rate]
    
    def step(self, action):
        this_price, next_price = self.prices

        if action == Actions.IDLE.value:
            self.action = 'Idle'
            if self.current_position == 0:
                self.step_gain = 0
            elif self.current_position == 1:
                self.step_gain = (next_price - this_price) * self.amount
            else: # self.current_position == -1
                self.step_gain = (this_price - next_price) * self.amount             
            self.reward = self.step_gain
            self.gain += self.step_gain            
        elif action == Actions.LONG.value:
            if self.current_position == 0:
                self.action = 'Long'
                self.current_position = 1
                transaction_cost = self.commission_rate * self.balance
                self.balance -= transaction_cost
                self.amount = self.balance / this_price
                self.step_gain = (next_price - this_price) * self.amount
                self.reward = self.step_gain
                self.balance = 0
            elif self.current_position == 1: # action idle long->long
                self.action = 'Idle'
                self.step_gain = (next_price - this_price) * self.amount
            else: # self.current_position == -1 # action idle short->short
                self.action = 'Idle'
                self.step_gain = (this_price - next_price) * self.amount                
            self.gain += self.step_gain
        elif action == Actions.SHORT.value:
            if self.current_position == 0:
                self.action = 'Short'
                self.current_position = -1
                short_balance = self.balance
                transaction_cost = self.commission_rate * short_balance
                short_balance -= transaction_cost
                self.amount = short_balance / this_price
                self.step_gain = (this_price - next_price) * self.amount
                self.reward = self.step_gain
                self.balance += short_balance 
            elif self.current_position == -1: # action idle short->short
                self.action = 'Idle'
                self.step_gain = (this_price - next_price) * self.amount
            else: # self.current_position == 1, action idle long->long
                self.action = 'Idle'
                self.step_gain = (next_price - this_price) * self.amount
            self.gain += self.step_gain
        else: # action == Actions.CLOSE.value:
            self.action = 'Close'
            if self.current_position == 1:
                portfolio_value = self.amount * this_price
                transaction_cost = self.commission_rate * portfolio_value
                self.balance += portfolio_value - transaction_cost
                self.reward = (this_price - next_price) * self.amount - transaction_cost
                self.step_gain = - transaction_cost
            elif self.current_position == -1:
                cover_value = self.amount * this_price
                transaction_cost = self.commission_rate * cover_value
                self.balance -= cover_value - transaction_cost
                self.reward = (next_price - this_price) * self.amount - transaction_cost
                self.step_gain = -transaction_cost
            else:
                self.action = 'Idle'
                self.step_gain = 0
            self.current_position = 0
            self.amount = 0
            self.gain += self.step_gain

        self.reward /= 100 # reward scaling
        info = {'profit':self.step_gain, 'timestamp':self.date_time}
        self.legal_actions = get_legal_actions(self.current_position)

        if self.current_step < len(self.dataset) - 1:
            self.current_step += 1
            self.ohlc, self.prices, self.fund_rate, self.date_time = self._get_observation()
            done = False
        else:
            done = True
        
        next_state = [self.ohlc, None, self.gain, self.current_position, self.fund_rate]
        return next_state, self.reward, done, info

    def set_ensemble(self, actions, ensemble=True):
        self.actions = actions
        self.ensemble = ensemble    

    def _get_observation(self):
        return self.dataset[self.current_step]

    
