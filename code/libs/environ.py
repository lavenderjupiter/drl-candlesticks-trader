import enum
import gym
import numpy as np
from gym import spaces
import enum
import random
import cv2
from PIL import Image, ImageFont, ImageDraw 
from libs.utilities import HYPERPARAMS

FONT_PATH = './libs/Monaco.ttf'
FONT = ImageFont.truetype(FONT_PATH, 12)
TEXT_COLOR_TITLE = (153, 255, 255) # BGR
TEXT_COLOR_AGENTS = (204, 204, 255) # BGR
TEXT_COLOR_RESULT = (238, 200, 222) # BGR
TEXT_COLOR_ENSEMBLE = (238, 238, 175) # BGR
    
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

    def __init__(self, dataset, image_shape, balance, threshold=0.2, commission_perc=0):
        self.dataset = dataset
        self.init_balance = balance
        self.threshold = threshold
        self.commission_rate = commission_perc / 100
        self.action_space = spaces.Discrete(n=len(Actions))
        self.legal_actions = list(range(len(Actions)))
        self.channel, self.height, self.width = image_shape
        self.observation_space = spaces.Box(low=-1, high=1, shape=image_shape, dtype=np.float32)

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
  
        self.image, self.ratio, self.prices, self.fund_rate, self.date_time = self._get_observation()
        self.canvas = self._draw_candlestick(self.image)
        return [self.image, self.ratio, self.gain, self.current_position, self.fund_rate]
    
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
        self.canvas = self._draw_candlestick(self.image, action)
        info = {'profit':self.step_gain, 'timestamp':self.date_time}
        self.legal_actions = get_legal_actions(self.current_position)

        if self.current_step < len(self.dataset) - 1:
            self.current_step += 1
            self.image, self.ratio, self.prices, self.fund_rate, self.date_time = self._get_observation()
            done = False
        else:
            done = True
        
        next_state = [self.image, self.ratio, self.gain, self.current_position, self.fund_rate]
        return next_state, self.reward, done, info

    def set_ensemble(self, actions, ensemble=True):
        self.actions = actions
        self.ensemble = ensemble    
   
    def render(self, mode='human'):
        assert mode in ["human", "rgb_array"]
        if mode == 'human':
            cv2.imshow("Automated Cryptocurrency Trading", self.canvas)
            cv2.waitKey(10)
        elif mode == 'rgb_array':
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()

    def _get_observation(self):
        return self.dataset[self.current_step]

    def _draw_candlestick(self, obs, decision=None):
        img_title = Image.new('RGB', size=(self.width, 30)) 
        img_edit = ImageDraw.Draw(img_title)

        img_candel = obs * 0.5 + 0.5 # de-normalization x=(x-mean)/std -> x=x*std+mean
        img_candel = img_candel.numpy()
        img_candel = np.transpose(img_candel, (1, 2, 0)) # swap axis 0->1, 1->2, 2->0, image.shape => (224, 224, 3). 
        img_candel = img_candel.clip(0, 1)
        img_candel = cv2.cvtColor(img_candel, cv2.COLOR_RGB2BGR)

        if self.ensemble == False:
            text = 'Action: {} | Reward: {:.2f}'.format(self.action, self.reward)
            img_edit.text((5,10), text, TEXT_COLOR_TITLE, font=FONT)
            img_title = (np.asarray(img_title) / 255).astype('float32')
            img_dashboard = cv2.vconcat([img_title, img_candel])
        else:
            text = 'Transaction Cost: {}%'.format(self.commission_rate * 100)
            img_edit.text((5,10), text, TEXT_COLOR_TITLE, font=FONT)
            img_title = (np.asarray(img_title) / 255).astype('float32')

            img_detail = Image.new('RGB', size=(self.width, 66))
            img_edit = ImageDraw.Draw(img_detail)
            text = 'Legal Action: {} \nCurrent Profit: {:.4f} \nCum Gain/Loss: {:.4f}'.format(self.action, self.step_gain, self.gain)
            img_edit.text((10,10), text, TEXT_COLOR_RESULT, font=FONT)
            img_detail = (np.asarray(img_detail) / 255).astype('float32')
            
            img_left = cv2.vconcat([img_title, img_candel, img_detail])

            img_right = Image.new('RGB', size=(self.width, 320))
            img_edit = ImageDraw.Draw(img_right)
            text = 'Ensemble Result'
            img_edit.text((20,20), text, TEXT_COLOR_TITLE, font=FONT)

            if len(self.actions) > 0:
                for i in range(HYPERPARAMS['folder_counts']):
                    text = 'Agent: {},  Decision: {}'.format(i + 1, self.actions[i])
                    img_edit.text((20, 70 + 20 * i), text, TEXT_COLOR_AGENTS, font=FONT)

            text = 'Ensemble Decision: {}'.format(decision)
            img_edit.text((20,260), text, TEXT_COLOR_ENSEMBLE, font=FONT)
            img_right = (np.asarray(img_right) / 255).astype('float32')

            img_dashboard = cv2.hconcat([img_left, img_right])

        return img_dashboard
        
