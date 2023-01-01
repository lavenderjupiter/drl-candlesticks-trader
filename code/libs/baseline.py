import numpy as np
import pandas as pd
import time
import collections
import libs.environ as environ

class Baseline:
    def __init__(self, result_path, ini_balance, comm_percent, fps):
        self.result_path = result_path
        self.init_account_balance = ini_balance
        self.commision_percent = comm_percent / 100
        self.FPS = fps
        self.channels = 3
        self.height = 224
        self.width = 224

    def randomized_policy(self, test_set):
        seed = np.random.randint(0, high=100)
        np.random.seed(seed)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        filename = self.result_path + 'rp_tc_' + str(self.commision_percent * 100) + '@' + timestamp + '.csv'
        date_times, actions, profits = [], [], []
        cum_profit = 0

        env = environ.BitcoinEnv(test_set, image_shape=(self.channels, self.height, self.width), balance=self.init_account_balance,
                                threshold=1, commission_perc=self.commision_percent)
        _ = env.reset()
        c = collections.Counter()
        current_step = 0

        while True:
            start_ts = time.time()
            env.render()
            current_step += 1
            print('\rcurrent step: {}'.format(current_step), end='')

            action = np.random.choice(env.legal_actions)
            c[action] += 1
            actions.append(action)
            state, reward, done, info = env.step(action)
            cum_profit += info['profit']
            date_times.append(info['timestamp']) 
            profits.append(info['profit'])
            if done:
                break

            delta = 1 / self.FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)

        result = pd.DataFrame(data={'datetime': date_times, 'action': actions, 'profit': profits})
        result.to_csv(filename)  

        print()
        msg_gain = 'Total gain: %.2f' % cum_profit
        msg_action = 'Action counts: {}'.format(c)
        print(msg_gain)
        print(msg_action)
        env.close()

    def buy_and_hold(self, test_set):
        filename = self.result_path + 'bh_tc_' + str(self.commision_percent * 100) + '.csv'
        date_times, actions, profits = [], [], []
        cum_profit = 0
        steps = len(test_set)

        env = environ.BitcoinEnv(test_set, image_shape=(self.channels, self.height, self.width), balance=self.init_account_balance,
                                threshold=1, commission_perc=self.commision_percent)

        state = env.reset()
        c = collections.Counter()
        trade_count = 0

        while True:
            trade_count += 1
            start_ts = time.time()
            env.render()
            print('\rcurrent step: {}'.format(trade_count + 1), end='')

            if trade_count == 1:
                action = environ.Actions.LONG.value
            elif trade_count == steps:
                action = environ.Actions.CLOSE.value
            else:
                action = environ.Actions.IDLE.value
            c[action] += 1
            actions.append(action)
            _, reward, done, info = env.step(action)
            cum_profit += info['profit']
            date_times.append(info['timestamp']) 
            profits.append(info['profit'])
            if done:
                break

            delta = 1 / self.FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)

        result = pd.DataFrame(data={'datetime': date_times, 'action': actions, 'profit': profits})
        result.to_csv(filename)

        print()
        msg_gain = 'Total gain: %.2f' % cum_profit
        msg_action = 'Action counts: {}'.format(c)
        print(msg_gain)
        print(msg_action)
        env.close()

    def range_break(self, dataframe, factor=0.5):
        assert isinstance(dataframe, pd.DataFrame), "test_set should be a dateframe"
        balance = self.init_account_balance
        filename = self.result_path + 'hp_tc_' + str(self.commision_percent * 100) + '_factor' + str(factor) + '.csv'
        date_times, actions, profits = [dataframe.index.values[0]], [0], [0]
        cum_profit = 0
        c = collections.Counter()

        for j in range(1, len(dataframe)):
            current_open, current_high, current_low, current_close = dataframe.iloc[j][1:5]

            last_high = dataframe.iloc[j-1]['High']
            last_low = dataframe.iloc[j-1]['Low']
            last_range = last_high - last_low
            upward = current_open + factor * last_range
            downward = current_open - factor * last_range

            if current_high > upward:
                # -----open long position-----
                action = environ.Actions.LONG.value
                open_value = balance
                transaction_cost = self.commision_percent * balance
                balance -= transaction_cost
                amount = balance / upward
                balance = 0
                # -----close the position-----
                portfolio_value = amount * current_close
                transaction_cost = self.commision_percent * portfolio_value
                balance += (portfolio_value - transaction_cost)
                close_value = balance
                profit = close_value - open_value
            elif current_high <= upward and current_low >= downward:
                action = environ.Actions.IDLE.value
                profit = 0
            elif current_low < downward:
                # -----open short position-----
                action = environ.Actions.SHORT.value
                open_value = balance
                transaction_cost = self.commision_percent * balance
                portfolio_value = balance - transaction_cost
                balance += portfolio_value
                amount = portfolio_value / downward
                # -----close the position-----
                portfolio_value = amount * current_close
                transaction_cost = self.commision_percent * portfolio_value
                balance -= (portfolio_value + transaction_cost)
                close_value = balance
                profit = close_value - open_value 
   
            c[action] += 1
            actions.append(action)
            timestamp = dataframe.index.values[j]
            date_times.append(timestamp) 
            profits.append(profit)
            cum_profit += profit

        result = pd.DataFrame(data={'datetime': date_times, 'action': actions, 'profit': profits})
        result.to_csv(filename)

        print()
        msg_gain = 'Total gain: %.2f' % cum_profit
        msg_action = 'Action counts: {}'.format(c)
        print(msg_gain)
        print(msg_action)