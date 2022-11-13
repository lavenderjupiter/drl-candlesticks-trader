import collections
from collections import deque
import numpy as np
import time
import os
import pandas as pd

import torch
import torch.nn as nn
import torchvision.models as torch_models

import libs.ohlc_environ as environ
import libs.agents as agents
import libs.models as models
import libs.ensemble as ensemble
from libs.utilities import HYPERPARAMS, INIT_PATHES, make_path
from libs.baseline import Baseline
from tensorboardX import SummaryWriter

INITIAL_ACCOUNT_BALANCE = 100000
COMMISSION_PERCENT = 0.05
LEARN_THRESHOLD = 0.2
ACCOUNT_FEATURES = ['position', 'fundrate'] 

WEIGHT_DIR = make_path(INIT_PATHES['weight'])
TRAIN_RESULT_DIR = make_path(INIT_PATHES['train'])
VALID_RESULT_DIR = make_path(INIT_PATHES['valid'])
TEST_RESULT_DIR = make_path(INIT_PATHES['test'])

RESNET = torch_models.resnet18(pretrained=False)
BACKBONE = nn.Sequential(*( list(RESNET.children())[:-1]))
MODE = None

class train:
    def __init__(self, device, name):
        print('using device:', device)
        self.device = device
        self.name = name.lower().replace(' ', '')
        self.message = self.name + '(' + MODE + '-raw)'

        if self.name == 'ppo':
            self.experience = collections.namedtuple('Experience', field_names=['state', 'action', 'probability', 'value', 'reward', 'done'])
        else:
            self.experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])

    def run(self, count, train_set, episodes=50, run_checkpoint=False):
        message = self.message + '_agent' + str(count) # dqn(a)_tc0_agent0
        path_checkpoint = WEIGHT_DIR + '-' + message + '_checkpoint.pth' # dqn(a)_tc0_agent0_checkpoint.pth
        actor_path_checkpoint = WEIGHT_DIR + '-' + message + '_checkpoint_actor.pth'
        critic_path_checkpoint = WEIGHT_DIR + '-' + message + '_checkpoint_critic.pth'
        path_best = WEIGHT_DIR + '-' + message + '_best.pth' # dqn(a)_tc0_agent0_best.pth
        start_episode = 0
        obs_idx = 0
        done_idx = 0
        best_mean_reward = None
        rewards =  deque(maxlen=10)

        writer = SummaryWriter(comment='_bitcoin-' + self.message)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        filename = TRAIN_RESULT_DIR + message + '_tc' + str(COMMISSION_PERCENT) + '@' + timestamp + '.txt'

        env = environ.BitcoinEnv(train_set, balance=INITIAL_ACCOUNT_BALANCE, threshold=LEARN_THRESHOLD, commission_perc=COMMISSION_PERCENT)

        print(self.name + '@' + timestamp)
        if self.name.__contains__('dqn'):
            if self.name == 'double-dqn':
                agent = agents.DoubleDQN(env, BACKBONE, INITIAL_ACCOUNT_BALANCE, HYPERPARAMS, acc_feats=ACCOUNT_FEATURES, device=self.device, raw_data=MODE)
            elif self.name == 'dueling-dqn':
                agent = agents.DuelingDQN(env, BACKBONE, INITIAL_ACCOUNT_BALANCE, HYPERPARAMS, acc_feats=ACCOUNT_FEATURES, device=self.device, raw_data=MODE)
            else:
                agent = agents.DQN(env, BACKBONE, INITIAL_ACCOUNT_BALANCE, HYPERPARAMS, acc_feats=ACCOUNT_FEATURES, device=self.device, raw_data=MODE)
                
            if run_checkpoint:
                start_episode, obs_idx, reward_list, filename = agent.resume(path_checkpoint)
                rewards = deque([float(r) for r in reward_list], maxlen=10) 
                best_mean_reward = np.mean(rewards)
            with open(filename, 'a') as file:
                try:
                    for e in range(start_episode, episodes):
                        print('--- episode:{} ---'.format(e + 1))
                        ts = time.time()
                        state = env.reset(e == start_episode)
                        episode_reward = 0

                        while True:
                            obs_idx += 1
                            epsilon = max(HYPERPARAMS['epsilon_final'], HYPERPARAMS['epsilon_start'] - obs_idx / HYPERPARAMS['epsilon_decay_last_frame'])
                            action = agent.choose_action(state, epsilon)
                            next_state, reward, done, info = env.step(action)
                            exp = self.experience(state, action, reward, done, next_state)
                            agent.store_transition(exp)
                            episode_reward += reward
                            state = next_state
                            print('\rcurrent step: {}'.format(obs_idx), end='')

                            if len(agent.memory) >= HYPERPARAMS['replay_buffer_size']:
                                loss = agent.learn()

                            if done:
                                rewards.append(episode_reward)
                                speed = (obs_idx - done_idx) / (time.time() - ts)
                                mean_reward = np.mean(rewards)
                                done_msg = 'done %d steps, mean reward %.3f, eps %.2f, speed %.2f f/s' % (
                                    (obs_idx - done_idx), mean_reward, epsilon, speed)
                                print('\n' + done_msg)
                                file.write('--e:{}, '.format(e + 1) + done_msg + '\n')
                                file.flush()
                                writer.add_scalar("epsilon", epsilon, obs_idx)
                                writer.add_scalar("reward", episode_reward, obs_idx)
                                writer.add_scalar("reward_10", mean_reward, obs_idx)
                                writer.add_scalar("loss", loss, obs_idx)
                                done_idx = obs_idx
                                ts = time.time()

                                if e > 2:
                                    if best_mean_reward is None or best_mean_reward < mean_reward:
                                        agent.save_model(e, obs_idx, rewards, filename, path_best)
                                        if best_mean_reward is not None:
                                            weight_update_msg = 'Best mean reward updated %.3f -> %.3f, model saved' % (best_mean_reward, mean_reward)
                                            print(weight_update_msg)
                                            file.write(weight_update_msg + '\n')
                                            file.flush()
                                        best_mean_reward = mean_reward
                                break
                        agent.save_model(e, obs_idx, rewards, filename, path_checkpoint) 
                        torch.cuda.empty_cache()
                except KeyboardInterrupt:
                    agent.save_model(e, obs_idx, rewards, filename, path_checkpoint) 

        elif self.name == 'ppo':
            agent = agents.PPO(env, BACKBONE, INITIAL_ACCOUNT_BALANCE, HYPERPARAMS, acc_feats=ACCOUNT_FEATURES, device=self.device, raw_data=MODE)
            if run_checkpoint:
                start_episode, reward_list, filename = agent.resume(actor_path_checkpoint, critic_path_checkpoint)
                rewards = deque([float(r) for r in reward_list], maxlen=10) 
                best_mean_reward = np.mean(rewards)
            with open(filename, 'a') as file:
                try:
                    for e in range(start_episode, episodes):
                        print('--- episode:{} ---'.format(e + 1))
                        ts = time.time()
                        state = env.reset(e == start_episode)
                        episode_reward = 0

                        while True:
                            obs_idx += 1
                            action, log_prob, value= agent.choose_action(state)
                            next_state, reward, done, info = env.step(action)
                            exp = self.experience(state, action, log_prob, value, reward, done)
                            agent.store_transition(exp)
                            episode_reward += reward
                            state = next_state
                            print('\rcurrent step: {}'.format(obs_idx), end='')

                            if done:
                                print()
                                loss = agent.learn()

                                rewards.append(episode_reward)
                                speed = (obs_idx - done_idx) / (time.time() - ts)
                                mean_reward = np.mean(rewards)
                                done_msg = 'done %d steps, mean reward %.3f, speed %.2f f/s' % (
                                    (obs_idx - done_idx), mean_reward, speed)
                                done_idx = obs_idx
                                ts = time.time()
                                print(done_msg)
                                file.write('--e:{}, '.format(e + 1) + done_msg + '\n')
                                file.flush()
                                writer.add_scalar("reward", episode_reward, obs_idx)
                                writer.add_scalar("reward_10", mean_reward, obs_idx)
                                writer.add_scalar("loss", loss, obs_idx)

                                if e > 2:
                                    if best_mean_reward is None or best_mean_reward < mean_reward:
                                        agent.save_models(e, obs_idx, rewards, filename, path_best, None)
                                        if best_mean_reward is not None:
                                            weight_update_msg = 'Best mean reward updated %.3f -> %.3f, model saved' % (best_mean_reward, mean_reward)
                                            print(weight_update_msg)
                                            file.write(weight_update_msg + '\n')
                                            file.flush()
                                        best_mean_reward = mean_reward
                                break    
                        agent.save_models(e, obs_idx, rewards, filename, actor_path_checkpoint, critic_path_checkpoint) 
                        torch.cuda.empty_cache()
                except KeyboardInterrupt:
                    agent.save_models(e, obs_idx, rewards, filename, actor_path_checkpoint, critic_path_checkpoint) 
        
        agent.to('cpu')
        writer.close()

class valid:
    def __init__(self, name):
        self.name = name.lower().replace(' ', '')
        self.message = self.name + '(' + MODE + '-raw)'

    def run(self, count, valid_set):
        message = self.message + '_agent' + str(count)
        filename = VALID_RESULT_DIR + message + '_tc' + str(COMMISSION_PERCENT) + '.csv'
        log_filename = VALID_RESULT_DIR + 'log/' + message + '_tc' + str(COMMISSION_PERCENT) + '.txt'
        model_path = WEIGHT_DIR + '-' + message + '_best.pth'
        date_times, actions, profits = [], [], []

        env = environ.BitcoinEnv(valid_set, balance=INITIAL_ACCOUNT_BALANCE, threshold=1, commission_perc=COMMISSION_PERCENT)

        if self.name == 'dqn' or self.name == 'double-dqn':
            net = models.DQNNet(env.observation_space.shape, env.action_space.n, BACKBONE, account_feats=len(ACCOUNT_FEATURES), raw_data=MODE)
        elif self.name == 'dueling-dqn':
            net = models.DuelingNet(env.observation_space.shape, env.action_space.n, BACKBONE, account_feats=len(ACCOUNT_FEATURES), raw_data=MODE)
        elif self.name == 'ppo':
            net = models.ActorPPO(env.observation_space.shape, env.action_space.n, BACKBONE, account_feats=len(ACCOUNT_FEATURES), raw_data=MODE)
        params = torch.load(model_path, map_location='cpu')  
        net.load_state_dict(params['net'])
        net.eval()

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        print(self.name + '@' + timestamp)

        with open(log_filename, 'a') as log:
            log.truncate(0)
            state = env.reset()
            ohlc, _, gain, position, fund_rate = state
            c = collections.Counter()
            current_step = 0

            with torch.no_grad():
                while True:
                    current_step += 1
                    print('\rcurrent step: {}'.format(current_step), end='')

                    account_feats = []
                    for name in ACCOUNT_FEATURES:
                        if name == 'gain':
                            account_feats.append(gain)
                        elif name == 'position':
                            account_feats.append(position)
                        elif name == 'fundrate':
                            account_feats.append(fund_rate)

                    ohlc_t = ohlc.unsqueeze(0)
                    account_t = torch.tensor(account_feats).unsqueeze(0)
                    if self.name == 'ppo':
                        action_mask = torch.zeros((1, env.action_space.n))
                        action_mask[0, env.legal_actions] = 1
                        dist = net(ohlc_t, account_t, action_mask)
                        action = dist.sample().data.numpy()[0]
                    else:
                        action_mask = np.zeros((1, env.action_space.n))
                        action_mask[0, env.legal_actions] = 1
                        action_mask = torch.tensor(action_mask)

                        q_values = net(ohlc_t, account_t)
                        q_values[action_mask == 0] = -9999999
                        action = np.argmax(q_values.data.numpy()[0])
                    c[action] += 1
                    actions.append(action)
                    [ohlc, _, gain, position, fund_rate], _, done, info = env.step(action)
                    date_times.append(info['timestamp']) 
                    profits.append(info['profit'])
                    log_msg = 'position: {}, gain: {:.4f}'.format(position, gain)
                    log.write(log_msg + '\n')
                    if done:
                        break

            result = pd.DataFrame(data={'datetime': date_times, 'action': actions, 'profit': profits})
            result.to_csv(filename)

            print()
            msg_gain = 'Total gain: %.2f' % sum(profits)
            msg_action = 'Action counts: {}'.format(c)
            print(msg_gain)
            print(msg_action)
            log.write(msg_gain + '\n')
            log.write(msg_action + '\n')

class test:
    def __init__(self, name, datetime):
        self.name = name.lower().replace(' ', '')
        self.test_path = make_path(TEST_RESULT_DIR + datetime.split(' ')[0] + '/')
        self.test_log_path = make_path(self.test_path + '/log/')
        self.message = self.name + '(' + MODE + '-raw)'

    def run(self, test_set, record=False):
        decisions = np.zeros((len(test_set), HYPERPARAMS['folder_counts']), dtype=int)
        profits = np.zeros((len(test_set), HYPERPARAMS['folder_counts']), dtype=np.float64)
        message = self.message + '_agent' # dqn(a)_agent
        result_filename = self.test_path + self.message + '_tc' + str(COMMISSION_PERCENT) + '_ensemble_result.csv' # save result
        actions_filename = self.test_path + self.message + '_tc' + str(COMMISSION_PERCENT) + '_agent_actions.csv' # save result
        profits_filename = self.test_path + self.message + '_tc' + str(COMMISSION_PERCENT) + '_agent_profits.csv' # save result
        log_filename = self.test_log_path + self.message + '_tc' + str(COMMISSION_PERCENT) + '.txt'
        model_paths = [WEIGHT_DIR + '-' + message + str(a) + '.pth' for a in range(HYPERPARAMS['folder_counts'])]
        date_times, final_actions, final_profits = [], [], []

        env = environ.BitcoinEnv(test_set, balance=INITIAL_ACCOUNT_BALANCE, threshold=1, commission_perc=COMMISSION_PERCENT)

        if self.name == 'dqn' or self.name == 'double-dqn':
            net = models.DQNNet(env.observation_space.shape, env.action_space.n, BACKBONE, account_feats=len(ACCOUNT_FEATURES), raw_data=MODE)
        elif self.name == 'dueling-dqn':
            net = models.DuelingNet(env.observation_space.shape, env.action_space.n, BACKBONE, account_feats=len(ACCOUNT_FEATURES), raw_data=MODE)
        elif self.name == 'ppo':
            net = models.ActorPPO(env.observation_space.shape, env.action_space.n, BACKBONE, account_feats=len(ACCOUNT_FEATURES), raw_data=MODE)

        print('agent:', self.message)
        if os.path.exists(actions_filename) == False:
            print('choosing actions for each agent')
            for a in range(HYPERPARAMS['folder_counts']):
                print('--- agent {} ---'.format(a + 1))
                params = torch.load(model_paths[a], map_location='cpu')  #dqn(a)_tc0_agent0_best.pth
                net.load_state_dict(params['net'])
                net.eval()
                state = env.reset()
                ohlc, ratio, gain, position, fund_rate = state
                current_step = 0

                with torch.no_grad():
                    while True:
                        print('\rcurrent step: {}'.format(current_step + 1), end='')

                        account_feats = []
                        for name in ACCOUNT_FEATURES:
                            if name == 'ratio':
                                account_feats.append(ratio)
                            elif name == 'gain':
                                account_feats.append(gain)
                            elif name == 'position':
                                account_feats.append(position)
                            elif name == 'fundrate':
                                account_feats.append(fund_rate)

                        ohlc_t = ohlc.unsqueeze(0)
                        account_t = torch.tensor(account_feats).unsqueeze(0)
                        if self.name == 'ppo':
                            action_mask = torch.zeros((1, env.action_space.n))
                            action_mask[0, env.legal_actions] = 1
                            dist = net(ohlc_t, account_t, action_mask)
                            action = dist.sample().data.numpy()[0]
                        else:
                            action_mask = np.zeros((1, env.action_space.n))
                            action_mask[0, env.legal_actions] = 1
                            action_mask = torch.tensor(action_mask)

                            q_values = net(ohlc_t, account_t)
                            q_values[action_mask == 0] = -9999999
                            action = np.argmax(q_values.data.numpy()[0])
                        decisions[current_step, a] = action
                        [ohlc, ratio, gain, position, fund_rate], _, done, info = env.step(action)
                        profits[current_step, a] = info['profit']
                        current_step += 1
                        if done:
                            break
                print()

            agent_actions = pd.DataFrame(decisions, columns=['agent'+ str(a) for a in range(HYPERPARAMS['folder_counts'])])
            agent_actions.to_csv(actions_filename)
            agent_profits = pd.DataFrame(profits, columns=['agent'+ str(a) for a in range(HYPERPARAMS['folder_counts'])])
            agent_profits.to_csv(profits_filename)
        else:
            decisions = pd.read_csv(actions_filename, index_col=0).to_numpy()

        print('+---- agents ensemble ----+')
        self.ensemble = ensemble.Ensemble(HYPERPARAMS['folder_counts'], self.message, self.test_path, INITIAL_ACCOUNT_BALANCE, COMMISSION_PERCENT)
        
        with open(log_filename, 'a') as log:
            log.truncate(0)
            state = env.reset(ensemble=True)
            ohlc, ratio, gain, position, fund_rate = state
            c = collections.Counter()
            current_step = 0

            while True:
                agent_decisions = decisions[current_step]
                ensemble_action = self.ensemble.make_decision(agent_decisions)
                env.set_ensemble(agent_decisions, ensemble=True)
                [ohlc, ratio, gain, position, fund_rate], _, done, info = env.step(ensemble_action)
                legal_action = env.action
                final_actions.append(legal_action)
                c[legal_action] += 1

                current_step += 1
                print('\rdone {} step'.format(current_step), end='')
                date_times.append(info['timestamp']) 
                final_profits.append(info['profit'])
                log_msg = 'position: {}, gain: {:.4f}'.format(position, gain)
                log.write(log_msg + '\n')
                if done:
                    break

            result = pd.DataFrame(data={'datetime': date_times, 'action': final_actions, 'profit': final_profits})
            result.to_csv(result_filename)

            print()
            msg_gain = 'Total gain: %.2f' % sum(final_profits)
            msg_action = 'Action counts: {}'.format(c)
            print(msg_gain)
            print(msg_action)
            log.write(msg_gain + '\n')
            log.write(msg_action + '\n')
