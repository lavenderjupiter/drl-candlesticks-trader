import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

import libs.environ as environ
import libs.models as models
import libs.utilities as utilities


class DQN:
    def __init__(self, env, backbone, init_balance, hyperparams, acc_feats, device='cpu', raw_data='None'):
        assert isinstance(env, gym.Env)
        self.env = env
        self.init_balance = init_balance
        self.learn_step_counter = 0
        self.device = device
        self._double = False
        self.name = 'dqn'

        self.acc_feats = {name: utilities.STATE_FEATURES[name] for name in acc_feats}

        self.learning_rate = hyperparams['learning_rate']
        self.batch_size = hyperparams['batch_size']
        self.gamma = hyperparams['gamma']
        self.memory = utilities.ExperienceBuffer(hyperparams['replay_buffer_size'])
        self.sync_steps = hyperparams['sync_target_steps']

        self.eval_net = models.DQNNet(env.observation_space.shape, env.action_space.n, backbone, account_feats=len(acc_feats), raw_data=raw_data)
        self.target_net = models.DQNNet(env.observation_space.shape, env.action_space.n, backbone, account_feats=len(acc_feats), raw_data=raw_data)
        self.eval_net = self.eval_net.to(self.device)
        self.target_net = self.target_net.to(self.device)

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def _get_legal_actions(self, state):
        position = state[utilities.STATE_FEATURES['position']]
        legal_actions = environ.get_legal_actions(position)
        return legal_actions

    def choose_action(self, state, epsilon=0):
        self.eval_net.eval()
        
        if np.random.random() < epsilon:
            action = np.random.choice(self.env.legal_actions)
        else:
            action_mask = np.zeros((1, self.env.action_space.n))
            action_mask[0, self.env.legal_actions] = 1
            action_mask = torch.tensor(action_mask)

            image_t = state[0].unsqueeze(0).to(self.device)
            account_feats = utilities.get_account_features(state, self.acc_feats, self.init_balance)
            account_t = torch.tensor(account_feats).unsqueeze(0).to(self.device)
            action_value_t = self.eval_net(image_t, account_t) 
            action_value_t[action_mask == 0] = -9999999 # set q value of illegal action to a large negative number
            _, action_t = torch.max(action_value_t, dim=1)
            action = int(action_t.item())
        
        self.eval_net.train()
        return action

    def store_transition(self, experience):
        self.memory.append(experience)
    
    def learn(self):
        if self.learn_step_counter % self.sync_steps == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print('\nTarget net updated!')
        self.learn_step_counter += 1

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, dones, next_states = batch

        images = [s[0] for s in states]
        accounts = [torch.tensor(utilities.get_account_features(s, self.acc_feats, self.init_balance)) for s in states]        
        next_images = [s[0] for s in next_states]
        next_accounts = [torch.tensor(utilities.get_account_features(s, self.acc_feats, self.init_balance)) for s in next_states]

        target_legal_actions = [self._get_legal_actions(s) for s in next_states]
        target_action_mask = torch.zeros((self.batch_size, self.env.action_space.n))
        for i, a in enumerate(target_legal_actions):
            mask = torch.zeros((1, self.env.action_space.n))
            mask[0, a] = 1
            target_action_mask[i] = mask

        images_t = torch.stack(images).to(self.device)
        accounts_t = torch.stack(accounts).to(self.device)
        next_images_t = torch.stack(next_images).to(self.device)
        next_accounts_t = torch.stack(next_accounts).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)

        state_action_values = self.eval_net(images_t, accounts_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        if self._double: # legal_actions masking is not applied
            next_state_actions = self.eval_net(next_images_t, next_accounts_t).max(1)[1]
            next_action_values = self.target_net(next_images_t, next_accounts_t).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        else:
            target_values = self.target_net(next_images_t, next_accounts_t)
            target_values[target_action_mask == 0] = -9999999
            next_action_values = target_values.max(1)[0] 
        next_action_values[done_mask] = 0.0
        next_action_values = next_action_values.detach()
        expected_action_values = next_action_values * self.gamma + rewards_t

        loss = self.loss_func(state_action_values, expected_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save_model(self, episode, obs_idx, rewards, log_path, check_path):
        rewards_list = [reward for reward in rewards] # deque -> list
        checkpoint = {
            "net": self.eval_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "episode": episode,
            'frame_idx': obs_idx,
            'rewards': rewards_list,
            'log_path': log_path
        }
        torch.save(checkpoint, check_path)  

    def resume(self, path_checkpoint):
        checkpoint = torch.load(path_checkpoint)  
        self.eval_net.load_state_dict(checkpoint['net'])  
        self.optimizer.load_state_dict(checkpoint['optimizer'])  
        start_episode = int(checkpoint['episode'])  
        start_index = int(checkpoint['frame_idx'])
        rewards = checkpoint['rewards'][1:-1].split(',')
        log_path = checkpoint['log_path']
        return start_episode, start_index, rewards, log_path

    def to(self, device):
        self.eval_net = self.eval_net.to(device)
        self.target_net = self.target_net.to(device)

class DoubleDQN(DQN):
    def __init__(self, env, backbone, init_balance, hyperparams, acc_feats, device='cpu', raw_data='None'):
        super(DoubleDQN, self).__init__(env, backbone, init_balance, hyperparams, acc_feats=len(acc_feats), device=device, raw_data=raw_data)
        self.name = 'double-dqn'
        self._double = True
        self.acc_feats = {name: utilities.STATE_FEATURES[name] for name in acc_feats}

class DuelingDQN(DQN):
    def __init__(self, env, backbone, init_balance, hyperparams, acc_feats, device='cpu', raw_data='None'):
        assert isinstance(env, gym.Env)
        self.env = env
        self.init_balance = init_balance
        self.learn_step_counter = 0
        self.device = device
        self._double = False
        self.name = 'dueling-dqn'

        self.acc_feats = {name: utilities.STATE_FEATURES[name] for name in acc_feats}

        self.learning_rate = hyperparams['learning_rate']
        self.batch_size = hyperparams['batch_size']
        self.gamma = hyperparams['gamma']
        self.memory = utilities.ExperienceBuffer(hyperparams['replay_buffer_size'])
        self.sync_steps = hyperparams['sync_target_steps']
        self.learning_rate = hyperparams['learning_rate']

        self.eval_net = models.DuelingNet(env.observation_space.shape, env.action_space.n, backbone, account_feats=len(acc_feats), raw_data=raw_data)
        self.target_net = models.DuelingNet(env.observation_space.shape, env.action_space.n, backbone, account_feats=len(acc_feats), raw_data=raw_data)
        self.eval_net = self.eval_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.eval_net.device = self.device
        self.target_net.device = self.device

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, epsilon=0):
        self.eval_net.eval()
        
        if np.random.random() < epsilon:
            action = np.random.choice(self.env.legal_actions)
        else:
            image_t = state[0].unsqueeze(0).to(self.device) # ohlc
            account_feats = utilities.get_account_features(state, self.acc_feats, self.init_balance)
            account_t = torch.tensor(account_feats).unsqueeze(0).to(self.device)
            action_value_t = self.eval_net(image_t, account_t, [self.env.legal_actions]) 
            _, action_t = torch.max(action_value_t, dim=1)
            action = int(action_t.item())
        
        self.eval_net.train()
        return action

    def learn(self):
        if self.learn_step_counter % self.sync_steps == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print('\nTarget net updated!')
        self.learn_step_counter += 1

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, dones, next_states = batch

        images = [s[0] for s in states]
        accounts = [torch.tensor(utilities.get_account_features(s, self.acc_feats, self.init_balance)) for s in states]        
        next_images = [s[0] for s in next_states]
        next_accounts = [torch.tensor(utilities.get_account_features(s, self.acc_feats, self.init_balance)) for s in next_states]

        target_legal_actions = [self._get_legal_actions(s) for s in next_states]

        images_t = torch.stack(images).to(self.device)
        accounts_t = torch.stack(accounts).to(self.device)
        next_images_t = torch.stack(next_images).to(self.device)
        next_accounts_t = torch.stack(next_accounts).to(self.device)
        actions_t = torch.tensor(actions).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)

        state_action_values = self.eval_net(images_t, accounts_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        next_action_values = self.target_net(next_images_t, next_accounts_t, target_legal_actions).max(1)[0] 
        next_action_values[done_mask] = 0.0
        next_action_values = next_action_values.detach()
        expected_action_values = next_action_values * self.gamma + rewards_t

        loss = self.loss_func(state_action_values, expected_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class PPO:
    def __init__(self, env, backbone, init_balance, hyperparams, acc_feats, device='cpu', raw_data='None'):
        assert isinstance(env, gym.Env)
        self.env = env
        self.init_balance = init_balance
        self.learn_step_counter = 0
        self.device = device
        self.name = 'ppo'

        self.acc_feats = {name: utilities.STATE_FEATURES[name] for name in acc_feats}

        self.actor_learning_rate = hyperparams['actor_learning_rate']
        self.critic_learning_rate = hyperparams['critic_learning_rate']
        self.batch_size = hyperparams['batch_size']
        self.gamma = hyperparams['gamma']
        self.clip_epsilon = hyperparams['clip_epsilon']
        self.gae_lambda = hyperparams['gae_lambda']
        self.ppo_epochs = hyperparams['ppo_epochs']
        self.critic_discount = hyperparams['critic_discount']
        self.entropy_beta = hyperparams['entropy_beta']

        self.actor = models.ActorPPO(env.observation_space.shape, env.action_space.n, backbone, account_feats=len(acc_feats), raw_data=raw_data)
        self.critic = models.CriticPPO(env.observation_space.shape, backbone, account_feats=len(acc_feats), raw_data=raw_data)
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self.loss_func = nn.MSELoss()

        self.memory = utilities.ExperienceBuffer(hyperparams['horizion_steps'])

    def _normalize(self, x):
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x 

    def _get_batch(self, indices, advantages, returns):
        advantages_batch = advantages[indices]
        returns_batch = returns[indices]
        states, actions, probs, values, rewards, dones = zip(*[self.memory.buffer[idx] for idx in indices])
        return states, actions, probs, advantages_batch, returns_batch

    def _get_actions_mask(self, state):
        position = state[utilities.STATE_FEATURES['position']]
        legal_actions = environ.get_legal_actions(position)
        action_mask = torch.zeros((1, self.env.action_space.n))
        action_mask[0, legal_actions] = 1
        return action_mask

    def store_transition(self, experience):
        self.memory.append(experience)

    def choose_action(self, state):
        self.actor.eval()
        self.critic.eval()

        image_t = state[0].unsqueeze(0).to(self.device)
        account_feats = utilities.get_account_features(state, self.acc_feats, self.init_balance)
        account_t = torch.tensor(account_feats).unsqueeze(0).to(self.device)
        action_mask_t = self._get_actions_mask(state).to(self.device)

        dist = self.actor(image_t, account_t, action_mask_t)
        action = dist.sample()
        value = self.critic(image_t, account_t)
        
        log_prob = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        self.actor.train()
        self.critic.train()
        return action, log_prob, value

    def get_advantages(self, values, rewards):
        returns = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
                gae = delta
            else:    
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            returns.insert(0, gae + values[t])
        advantages = np.array(returns) - values
        advantages = torch.tensor(advantages, dtype=torch.float)
        returns = torch.tensor(returns, dtype=torch.float)

        return returns, self._normalize(advantages)

    def learn(self):
        for e in range(self.ppo_epochs):
            print('\rlearning epoch {}'.format(e + 1), end='')

            batches = self.memory.get_batch_indices(self.batch_size)
            values, rewards = [], []
            for transition in self.memory.buffer:
                values.append(transition.value)
                rewards.append(transition.reward)
            
            returns, advantages = self.get_advantages(values, rewards)
            values = torch.tensor(values, dtype=torch.float)

            for batch_indices in batches:
                states_batch, actions_batch, old_probs_batch, advantages_batch, returns_batch = self._get_batch(batch_indices, advantages, returns)

                images = [s[0] for s in states_batch]
                accounts = [torch.tensor(utilities.get_account_features(s, self.acc_feats, self.init_balance)) for s in states_batch]
                action_masks = [self._get_actions_mask(s) for s in states_batch]
                images_t = torch.stack(images).to(self.device)
                accounts_t = torch.stack(accounts).to(self.device)
                actions_t = torch.tensor(actions_batch).to(self.device)
                old_probs_t = torch.tensor(old_probs_batch).to(self.device)
                returns_t = returns_batch.unsqueeze(-1).to(self.device)
                advantages_t = advantages_batch.to(self.device)
                action_masks_t = torch.stack(action_masks).squeeze(1).to(self.device)

                dists = self.actor(images_t, accounts_t, action_masks_t)
                critic_values = self.critic(images_t, accounts_t)
                entropy = dists.entropy().mean()
                new_probs_t = dists.log_prob(actions_t)

                ratio = (new_probs_t - old_probs_t).exp()
                surr1 = ratio* advantages_t
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t

                loss_actor = - torch.min(surr1, surr2).mean()
                loss_critic = self.loss_func(returns_t, critic_values.detach())
                loss = loss_actor + self.critic_discount * loss_critic - self.entropy_beta * entropy

                # update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        print()
        self.memory.clear()
        return loss

    def save_models(self, episode, obs_idx, rewards, log_path, actor_path, critic_path=None):
        rewards_list = [reward for reward in rewards]
        actor_checkpoint = {
            "net": self.actor.state_dict(),
            'optimizer': self.actor_optimizer.state_dict(),
            "episode": episode,
            'frame_idx': obs_idx,
            'rewards': rewards_list,
            'log_path': log_path
        }
        torch.save(actor_checkpoint, actor_path)  

        if critic_path is not None:
            critic_checkpoint = {
                "net": self.critic.state_dict(),
                'optimizer': self.critic_optimizer.state_dict(),
            }
            torch.save(critic_checkpoint, critic_path)  

    def resume(self, actor_path_checkpoint, critic_path_checkpoint=None):
        actor_checkpoint = torch.load(actor_path_checkpoint)  
        self.actor.load_state_dict(actor_checkpoint['net'])  
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer'])  
        
        if critic_path_checkpoint is not None:
            critic_checkpoint = torch.load(critic_path_checkpoint)  
            self.critic.load_state_dict(critic_checkpoint['net'])  
            self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer'])  

        start_episode = actor_checkpoint['episode'] 
        start_index = int(actor_checkpoint['frame_idx'])
        rewards = actor_checkpoint['rewards'][1:-1].split(',')
        log_path = actor_checkpoint['log_path']
        return start_episode, start_index, rewards, log_path

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

