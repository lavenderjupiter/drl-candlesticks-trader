from libs.utilities import *
import pandas as pd
import enum
import random

class Decisions(enum.Enum):
    IDLE = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3

class Ensemble:
    def __init__(self, folders, name, ensemble_path, alpha, init_balance, commission_percent, use_attention=False, ratio='sharpe'):
        self.name = name
        self.alpha = alpha
        self.folders = folders
        self.commission_percent = commission_percent
        self.weight_path = INIT_PATHES['weight']
        self.result_path = INIT_PATHES['valid']
        self.ensemble_path = ensemble_path

        if use_attention == True:
            self.message = self.name + '(a)_agent'
        else:
            self.message = self.name + '_agent'

        self.sortino_ratios = self._cal_sortinos(init_balance)
        self.sharpe_ratios = self._cal_sharpes(init_balance)
        self.ratios = self.sharpe_ratios if (ratio == 'sharpe') else self.sortino_ratios
        self.agent_indices, _ = self._get_maximum(self.ratios)
        self.vote_weights = self._cal_weights()

        ensemble_df = pd.DataFrame(
            np.array([self.sortino_ratios, self.sharpe_ratios, self.vote_weights]), 
            columns=['agent' + str(a) for a in range(self.folders)],
            index=['sortino', 'sharpe', 'weight'])
        ensemble_df.to_csv(self.ensemble_path + self.message[:-6] 
            + '_tc' + str(self.commission_percent) 
            + '_ensemble_weights@alpha_' + str(self.alpha) + '.csv')

    def _cal_sortinos(self, init_balance):
        ratios = []
        for folder in range(self.folders):
            returns_df = pd.read_csv(self.result_path + self.message + str(folder) 
                + '_tc' + str(self.commission_percent) + '.csv')
            sortino_ratio = Metrics(init_balance, returns_df).sortino_ratio()
            ratios.append(sortino_ratio)
        return ratios

    def _cal_sharpes(self, init_balance):
        ratios = []
        for folder in range(self.folders):
            returns_df = pd.read_csv(self.result_path + self.message + str(folder) 
                + '_tc' + str(self.commission_percent) + '.csv')
            sharpe_ratio = Metrics(init_balance, returns_df).sharpe_ratio()
            ratios.append(sharpe_ratio)
        return ratios

    def _cal_weights(self):
        weights = []
        denominator = sum([pow((1 - self.alpha), k) for k in range(self.folders)])
        for i, ratio in enumerate(self.ratios):
            if np.isnan(ratio):
                weight = 0
            else:
                weight = ratio * pow((1 - self.alpha), self.folders - (i + 1)) / denominator
            weights.append(weight)
        return weights

    def _get_maximum_three(self, ratio, num=3):
        ratios = list(zip(range(len(ratio)), ratio))
        ratios.sort(key = lambda x:x[1], reverse=True)
        ratio_dict = dict(ratios[:num])
        return list(ratio_dict.keys()), list(ratio_dict.values())

    def make_decision(self, actions):
        action_weights = {            
            Decisions.IDLE.value: [], 
            Decisions.LONG.value: [], 
            Decisions.SHORT.value: [], 
            Decisions.CLOSE.value: [],}
        actions_three = actions[self.agent_indices]
        vote_weights = self.vote_weights[self.agent_indices]
        for i, action in enumerate(actions_three):
            action_weights[action].append(vote_weights[i])

        decision_weights = {
            Decisions.IDLE.value: -9999, 
            Decisions.LONG.value: -9999, 
            Decisions.SHORT.value: -9999, 
            Decisions.CLOSE.value: -9999,
        }
        for action, weights in action_weights.items():
            if len(weights) > 0:
                decision_weights[action] = sum(weights)

        max_weight = max(decision_weights.values())
        decision = random.choice([d for d in decision_weights.keys() if decision_weights[d] == max_weight])
        return decision
            
