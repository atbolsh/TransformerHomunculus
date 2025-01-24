import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.distributions import Categorical

import math
import copy

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset

# THis si only for text output
# Image output will need a more elaborate 'generate' function
# I will get to that later.

# Quick guide: self.traces includes both the seeds and all actions taken
# terminated, past_terminated, values, rewards, returns, all are actions + 1 (include eval for seed alone)
# logpas, entropy, and gaes are all length 'actions' alone; no extra term
class SentenceOutputSingleEpisode:
    def __init__(self, policy_model, value_model, gamma, tau, reward_func, default_batches=16):
        # I'm combining states_mem and actions_mem, because the state in this case is the previous actions

        self.seed_offset = 0 # will be set to be the offset where machine output starts
        self.tau = tau
        self.gamma = gamma
        self.default_batches = default_batches
        self.contexts = None
        self.logpas = None # later will be the log-probs of actions taken
        self.traces = None # later will be the actual states / actions
        self.entropies = None
        self.gaes = None
        self.terminated = None
        self.past_terminated = None # kinda lame to duplicate memory, but makes computations easier
        self.values = None
        self.rewards = None
        self.returns = None # merely the target for the value function, later

        self.policy_model = policy_model
        self.value_model = value_model
        # accepts traces, seed offset, past_terminated bools, and contexts in that order
        # must return 0s for all the past-terminated states
        self.reward_func = reward_func # by default, only called once per episode

        self.discounts = torch.logspace(0, 33, 33, base=gamma)
        self.tau_discounts = torch.logspace(0, 33, 33, base=gamma*tau)

        # policy model and value model are treated as external entities and don't change device with this object
        # all the tensors for the episodes do change, however.
        self.tensor_keys = ['contexts', 'logpas', 'traces', 'entropies', 'gaes', 'terminated', 'past_terminated', 'values', 'rewards', 'returns', 'discounts', 'tau_discounts']

    def fill(self, seeds, contexts=None):
#        print(seeds)
        self.seed_offset = seeds.size()[1]
        self.contexts = contexts
        with torch.no_grad():
            self.traces, self.logpas, self.entropies = self.policy_model.generate(seeds, contexts) # and other terms later, figure it out.
#            print(self.traces)
#            print(self.logpas)
#            print(self.entropies)
            # compute termination points
            self.terminated, self.past_terminated = self.get_terminations()
            # compute values
            self.values = self.get_values()
            # call reward func and compute returns
            self.rewards, self.returns = self.get_rewards_and_returns()
            # use the values and rewards to compute GAE
            self.gaes = self.get_gaes() # automatically stores in self.gaes

    def _get_terminations(self):
        batches, actions = self.logpas.size()
        terminated = torch.zeros(batches, actions + 1, dtype=torch.bool, device=self.logpas.device)
        past_terminated = torch.zeros(batches, actions + 1, dtype=torch.bool, device=self.logpas.device)
        for i in range(actions + 1): # include initial state, too
#           print(i)
           terminated[:, i] = (self.traces[:, self.seed_offset + i - 1] == 2) # include evaluation for seed alone
           if i > 0:
               terminated[:, i] = torch.logical_or(terminated[:, i], terminated[:, i - 1])
               past_terminated[:, i] = terminated[:, i - 1]
        return terminated, past_terminated 

    def _get_values(self):
        batches, actions = self.logpas.size()
        values = torch.zeros(batches, actions + 1, device = self.logpas.device)
        for i in range(actions + 1): # include initial state, too
            # get value from value-func; zero-out past the end of the episode
            values[:, i] = self.value_model(self.traces[:, :(self.seed_offset + i - 1)], self.contexts).squeeze() * torch.logical_not(self.past_terminated[:, i]) # that's really the value of the action take in state[i - 1]
        return values

    def _get_rewards_and_returns(self):
        batches, actions = self.logpas.size()
        rewards = self.reward_func(self.traces, self.seed_offset, self.past_terminated, self.contexts) # should be batches, actions + 1
        returns = torch.zeros(batches, actions + 1, device=self.traces.device)
        returns[:, -1] = rewards[:, -1]
        for i in range(actions):
            returns[:, -1-i-1] = rewards[:, -1-i-1] + self.gamma * returns[:, -1-i]
        return rewards, returns

    def _get_gaes(self):
        T = self.rewards.size()[1] - 1
#        print(rewards.size())
#        print(values.size())
        deltas = self.rewards[:, 1:] + self.gamma*self.values[:, 1:] - self.values[:, :-1] # reward for the action (so skip the reward for the seed alone)
        gaes = torch.zeros(deltas.size(), device=deltas.device) # batches, actions; 1 fewer than rewards / values / terminated info
        gaes[:, -1] = deltas[:, -1] * torch.logical_not(self.past_terminated[:, -1])
        for i in range(T - 1):
            gaes[:, -1 - i - 1] = (deltas[:, -1 -i -1] + self.gamma*self.tau*gaes[:, -1 - i] ) * torch.logical_not(self.past_terminated[:, -1 -i -1])
        return gaes

    def get_values(self, evaluation = True):
        if evaluation:
            with torch.no_grad():
                return self._get_values()
        else:
            return self._get_values()

    def get_terminations(self, evaluation = True):
        if evaluation:
            with torch.no_grad():
                return self._get_terminations()
        else:
            return self._get_terminations() 

    def get_rewards_and_returns(self, evaluation = True):
        if evaluation:
            with torch.no_grad():
                return self._get_rewards_and_returns()
        else:
            return self._get_rewards_and_returns() 

    def get_gaes(self, evaluation = True):
        if evaluation:
            with torch.no_grad():
                return self._get_gaes()
        else:
            return self._get_gaes() 

    def to(self, device):
        for key in self.tensor_keys:
            if not (self.__dict__[key] is None):
                self.__dict__[key] = self.__dict__[key].to(device)
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def get_device(self):
        return self.discounts.device
                   


 
