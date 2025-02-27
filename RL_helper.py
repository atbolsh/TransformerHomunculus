# Based on PPO helper
# uses game mechanics; stores game settings at each step.

# Basic object: list of games
# 2nd common object: list of settings, batchsize X timestamp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.distributions import Categorical

import math
import copy
from copy import deepcopy # only for game settings

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset

from game import *
from visual_transformer import *

def get_images_games(game_batch, device='cuda'):
    batch_size = len(game_batch)
    img = torch.zeros(batch_size, 224, 224, 3)
    for i in range(batch_size):
        img[i] = torch.tensor(game_batch[i].getData())
    img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(device)
    return img

def get_images_settings(settings_batch, device='cpu'):
    batch_size = len(settings_batch)
    img = torch.zeros(batch_size, 224, 224, 3)
    for i in range(batch_size):
        G2 = discreteGame(settings_batch[i])
        img[i] = torch.tensor(G2.getData())
    img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(device)
    return img

# Read from a file? Better names?
special_symbols = set([1, 3, 4, 108])
symbol_action_map = { 1:1, 3:3, 4:4, 108:2}

def move_all(game_batch, action_batch, device='cpu'):
    # Let's prevent multiple GPU loads
    arr = action_batch.cpu().numpy()
    rewards = torch.zeros(len(game_batch), device='cpu', dtype=torch.int64)
    for i in range(len(game_batch)):
        symbol = arr[i]
        if symbol in special_symbols:
            rewards[i] = game_batch[i].actions[symbol_action_map[symbol]]() # rewards explicitly returned; game batches update internal state.
    return rewards.to(device)

# Gets responses from brain but does not call move_all; that will be part of the buffer object.
def extend_all(game_batch, traces, is_terminated, brain, temp=1.0):
    device = brain.get_device()
    imgs = get_images_games(game_batch)
    # assume traces and is_terminated are already on device
    traces, preds, log_probs, entropy, is_terminated = brain.extend(traces, is_terminated, context=imgs, temp=temp)
    return traces, preds, log_probs, entropy, is_terminated

# That's enough general things; let's get into the buffer class itself

class GameOutputsBuffer:
    def __init__(self, policy_model, value_model, gamma, tau, reward_func, default_batches=16):

        self.seed_offset = 0

        self.tau = tau
        self.gamma = gamma
        self.default_batches = default_batches
        self.contexts = None
        self.logpas = None # later will be the log-probs of actions taken
        self.traces = None # later will be the actual states / actions
        self.settings_buffer = None # later will be nested list of game Settings objects
        self.games = None # later will be the buffer of the active games

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

    def get_device(self):
        return self.policy_model.get_device()

    def generate(self, parent_game, batches=None, traces=None, end_on_gold=True, maxlen = None, temp=1.0, default_batches = 1):
        device = self.get_device()
        if batches is None:
            if traces is None:
                batches = default_batches
            else:
                batches, _ = traces.size()

        if maxlen is None:
            maxlen = self.policy_model.text_enc.sequence_length # Find better way? Maybe so you can call it only on part of what's going on?

        if traces is None:
            traces = torch.zeros((batches, 1), device=device, dtype=torch.long)
        else:
            if traces.size()[0] != batches:
                raise ValueError("Error: trace seeds provided do not match provided batch size")
        self.traces = traces

        self.games = [discreteEngine(parent_game.random_bare_settings(gameSize=224, max_agent_offset=0.5)) for i in range(batch_size)]
        self.settings_buffer = [[deepcopy(G.settings)] for G in self.games]

        self.logpas = torch.zeros((batches, 1), device=device) # default dtype
        self.entropies = torch.zeros((batches, 1), device=device)
        self.rewards = torch.zeros((batches, 1), device=device)

        is_terminated = torch.zeros(batches, dtype=torch.bool, device=device) # none are terminated initially
        trace_fixup = torch.zeros(batches, dtype=torch.bool, device=device) # hacky way to make sure the sequence ends with a '2', or </s> token

        while (self.traces.size()[1] < maxlen) and (not torch.all(is_terminated)):
            self.traces, actions, newlp, newent, is_terminated = extend_all(self.games, self.traces, is_terminated, self.policy_model, temp) # call policy model
            #hacky way to propagate 'ended last move due to gold':
            if torch.any(trace_fixup):
                actions = actions * torch.logical_not(trace_fixup) + (2 * trace_fixup)
                self.trace[:, -1] = actions # replace
                trace_fixup = torch.zeros(batches, dtype=torch.bool, device=device) # reset the trace fixup flags

            newrewards = move_all(self.games, actions, device=device) # move the actual env variables

            if end_on_gold:
                trace_fixup = (newrewards > 1e-4) # finished this move due to gold; put in a '2' next stage
                is_terminated = torch.logical_and(is_terminated, trace_fixup)

            if firstGone: # so, in all cases except the first value
                self.logpas = F.pad(self.logpas, (0, 1))
                self.entropies = F.pad(self.entropies, (0, 1))
                self.rewards = F.pad(self.rewards, (0, 1))
            else:
                firstGone=True
            self.logpas[:, -1] += newlp
            self.entropies[:, -1] += newent
            self.rewards[:, -1] += newrewards
            for i in range(batches):
                self.settings_buffer[i].append(deepcopy(self.games[i].settings))
        return None

    def fill(self, parent_game, num_games, seeds=None):
#        print(seeds)
        if seeds is not None:
            self.seed_offset = seeds.size()[1]
        with torch.no_grad():
            self.generate(parent_game, batches=num_games, traces=seeds)
#            print(self.traces)
#            print(self.logpas)
#            print(self.entropies)
            # compute termination points
            self.terminated, self.past_terminated = self.get_terminations()
            # compute values
            self.values = self.get_values()
            # call reward func and compute returns
            self.returns = self.get_returns() # rewards computed already, from the gold
            # use the values and rewards to compute GAE
            self.gaes = self.get_gaes() # automatically stores in self.gaes

    def _get_terminations(self):
        batches, actions = self.logpas.size()
        terminated = torch.zeros(batches, actions + 1, dtype=torch.bool, device=self.logpas.device)
        # For completeness: maybe add code to compute termination in seed itself?
        terminated[:, 1:] +=  (self.traces[:, self.seed_offset:] == 2) # mark the termination tokens; can't be terminated before first action, not in my setup
        past_terminated = torch.zeros(batches, actions + 1, dtype=torch.bool, device=self.logpas.device)

        # propagate the terminations down to the end
        for i in range(actions + 1): # include initial state, too
           if i > 0:
               terminated[:, i] = torch.logical_or(terminated[:, i], terminated[:, i - 1])
               past_terminated[:, i] = terminated[:, i - 1]
        return terminated, past_terminated 

    def _get_values(self):
        batches, actions = self.logpas.size()
        values = torch.zeros(batches, actions + 1, device = self.logpas.device)
        for i in range(actions + 1): # include initial state, too
            # get imgs tensor:
            imgs = get_images_settings(self.settings_buffer[:, i], device=self.get_device())
            # get value from value-func; zero-out past the end of the episode
            values[:, i] = self.value_model(self.traces[:, :(self.seed_offset + i - 1)], imgs).squeeze() * torch.logical_not(self.past_terminated[:, i]) # that's really the value of the action take in state[i - 1]
        return values

    def _get_returns(self):
        batches, actions = self.logpas.size()
        # If I want to add extra rewards later, I can
        if self.reward_func is not None:
            self.rewards += self.reward_func(self.traces, self.seed_offset, self.past_terminated, self.settings_buffer) # should be batches, actions + 1
        returns = torch.zeros(batches, actions + 1, device=self.traces.device)
        returns[:, -1] = rewards[:, -1]
        for i in range(actions):
            returns[:, -1-i-1] = self.rewards[:, -1-i-1] + self.gamma * returns[:, -1-i]
        return returns

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

    def get_returns(self, evaluation = True):
        if evaluation:
            with torch.no_grad():
                return self._get_returns()
        else:
            return self._get_returns() 

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
 









