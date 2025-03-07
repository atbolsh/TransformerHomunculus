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
def extend_all(game_batch, traces, is_terminated, brain, temp=1.0, img_gradient=False):
    device = brain.get_device()
    imgs = get_images_games(game_batch, device=device)
    # assume traces and is_terminated are already on device
    #print(traces.device)
    #print(is_terminated.device)
    if img_gradient:
        context = brain.img_enc(imgs)
    else:
        with torch.no_grad():
            context = brain.img_enc(imgs)
    traces, preds, log_probs, entropy, is_terminated = brain.extend(traces, is_terminated, context=context, temp=temp)
    return traces, preds, log_probs, entropy, is_terminated

# Get's all the probabilites and entropies from an *existing* set of actions
# 0s all around if something is terminated
def probs_and_entropies_all(settings_batch, traces, past_terminated, brain, temp=1.0, img_gradient=False):
    device = brain.get_device()
    imgs = get_images_settings(settings_batch, device=device)
    if img_gradient:
        context = brain.img_enc(imgs)
    else:
        with torch.no_grad():
            context = brain.img_enc(imgs)
    logpas, ents = brain.compute_probabilities(traces, single=True, context=context)
#    print(f"logpas, ents size: {logpas.size()}, {ents.size()}")
    mask = torch.logical_not(past_terminated)
    return logpas*mask, ents*mask
    
# That's enough general things; let's get into the buffer class itself

class GameOutputBuffer:
    def __init__(self, policy_model, value_model, gamma, tau, reward_func=None, default_batch_size=16):

        self.seed_offset = 1 # used to be 0, but strings always start with <s> by design (token 0), even when generated from scratch

        self.tau = tau
        self.gamma = gamma
        self.default_batch_size = default_batch_size
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

        self.discounts = torch.logspace(0, 33, 33, base=gamma, device=self.policy_model.get_device())
        self.tau_discounts = torch.logspace(0, 33, 33, base=gamma*tau, device=self.policy_model.get_device())

        # policy model and value model are treated as external entities and don't change device with this object
        # all the tensors for the episodes do change, however.
        self.tensor_keys = ['contexts', 'logpas', 'traces', 'entropies', 'gaes', 'terminated', 'past_terminated', 'values', 'rewards', 'returns', 'discounts', 'tau_discounts']

    def get_device(self):
        return self.policy_model.get_device()

    def generate(self, parent_game, batch_size=None, traces=None, end_on_gold=True, maxlen = None, temp=1.0, default_batch_size = 1):
        device = self.get_device()
        #print(f"buff device {device}")
        if batch_size is None:
            if traces is None:
                batch_size = default_batch_size
            else:
                batch_size, _ = traces.size()

        if maxlen is None:
            maxlen = self.policy_model.text_enc.sequence_length # Find better way? Maybe so you can call it only on part of what's going on?

        if traces is None:
            traces = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
        else:
            if traces.size()[0] != batch_size:
                raise ValueError("Error: trace seeds provided do not match provided batch size")
        self.traces = traces

        self.games = [discreteGame(parent_game.random_bare_settings(gameSize=224, max_agent_offset=0.5)) for i in range(batch_size)]
        self.settings_buffer = [[deepcopy(G.settings)] for G in self.games]

        self.logpas = torch.zeros((batch_size, 1), device=device) # default dtype
        self.entropies = torch.zeros((batch_size, 1), device=device)
        self.rewards = torch.zeros((batch_size, 2), device=device) # rewards, by default, has shape actions + 1, just like traces. This let's us store an 'initial reward' from reward_func later, if we want
        firstGone = False # past the first output, we must pad the tensor every time

        is_terminated = torch.zeros(batch_size, dtype=torch.bool, device=device) # none are terminated initially
        trace_fixup = torch.zeros(batch_size, dtype=torch.bool, device=device) # hacky way to make sure the sequence ends with a '2', or </s> token

        while (self.traces.size()[1] < maxlen) and (not torch.all(is_terminated)):
#            print('is_terminated before')
#            print(is_terminated)
            self.traces, actions, newlp, newent, is_terminated = extend_all(self.games, self.traces, is_terminated, self.policy_model, temp) # call policy model
#            print('actions and is_terminated after')
#            print(actions)
#            print(is_terminated)
            #hacky way to propagate 'ended last move due to gold':
            if torch.any(trace_fixup):
                actions = actions * torch.logical_not(trace_fixup) + (2 * trace_fixup)
                self.traces[:, -1] = actions # replace
                trace_fixup = torch.zeros(batch_size, dtype=torch.bool, device=device) # reset the trace fixup flags

            newrewards = move_all(self.games, actions, device=device) # move the actual env variables

            if end_on_gold:
                trace_fixup = (newrewards > 1e-4) # finished this move due to gold; put in a '2' next stage
                is_terminated = torch.logical_or(is_terminated, trace_fixup)
#            print('trace fixup and final is_terminated')
#            print(trace_fixup)
#            print(is_terminated)
#            print('\n\n')

            if firstGone: # so, in all cases except the first value
                self.logpas = F.pad(self.logpas, (0, 1))
                self.entropies = F.pad(self.entropies, (0, 1))
                self.rewards = F.pad(self.rewards, (0, 1))
            else:
                firstGone=True
            self.logpas[:, -1] += newlp
            self.entropies[:, -1] += newent
            self.rewards[:, -1] += newrewards
            for i in range(batch_size):
                self.settings_buffer[i].append(deepcopy(self.games[i].settings))
        return None

    def fill(self, parent_game, num_games=None, seeds=None):
#        print(seeds)
        if num_games is None:
            if seeds is not None:
                num_games = seeds.size()[0]
            else:
                num_games = self.default_batch_size
        if seeds is not None:
            self.seed_offset = seeds.size()[1]
        with torch.no_grad():
            self.generate(parent_game, batch_size=num_games, traces=seeds)
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

    def _get_probabilities_and_entropies(self, policy_model=None, update_self=False, temp=1.0, batch_coords=None, action_coords=None, img_gradient=False):
        """Mostly used in the policy training loop, but can be used to update self.logpas and self.entropies. 
        Not called in self.fill or self.generate"""
        batch_size, actions = self.logpas.size()
        device=self.get_device()
        if batch_coords is None:
            batch_coords = (0, batch_size)
        if action_coords is None:
            action_coords = (0, actions)
        bS, bE = batch_coords # just a pain to write, otherwise
        batch_num = bE - bS
        aS, aE = action_coords
        action_num = aE - aS
        if policy_model is None:
            policy_model = self.policy_model
        logpas = torch.zeros((batch_num, action_num), device=device)
        entropies = torch.zeros((batch_num, action_num), device=device)
        for i in range(action_num):
#            print(i)
            ind = aS + i
            settings_ind = [settings_trace[ind] for settings_trace in self.settings_buffer[bS:bE]]
            past_terminated = self.past_terminated[bS:bE, 1 + ind] # we record action leading into '2', but no others
#            print(f"past_terminated size: {past_terminated.size()}")
            newlp, newent = probs_and_entropies_all(settings_ind, self.traces[bS:bE, :self.seed_offset + ind + 1], past_terminated, policy_model, temp, img_gradient)
            logpas[:, i] = newlp
            entropies[:, i] = newent
        if update_self:
            self.logpas[bS:bE, aS:aE] = logpas
            self.entropies[bS:bE, aS:aE] = entropies
        return logpas, entropies
        
    def _get_terminations(self):
        batch_size, actions = self.logpas.size()
        terminated = torch.zeros(batch_size, actions + 1, dtype=torch.bool, device=self.logpas.device)
        # For completeness: maybe add code to compute termination in seed itself?
        terminated[:, 1:] +=  (self.traces[:, self.seed_offset:] == 2) # mark the termination tokens; can't be terminated before first action, not in my setup
        past_terminated = torch.zeros(batch_size, actions + 1, dtype=torch.bool, device=self.logpas.device)

        # propagate the terminations down to the end
        for i in range(actions + 1): # include initial state, too
           if i > 0:
               terminated[:, i] = torch.logical_or(terminated[:, i], terminated[:, i - 1])
               past_terminated[:, i] = terminated[:, i - 1]
        return terminated, past_terminated 

    def _get_values(self, img_gradient=False, text_gradient=True):
        batch_size, actions = self.logpas.size()
        values = torch.zeros(batch_size, actions + 1, device = self.logpas.device)
        for i in range(actions + 1): # include initial state, too
            # get imgs tensor:
            settings_i = [settings_trace[i] for settings_trace in self.settings_buffer] # can't use proper tuples with lists, ugh
            imgs = get_images_settings(settings_i, device=self.get_device())
            # get value from value-func; zero-out past the end of the episode
            # the val of the action take in state[i - 1]
            values[:, i] = self.value_model(self.traces[:, :(self.seed_offset + i - 1)], imgs, img_gradient=img_gradient, text_gradient=text_gradient).squeeze() * torch.logical_not(self.past_terminated[:, i])
        return values

    def _get_returns(self):
        batch_size, actions = self.logpas.size()
        # If I want to add extra rewards later, I can
        if self.reward_func is not None:
            self.rewards += self.reward_func(self.traces, self.seed_offset, self.past_terminated, self.settings_buffer) # should be batch_size, actions + 1
        returns = torch.zeros(batch_size, actions + 1, device=self.traces.device)
        returns[:, -1] = self.rewards[:, -1]
        for i in range(actions):
            returns[:, -1-i-1] = self.rewards[:, -1-i-1] + self.gamma * returns[:, -1-i]
        return returns

    def _get_gaes(self):
        T = self.rewards.size()[1] - 1
#        print(rewards.size())
#        print(values.size())
        deltas = self.rewards[:, 1:] + self.gamma*self.values[:, 1:] - self.values[:, :-1] # reward for the action (so skip the reward for the seed alone)
        gaes = torch.zeros(deltas.size(), device=deltas.device) # batch_size, actions; 1 fewer than rewards / values / terminated info
        gaes[:, -1] = deltas[:, -1] * torch.logical_not(self.past_terminated[:, -1])
        for i in range(T - 1):
            gaes[:, -1 - i - 1] = (deltas[:, -1 -i -1] + self.gamma*self.tau*gaes[:, -1 - i] ) * torch.logical_not(self.past_terminated[:, -1 -i -1])
        return gaes

    # by default the errors *do* get propagated to the text encoder, but that can be changed to only train the dopamine function
    def get_values(self, evaluation = True, img_gradient = False, text_gradient = True):
        if evaluation:
            with torch.no_grad():
                return self._get_values()
        else:
            return self._get_values(img_gradient=img_gradient, text_gradient=text_gradient)

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

    # NOTE: this is the only function which defaults to evaluation=False, thanks to how it is normally used (in training loop)
    def get_probabilities_and_entropies(self, policy_model=None, update_self=False, temp=1.0, evaluation=False):
        if evaluation:
            with torch.no_grad():
                return self._get_probabilities_and_entropies(policy_model, update_self, temp)
        else:
            return self._get_probabilities_and_entropies(policy_model, update_self, temp)

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
 









