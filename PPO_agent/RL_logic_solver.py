# THis module copies code first developed in fake_traces in order to create automatic solutions to games
from game import *
from RL_helper import *

# checks if you will intersect the gold by moving forward only
# does not check walls or anything
def will_intersect_forward(G):
    gx, gy = G.settings.gold[0]
    ax, ay = G.settings.agent_x, G.settings.agent_y
    # turns the gold into the agent's field of reference; 
    # then, check if it's in front of you and within the line you'll sweep while moving forward
    rel_gx, rel_gy = G.backRot(gx-ax, gy-ay, G.settings.direction)
    return (rel_gx > 0) and (abs(rel_gy) < G.settings.agent_r)

import math
tau = 2*math.pi

# True if the shortest path is turning clockwise (for a forward trajectory)
# False if you should turn counterclockwise instead
# turning anticlockwise increases 'direction' value
# turning clockwise decreases 'direction' value
def should_turn_anticlockwise_forward(G):
    gx, gy = G.settings.gold[0]
    ax, ay = G.settings.agent_x, G.settings.agent_y
    theta = G.settings.direction
    theta_to_gold = G.direction_angle(ax, ay, gx, gy)
    cw_theta = (theta - theta_to_gold) % tau
    #print(cw_theta)
    acw_theta = (theta_to_gold - theta) % tau
    #print(acw_theta)
    return acw_theta < cw_theta

# best move right now, bare settings
# can use this, with game, to create track to gold (moving only forward)
def best_move_forward(G):
    if will_intersect_forward(G):
        return 1 # action 1, G.stepForward
    else:
        if should_turn_anticlockwise_forward(G):
            return 4 # action 4, G.swivel_anticlock
        else:
            return 3 # action 3, G.swivel_clock

from copy import deepcopy

def _trace_forward(settings, maxlen=1024, zeroPad=False, ret_rewards = False):
    G = discreteGame(deepcopy(settings))
    reward = 0
    steps = 0
    trace = [] # can also be numpy array or something; just getting something out there
    if ret_rewards:
        rewards = [0] # receive reward in next timestep
    while reward < 1e-4 and steps < maxlen - 1:
        action = best_move_forward(G)
        trace.append(action)
        res = G.actions[action]()
        reward += res
        if ret_rewards:
            rewards.append(res)
        steps += 1
    if zeroPad:
        if len(trace) < maxlen:
            trace.append(2)
            trace = trace + [0 for i in range(maxlen - len(trace))]
            if ret_rewards:
                rewards = rewards + [0 for i in range(maxlen - len(rewards))]
    if ret_rewards:
        return trace, rewards
    else:
        return trace

# best move right now, bare settings, both forward and backward
# can use this, with game, to create track to gold (either forward or backward)
def best_move(G):
    gx, gy = G.settings.gold[0]
    ax, ay = G.settings.agent_x, G.settings.agent_y
    # turns the gold into the agent's field of reference; 
    # then, check if it's in front of you and within the line you'll sweep while moving forward
    rel_gx, rel_gy = G.backRot(gx-ax, gy-ay, G.settings.direction)

    #print(rel_gx)
    #print(rel_gy)
    #print(G.settings.agent_r)
    if abs(rel_gy) < G.settings.agent_r:
        if rel_gx > 0:
            return 1
        else:
            return 108 # becomes '2' when decoded; this is the symbol used by the agent
    
    # if not, we have to figure out the correct angle
    theta = G.settings.direction
    back_theta = (theta + math.pi) % tau
    theta_to_gold = G.direction_angle(ax, ay, gx, gy)

    cw_theta = (theta - theta_to_gold) % tau
    cwb_theta = (back_theta - theta_to_gold) % tau
    #print(cw_theta)
    acw_theta = (theta_to_gold - theta) % tau
    acwb_theta = (theta_to_gold - back_theta) % tau

    #print(np.array([cw_theta, cwb_theta, acw_theta, acwb_theta]))
    ind = np.argmin(np.array([cw_theta, cwb_theta, acw_theta, acwb_theta]))
    #print(ind)
    if ind < 2: # cw_theta or cwb_theta
        return 3
    else:
        return 4

def _trace_any(settings, maxlen=1024, zeroPad=False, ret_rewards = False):
    G = discreteGame(deepcopy(settings))
    reward = 0
    steps = 0
    trace = [] # can also be numpy array or something; just getting something out there
    if ret_rewards:
        rewards = [0] # receive reward in next time step
    while reward < 1e-4 and steps < maxlen - 1:
        action = best_move(G)
        trace.append(action)
        res = G.actions[symbol_action_map[action]]()
        reward += res
        if ret_rewards:
            rewards.append(res)
        #print(reward)
        steps += 1
    if zeroPad:
        if len(trace) < maxlen:
            trace.append(2)
            trace = trace + [0 for i in range(maxlen - len(trace))]
            rewards = rewards + [0 for i in range(maxlen - len(rewards))]
    if ret_rewards:
        return trace, rewards
    else:
        return trace

# Got tired of writing two versions of consumer functions, so here's this wrapper
def get_trace(settings, maxlen=1024, zeroPad=False, ret_rewards = False, forward_only = True):
    if forward_only:
        return _trace_forward(settings, maxlen, zeroPad, ret_rewards)
    else:
        return _trace_any(settings, maxlen, zeroPad, ret_rewards)


# Fills in traces, settings, and also rewards, returns, values, and gae's
def fake_data_fill(destination_buffer, parent_game, batch_size, maxlen=32, forward_only=True, device='cpu'):
    games = [discreteGame(parent_game.random_bare_settings(gameSize=224, max_agent_offset=0.5)) for i in range(batch_size)]
    traces = []
    rewards = []
    for i in range(batch_size):
        trace, reward = get_trace(games[i].settings, maxlen=maxlen-1, zeroPad=True, ret_rewards=True, forward_only=forward_only)
        traces.append([0] + trace)
        rewards.append([0] + reward)
    destination_buffer.settings_buffer = [[G.settings] for G in games]
    #destination_buffer.rewards = torch.zeros(device
    for b in range(batch_size):
        G = games[b]
        for val in traces[b]:
            if val in special_symbols:
                G.actions[symbol_action_map[val]] # no need to store rewards, already stored
            destination_buffer.settings_buffer[b].append(deepcopy(G.settings))
    destination_buffer.traces = torch.tensor(traces).to(device)
    destination_buffer.rewards = torch.tensor(rewards).to(device)

    destination_buffer.seed_offset = 1
    destination_buffer.logpas = torch.zeros((batch_size, maxlen-1), device=device) # this is fine; assume the guide is 100 % confident
    destination_buffer.entropies = torch.zeros((batch_size, maxlen-1), device=device) # Also fine

    destination_buffer.terminated, destination_buffer.past_terminated = destination_buffer.get_terminations()
    destination_buffer.values = destination_buffer.get_values()
    destination_buffer.returns = destination_buffer.get_returns() # rewards computed already, from the gold
    destination_buffer.gaes = destination_buffer.get_gaes() # automatically stores in destination_buffer.gaes

    return None
