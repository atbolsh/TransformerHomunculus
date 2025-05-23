# THis is basically a copy of pseudoGRPO/RL_logic_solver
# Mostly will be used to train the agent when it's facing the gold and when it isn't
# I know it's kinda hacky to have this code in two places, but I am certainly reorganizing this repo into a new one within a month or two.

from game import *

def gold_direction_angle(G):
    gx, gy = G.settings.gold[0]
    ax, ay = G.settings.agent_x, G.settings.agent_y
    return G.direction_angle(ax, ay, gx, gy)

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

# debug this. Should work, but still
def true_angle_difference_magnitude(alpha, beta):
    theta1 = (alpha - beta) % tau
    theta2 = (beta - alpha) % tau
    return min(theta1, theta2)

# I pulled out the meat of the computation below so that 'should turn anticlockwise' can be used in arbitrary situations, 
# not just for gold.
def _should_turn_anticlockwise_forward_ENGINE(current_theta, target_theta):
    cw_theta = (current_theta - target_theta) % tau
    #print(cw_theta)
    acw_theta = (target_theta - current_theta) % tau
    #print(acw_theta)
    return acw_theta < cw_theta

# True if the shortest path is turning clockwise (for a forward trajectory)
# False if you should turn counterclockwise instead
# turning anticlockwise increases 'direction' value
# turning clockwise decreases 'direction' value
def should_turn_anticlockwise_forward(G):
    gx, gy = G.settings.gold[0]
    ax, ay = G.settings.agent_x, G.settings.agent_y
    theta = G.settings.direction
    theta_to_gold = G.direction_angle(ax, ay, gx, gy)
    return _should_turn_anticlockwise_forward_ENGINE(theta, theta_to_gold

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



