# This is a script version of the python notebook
# THis is what I will use while I am away, so that I do not have to maintain something with GUI.

from game import *

game_settings = BIG_tool_use_advanced_2_5
game_settings.gameSize = 224 # for compatibility with brain's expected size
G = discreteGame(game_settings) # kind of a waste; will only call this object to generate random versions of itself

from RL_helper import *

from visual_transformer import *
device = torch.device('cuda:1') # doing this on the P40

symbol_action_map = {1: 1, 2: 2, 3: 3, 4: 4} # changing this to the 5-output version

# Next, I need to set up the brain itself
old_brain = DefaultAgentBrain()
old_brain.load_state_dict(torch.load('brain_checkpoints/brain_weights_tutorial1_v3_batch95000.pth', weights_only=True, map_location='cpu'))

brain = DefaultAgentBrain(5)
brain.img_enc = old_brain.img_enc
brain.img_dec = old_brain.img_dec
brain = brain.to(device)

from RL_logic_solver import * # convenient way to get fake, logical traces

def average_return(bb):
    """The average return (at the end of the seeds alone) from a buffer-buffer"""
    s = torch.zeros(bb[0].returns[:, 0].size(), device = bb[0].returns[:, 0].device)
    for b in bb:
        #s += bb[0].returns[:, 0]
        s += b.returns[:, 0]
    return torch.sum(s).item()/(len(bb) * bb[0].returns.size()[0])

def get_bb(num_buffers=64, batch_size=1, start_with_guide=True):
    bb = []
    brain.eval()
    for i in range(num_buffers):
        #print(i)
        # In this case, we are only training the 'dopamine' layer on the val training loop
        buff = GameOutputBuffer(brain, brain.evaluate_text, gamma=0.99, tau=0.97, default_batch_size=batch_size)
        if start_with_guide and (i == 0):
            fake_data_fill(buff, G, batch_size, device=device) # This one guide is the only difference between this and the main one.
        else:
            buff.fill(G, num_games=batch_size)
        buff.cpu()
        bb.append(buff)
        #print(buff.traces)
    score = average_return(bb)
    score_tensor = torch.tensor([score], device=device)
    for i in range(num_buffers):
        buff = bb[i]
        buff.to(device)
        buff.use_values(score_tensor) # computes values tensor and gae's
        buff.cpu()
    return bb

mse_loss = nn.MSELoss()

def nvidia_smi_spoof(device=device):
    return torch.cuda.memory_allocated() / (1024 ** 3)

policy_optimizer = optim.Adam(list(brain.text_enc.parameters()) + list(brain.text_dec.parameters()), lr=0.00001, eps=1e-9)
policy_epochs = 4
epochs = policy_epochs

def train_policy(policy_optimizer, epochs, buffer_buffer, policy_clip_range=0.1, entropy_loss_weight=0.01):
    for epoch in range(epochs):
        print(f"==========Epoch {epoch}=====================")
        #print(nvidia_smi_spoof())
        brain.train()
        #print(nvidia_smi_spoof())
        train_loss = 0
        random.shuffle(buffer_buffer)
        i = 0
        for buffer in buffer_buffer:
            i += 1
            buffer.to(device)
            #print(nvidia_smi_spoof())
            policy_optimizer.zero_grad()
            #print(nvidia_smi_spoof())
            logpas, entropies = buffer.get_probabilities_and_entropies(evaluation=False)
            #print(nvidia_smi_spoof())
            ratios = (logpas - buffer.logpas).exp()
            #print(nvidia_smi_spoof())
            pi_obj = buffer.gaes * ratios
            #print(nvidia_smi_spoof())
            pi_obj_clipped = buffer.gaes * ratios.clamp(1.0 - policy_clip_range,
                                                       1.0 + policy_clip_range)
            #print(nvidia_smi_spoof())
            policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean()
            #print(nvidia_smi_spoof())
            entropy_loss = -entropies.mean() * entropy_loss_weight
            #print(nvidia_smi_spoof())
            loss = policy_loss + entropy_loss
            #print(nvidia_smi_spoof())
            loss.backward()
            #print(nvidia_smi_spoof())
            policy_optimizer.step()
            #print(nvidia_smi_spoof())
            train_loss += loss.item()
            #print(nvidia_smi_spoof())
            buffer.cpu()
            #print(nvidia_smi_spoof())
            torch.cuda.empty_cache()
            print(f"episode {i}, policy loss {loss.item()}\n")
        del loss, logpas, entropies, ratios, pi_obj, pi_obj_clipped, policy_loss, entropy_loss
        policy_optimizer.zero_grad()
        #print(nvidia_smi_spoof())
        print(f"Policy train loss in epoch {epoch}:{train_loss / (len(buffer_buffer))}")
#train_policy(policy_optimizer, policy_epochs, buffer_buffer, policy_clip_range, entropy_loss_weight)

def run_round(round_num, policy_optimizer, num_buffers=64, batch_size=6, policy_epochs=4, policy_clip_range=0.5, entropy_loss_weight=1e-3):
    # First, get some samples
    brain.eval()
#    get_value.eval()
    buffer_buffer = get_bb(num_buffers, batch_size) # run the inference side
    print(f"Return before training was {average_return(buffer_buffer)}")
    #if round_num > 0:
    print("\n~~~~~~~POLICY loop~~~~~~~\n")
    train_policy(policy_optimizer, policy_epochs, buffer_buffer, policy_clip_range, entropy_loss_weight)
    #print("\n~~~~~~~~VALUE loop~~~~~~~~~~~\n")
    #train_val_func(val_optimizer, val_epochs, buffer_buffer)
    del buffer_buffer

import time

policy_epochs=4
val_epochs=16
num_buffers=8 # keep it simpler
batch_size=32#16
num_rounds = 150*10*10 # give it more of a chance to learn policy, which can only change a little over each round.
policy_clip_range=0.5
entropy_loss_weight=5e-3
for i in range(num_rounds):
    start = time.time()
    print(f"**********************ROUND {i} ***************************\n")
    run_round(i, policy_optimizer, num_buffers, batch_size, policy_epochs, policy_clip_range, entropy_loss_weight)
    torch.save(brain.state_dict(), f'brain_checkpoints/brain_EXPERIMENTAL_5output_weights_semi-guided_RL_GRPO_v2_round{i + 1}.pth')
    elapsed = time.time() - start
    print(f"***********************TIME WAS {elapsed / 60} min*****************************\n")
    # I think the entropy was too low last time, let's see if this fixes the issue.
    if i > 40:
        entropy_loss_weight = max(entropy_loss_weight / 2, 1e-4)#1e-3)
