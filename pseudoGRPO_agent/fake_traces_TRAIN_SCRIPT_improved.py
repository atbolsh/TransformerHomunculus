# Just using supervised learning this time around

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

# by making nearby gold more likely, this version increases the probability the best move will be 1, which is otherwise low
# if this is not enough, I can always fast forward using trace generation. More on this later
def get_settings_batch(batch_size):
    return [G.random_bare_settings(gameSize=224, max_agent_offset=(0.15 + 0.4*i/batch_size)) for i in range(1, batch_size + 1)]

def get_best_moves(settings_batch, device=device):
    moves = []
    for settings in settings_batch:
        G = discreteGame(deepcopy(settings))
        moves.append(best_move_forward(G))
    return torch.tensor(moves, device=device).unsqueeze(1)

def get_images(settings_batch, device=device):
    batch_size = len(settings_batch)
    img = torch.zeros(batch_size, 224, 224, 3)
    for i in range(batch_size):
        G2 = discreteGame(settings_batch[i])
        img[i] = torch.tensor(G2.getData())
    img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(device)
    return img

ent_criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(list(brain.text_enc.parameters()) + list(brain.text_dec.parameters()), lr=0.00001, eps=1e-9)

batch_size = 16
num_batches = 150*1000*1000
loss_sum = 0.0

brain.train()
for b in range(num_batches):
    optimizer.zero_grad()
    S = get_settings_batch(batch_size)
    imgs = get_images(S)
    moves = get_best_moves(S)
    inp = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
    probs = brain(inp, imgs, ret_imgs=False)
    L = torch.sum(ent_criterion(probs, moves))
    L.backward()
    optimizer.step()
    loss_sum += L.item()
    if (b + 1) % 100 == 0:
        print(f"batch {b+1}:\t{loss_sum / 100}")
        loss_sum = 0.0
    if ((b + 1) in [1, 2, 3, 4, 5, 8, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]) or (((b + 1) > 1000) and ((b + 1) % 1000 == 0)):
        torch.save(brain.state_dict(), f'brain_checkpoints/brain_EXPERIMENTAL_5output_weights_fake-traces_improved_v1_round{b + 1}.pth')
 




