from general_framework import *
from image_to_settings import *
from copy import deepcopy

# first framework that extracts information from the recon image. Training for 'imagine being closer to the gold' and other questions.
# RAW. Relative losses within CL need to be calibrated

prompts_move_agent_closer = [ \
    "Can you imagine if the agent is closer to the gold?", \
    "Imagine the agent closer to the gold.", \
    "Would this be easier if the agent were closer to the gold? Imagine it.", \
    "Please imagine the agent somewhere closer to the gold."\
]

complex_loss_text_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(prompts_move_agent_closer)]).contiguous().to(device)

def complex_loss_text_sample(num_sample=40):
    num_texts = complex_loss_text_tensor.size()[0]
    text_inds = torch.randint(0, num_texts, size=(num_sample,), device=device)
    texts = complex_loss_text_tensor[text_inds]
    return texts

# settings batch was the input; agent_centers, directions, radii, single gold, etc, are all detected from output image
# This produces an image batch as if the agent centers, directions, and radii were legit; this let's us 
# make sure that the output still looks like a typical picture from the game.
def gamify_output(inp_settings_batch, agent_centers, directions, agent_radii, gold_centers, gold_radii, agent_filters, gold_filters):
    N = len(inp_settings_batch)
    # no gradients when creating the output tensor; the loss will be between this and the images the model had produced
    with torch.no_grad():
        out_settings_batch = []
        for i in range(N):
            setting = deepcopy(inp_settings_batch[i]) # copy the wall placement and other such details

            if agent_filters[i]:
                setting.direction = directions[i].item()
                center = agent_centers[i].cpu().detach().numpy()
                setting.agent_x = center[0]
                setting.agent_y = center[1]
                setting.agent_r = agent_radii[i].item()

            if gold_filters[i]:
                gold_center = gold_centers[i].cpu().detach().numpy()
                setting.gold_centers = [[gold_center[0], gold_center[1]]]
                setting.gold_r = gold_radii[i].item()

            # else condition in both cases: keep the old placement. Punish for not having an agent / gold piece at all.

            out_settings_batch.append(setting)

    return get_images(out_settings_batch)

# as before, the output values are detected from the image
# we penalize moving the gold, we reward moving the agent up to twice as close as before, and 
# we penalize changing the radii at all, or wall collisions.
# TODO: add wall hack penalty; add penalty for changing how agent is facing too much; add penalty for veering off old-agent / old-gold line
def complex_loss_func(inp_settings_batch, agent_centers, directions, agent_radii, gold_centers, gold_radii, filters):
    N = len(inp_settings_batch)
    loss = torch.tensor(0.0).to(device)
    for i in range(N):
        if filters[i]:
            s = inp_settings_batch[i]
            G2 = discreteGame(s)
            # skipping the wall-overlap loss; haven't thought of how to write it, yet
            loss += 100.0 * (agent_radii[i] - s.agent_r)**2 # balance the weights later, based on experience
            #print(loss)
            loss += 100.0 * (gold_radii[i] - s.gold_r)**2 # make smaller to avoid nan's?? remove altogether?
            #print(loss)
    
            gold_x, gold_y = s.gold[0] # only one gold element is assumed
            loss += 10.0 * ((gold_centers[i, 0] - gold_x)**2 + (gold_centers[i, 1] - gold_y)**2) # make smaller/remove to avoid nan's??
            #print(loss)
    
            old_distance_squared = (gold_x - s.agent_x)**2 + (gold_y - s.agent_y)**2 # scalar
            new_distance_squared = (gold_x - agent_centers[i, 0])**2 + (gold_y - agent_centers[i, 0])**2 # tensor
            loss += torch.relu((4 * new_distance_squared) - old_distance_squared) # becomes positive if agent is too far away; 0 within a certain circle
            #print(loss)

    return loss # game-like-ness not computed as part of complex loss


def _complex_loss_batch(batch_size, model, optimizer=None, batch_num=0, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.eval()

    if training:
        model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")
        
    inp_S = get_settings_batch(batch_size)
    inp_imgs = get_images(inp_S)

    texts = complex_loss_text_sample(batch_size)

    task_probs, task_recon = model(texts, inp_imgs, ret_imgs=True)

    text_loss = get_text_loss(task_probs, texts) # might as well store it early

    # These are the differentiable functions that extract this information from the images
    # The real question is whether this will ever be 'learnable' enough to encourage proper
    # learning, or if I must switch to some semi-adversarial or semi-RL mechanism later. We will see.
    agent_centers, directions, agent_radii, agent_filters = get_agent_info(task_recon) 
    gold_centers, gold_radii, gold_filters = get_SINGLE_gold_info(task_recon, return_radii = True)

    filters = torch.logical_and(agent_filters, gold_filters)
    #print(filters)

    # no gradient is computed creating these; these are targets, to make sure that the output images actually look like games
    target_imgs = gamify_output(inp_S, agent_centers, directions, agent_radii, gold_centers, gold_radii, agent_filters, gold_filters)

    gameiness_loss = img_criterion(target_imgs, task_recon)

    CL = complex_loss_func(inp_S, agent_centers, directions, agent_radii, gold_centers, gold_radii, filters) 

    loss = (CL / 500) + gameiness_loss + (text_loss / 5000) # TODO: the weights of CL and gameiness_loss need to be balanced 

    if training:
        loss.backward()#retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        if type(model) is EnhancedAgentBrain:
            model.soft_reset()
            
    if printing:
        print(f"Total loss: {loss.item()}; that's {CL.item()} task and {gameiness_loss.item()} 'gameiness' and {text_loss.item()} total text\n\n")

    if reset_model and (type(model) is EnhancedAgentBrain):
        model.reset()

    return loss.item(), CL.item(), gameiness_loss.item(), text_loss.item()


def complex_loss_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _complex_loss_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _complex_loss_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)

