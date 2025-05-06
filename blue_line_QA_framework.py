# Given a picture with a blue line, answer whether you are facing it or not.
# THis is not for generating blue lines; that's tutorial1. More of that sort will come, later

from general_framework import *
from game_logic_solver import *

prompts_blueLineDirection = [ \
    "Are you facing the blue line?", \
    "Does your direction line up with the blue line?", \
    "Are you facing where it's pointing?", \
    "Are you facing in the right direction?"
]

Yreplies_blueLineDirection = [ \
    "Yep", \
    "Absolutely.", \
    "Certainly", \
    "I think so.", \
    "Uh-huh.", \
    "Sure" \
]

Nreplies_blueLineDirection = [ \
    "Nuh-uh", \
    "No", \
    "I don't think so.", \
    "Certainly not", \
    "Absolutely not", \
    "Nah" \
]

prompts_blueLineDirection_tensor = tensorify_list(prompts_blueLineDirection)
Yreplies_blueLineDirection_tensor = tensorify_list(Yreplies_blueLineDirection)
Nreplies_blueLineDirection_tensor = tensorify_list(Nreplies_blueLineDirection)

prompts_blueLineDirection_lens = get_lens(prompts_blueLineDirection_tensor)

def get_arrow_near_agent_direction(agent_direction):
    return G.mod2pi((np.random.random()*math.pi/3) + agent_direction - (math.pi / 6)) # cone of arc from theta - pi / 6 to theta + pi / 6

def get_arrow_far_agent_direction(setting):
    return G.mod2pi(agent_directon + math.pi / 6 + ((5 * math.pi / 3)*np.random.random())) # sector from theta + pi / 6 to theta - pi / 6

def get_random_directions(settings_batch):
    batchsize = len(settings_batch)
    deciders = (np.random.random((batchsize,)) < 0.5) # True if supposed to benear real direction, False otherwise
    directions = []
    for i in range(batchsize):
        if deciders[i]:
            directions.append(get_arrow_near_agent_direction(settings_batch[i].direction))
        else:
            directions.append(get_arrow_far_agent_direction(settings_batch[i].direction))
    return directions

def get_blue_line_direction_data(batch_size):
    S = get_settings_batch(batch_size) 
    directions = get_random_directions(S)
    deciderDict = {}
    for i in range(batch_size):
        theta = true_angle_difference_magnitude(directions[i], S[i].dirction)
        same_direction = (theta < math.pi / 12) # very generous, 'rough' direction. 30 degree cone of arc
        deciderDict[S[i]] = same_direction

    # this hacks the text generator function from generalQA; this is better than copying that code, as strange as it is
    deciderFunc = lambda s: deciderDict[s]

    texts = text_generator_simple(S, \
                                  prompts_blueLineDirection_tensor, \
                                  Yreplies_blueLineDirection_tensor, \
                                  Nreplies_blueLineDirection_tensor, \
                                  prompts_blueLineDirection_lens, \
                                  deciderFunc, \
                                  device \
                                 )

    imgs = torch.zeros(num_sample, 224, 224, 3)
    for i in range(batch_size):
        G2 = discreteGame(S[i])
        G2.draw_arrow(extension = 1.0 + 3.0 * np.random.random(), direction = directions[i])
        imgs[i] = torch.tensor(G2.getData())
    imgs = torch.permute(imgs, (0, 3, 1, 2)).contiguous().to(device)
    return imgs, texts

def _blue_line_direction_batch(batch_size, model, optimizer=None, batch_num=0, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.eval()

    if training:
        model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    imgs, task_texts = get_blue_line_direction_data(batch_size)
        
    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = sdt[ind:ind + batch_size].to(device)

    all_texts = [control_texts, task_texts]
    text_inds = list(range(2))
    # new code. Shuffling helps enhanced brain not transfer information 
    # between tasks, despite not doing any resets
    if random_order:
        random.shuffle(text_inds)

    all_probs = [0 for t_ind in text_inds]
    # first to be computed needs to initialize context buffer
    if type(model) is EnhancedAgentBrain:
        all_probs[text_inds[0]] = (model(all_texts[text_inds[0]], imgs, ret_imgs=False, create_context=True))
    else:
        all_probs[text_inds[0]] = (model(all_texts[text_inds[0]], imgs, ret_imgs=False))
    for t_ind in text_inds[1:]:
        if type(model) is EnhancedAgentBrain:
            all_probs[t_ind] = (model(all_texts[t_ind], imgs, ret_imgs=False, create_context=False)) # should I rethink this term?
        else:
            all_probs[t_ind] = (model(all_texts[t_ind], imgs, ret_imgs=False))
    #now, all_probs, despite random execution order, has the corresponding output for every element of all_texts

    all_losses = [get_text_loss(all_probs[i], all_texts[i]) for i in range(2)]
    
    loss = sum(all_losses)

    if training:
        loss.backward()#retain_graph=True) # needed? consider.
        optimizer.step()
        optimizer.zero_grad()
        if type(model) is EnhancedAgentBrain:
            model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()}:\n{all_losses[0].item()} control,\n{all_losses[1].item()} recognizing the blue line direction\n\n")

    if reset_model and (type(model) is EnhancedAgentBrain):
        model.reset()

    return (loss.item(), all_losses[0].item(), all_losses[1].item())

def blue_line_direction_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _blue_line_direction_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _blue_line_direction_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)


