# Given a picture with a blue line, answer whether you are facing it or not.
# THis is not for generating blue lines; that's tutorial1. More of that sort will come, later

from general_framework import *
from game_logic_solver import *

prompts_goldProximity = [ \
    "Are you near the gold?", \
    "Does your direction line up with the gold?", \
    "Are so close to the gold you're salivating?", \
    "Is the meal right in front of you?", \
    "Are you almost at the reward?", \
    "The coin's right there, yes?", \
]

Yreplies_goldProximity = [ \
    "Yep", \
    "Absolutely.", \
    "Certainly", \
    "I think so.", \
    "Uh-huh.", \
    "Sure" \
]

Nreplies_goldProximity = [ \
    "Nuh-uh", \
    "No", \
    "I don't think so.", \
    "Certainly not", \
    "Absolutely not", \
    "Nah" \
]

prompts_goldProximity_tensor = tensorify_list(prompts_goldProximity)
Yreplies_goldProximity_tensor = tensorify_list(Yreplies_goldProximity)
Nreplies_goldProximity_tensor = tensorify_list(Nreplies_goldProximity)

prompts_goldProximity_lens = get_lens(prompts_goldProximity_tensor)

def gold_is_near(s):
    return (((s.agent_x - s.gold[0][0])**2 + (s.agent_y - s.gold[0][1])**2) < 0.15*0.15)

def get_gold_proximity_data(batch_size):
    S = get_settings_batch(batch_size) 
    deciderFunc = lambda s: will_intersect_forward(discreteEngine(s)) # usually not, but often enough

    texts = text_generator_simple(S, \
                                  prompts_goldProximity_tensor, \
                                  Yreplies_goldProximity_tensor, \
                                  Nreplies_goldProximity_tensor, \
                                  prompts_goldProximity_lens, \
                                  gold_is_near, \
                                  device \
                                 )
    imgs = get_images(S)

    return imgs, texts

def _gold_proximity_batch(batch_size, model, optimizer=None, batch_num=0, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.eval()

    if training:
        model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    imgs, task_texts = get_gold_proximity_data(batch_size)
        
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

def gold_proximity_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _gold_proximity_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _gold_proximity_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)


