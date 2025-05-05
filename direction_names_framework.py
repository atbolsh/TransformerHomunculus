# This framework is for teaching the agent to use the action <forward> when prompted to, and also 
# correctly name the <forward> action with the word "forward". Same for "clock" and "anticlock"

from general_framework import *
from generalQA_framework import *

prompts_for_action_names = [ \
    "Please go forward.<forward>", \
    "Go forward:<forward>", \
    "Please make the forward move<forard>", \
    "Please progress<forward>"
    "Please turn clockwise <clock>", \
    "Could you turn clockwise?<clock>Sure!", \
    "Just take the CW move.<clock>", \
    "Take the CW move.<clock>", \
    "Please take the CW move.<clock>", \
    "Please turn counter-clockwise <anticlock>", \
    "Could you turn clockwise?<anticlock>Sure!", \
    "Just take the CCW move.<anticlock>", \
    "Take the CCW move.<anticlock>", \
    "Please take the CCW move.<anticlock>", \
    "What action is <forward>? That's a move forward", \
    "What action is <clock>? That's a CW turn", \
    "What action is <clock>? That's a clockwise turn", \
    "What action is <anticlock>? That's a CCW turn", \
    "What action is <anticlock>? That's a counter-clockwise turn", \
    "<forward> What action did you just take? Forward!", \
    "<clock> What action did you just take? Clockwise turn!", \
    "<anticlock> What aciton did you just take? Counterclockwise turn!", \
    "<forward> What action did you just take? Forward move", \
    "<clock> What action did you just take? Clockwise turn", \
    "<anticlock> What aciton did you just take? I turned counter-clockwise, sir", \
    "<forward> What action did you just take? Forward move", \
    "<clock> What action did you just take? I turned clockwise, sir", \
    "<anticlock> What aciton did you just take? Counter-clockwise turn", \
    "<forward> What was that?? Forward move.", \
    "<clock> What was that?? Clockwise turn", \
    "<anticlock> What was that?? Counter-clockwise turn."
]

prompts_for_action_names_tensor = tensorify_list(prompts_for_action_names)

def _direction_names_batch(batch_size, model, optimizer=None, batch_num=0, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.eval()

    if training:
        model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")
        
    S = get_settings_batch(batch_size)
    imgs = get_images(S)
    
    texts_direction_names = simple_sample(batch_size, prompts_for_action_names_tensor, device=device) 

    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = sdt[ind:ind + batch_size].to(device)

    all_texts = [control_texts, texts_direction_names]
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
        print(f"Total loss: {loss.item()}:\n{all_losses[0].item()} control,\n{all_losses[1].item()} direction naming\n\n")

    if reset_model and (type(model) is EnhancedAgentBrain):
        model.reset()

    return (loss.item(), all_losses[0].item(), all_losses[1].item())

def direction_names_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _direction_names_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _direction_names_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)


