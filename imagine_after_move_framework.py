from general_framework import *
from game_logic_solver import *

# Given a prompt and descriptions of moves (NOT moves themselves), imagine what the scene will look like.
# Will be useful for planning

# 'turn one step' = 1 turn
# 'turn just a bit' = 5 turns
# 'turn a long time', 'several' = 10 turns
# 'other way' = 30 turns

prompts_imagineAfterMove = [ \
    "What will this game look if you go forward twice?", \
    "Imagine the scene if you just step forward", \
    "How would this look if you turn CW for a bit?", \
    "How would this look if you turn CCW for a bit?", \
    "Imagine this if you make a big turn clockwise", \
    "How would this look after several CCW turns?", \
    "Imagine this if you make a big turn counter-clockwise", \
    "How would this look after several CW turns?", \

    "How would this look if you turn CW one step?", \
    "How would this look if you turn CCW one step?", \
    "Imagine this if you make one step clockwise", \
    "How would this look after one step CCW?", \
    "Imagine this if you make one step counter-clockwise", \
    "How would this look after one CW step?", \

    "Woah, what would this look like if you step forward and then make a big turn CW?", \
    "How does this look after 3 forward steps?", \
    "Woah, imagine if you're looking the other way?", \
    "Picture stepping forward for me.", \
    "How do you figure this would look after turning CW just a bit?", \

    "Imagine going forward then turning around.", \
    "Imagine turning around then going forward", \
    "What does this look like if you take 2 steps forward then one step CCW?"
]

prompts_imagineAfterMove_tensor = tensorify_list(prompts_imagineAfterMove)

CW_small = ""
CCW_small = ""

for i in range(5):
    CW_small = CW_small + "<clock>"
    CCW_small = CCW_small + "<anticlock>"

CW_big = CW_small + CW_small
CCW_big = CCW_small + CCW_small

turn_around = CW_big + CW_big + CW_big

move_instructions_imagineAfterMove = [ \
    "<forward><forward>", \
    "<forward>", \
    CW_small, \
    CCW_small, \
    CW_big, \
    CCW_big, \
    CCW_big, \
    CW_big, \
    "<clock>", \
    "<anticlock>", \
    "<clock>", \
    "<anticlock>", \
    "<anticlock>", \
    "<clock>", \
    "<forward>" + CW_big, \
    "<forward><forward><forward>", \
    turn_around, \
    "<forward>", \
    CW_small, \
    "<forward>" + turn_around, \
    turn_around + "<forward>", \
    "<forward><forward><clock>" \
]
  
move_instructions_imagineAfterMove_tensor = tensorify_list(move_instructions_imagineAfterMOve, device='cpu') # only used by game objects, no reason to go onto device

def process_steps(game, instructions):
    valids = set([1, 3, 4])
    for action in instructions.numpy():
        if action in valids:
            game.actions[action]()
    return None
   

def imagineAfterMove_data(batch_size):
    S = get_settings_batch(batch_size)

    imgs_in = get_images(S)

    num_prompts = prompts_imagineAfterMove_tensor.size()[0]
    inds = torch.randint(0, num_prompts, (batch_size,))
    texts = prompts_imagineAfterMove_tensor[inds]
    
    instructions = move_instructions_imagineAfterMove_tensor[inds]
    imgs_out = torch.zeros(num_sample, 224, 224, 3)
    for i in range(batch_size):
        G2 = discreteGame(S[i])
        process_steps(G2, instructions[i])
        imgs_out[i] = torch.tensor(G2.getData())
    imgs_out = torch.permute(imgs_out, (0, 3, 1, 2)).contiguous().to(device)

    return texts, imgs_in, imgs_out

def _imagineAfterMove_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.eval()

    if training:
        model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    task_texts, inp, out = imagineAfterMove_data(batch_size)
    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = sdt[ind:ind + batch_size].to(device)

    flip = 0
    if random_order:
        flip += random.randint(0, 1)

    if flip:
        # do not need to set create_context; it's always true here, since we're return images
        task_probs, task_recon = model(task_texts, inp, ret_imgs=True)
        control_probs, control_recon = model(control_texts, inp, ret_imgs=True)
    else:
        control_probs, control_recon = model(control_texts, inp, ret_imgs=True)
        task_probs, task_recon = model(task_texts, inp, ret_imgs=True)

    l1 = img_criterion(task_recon, out)
    l2 = img_criterion(control_recon, inp)
    img_loss = l1 + l2 # may add coefficients later; in retrospect, no need, l1 is always much bigger anyway.
    tl1 = get_text_loss(task_probs, task_texts)
    tl2 = get_text_loss(control_probs, control_texts)
    text_loss = tl1 + tl2
    loss = img_loss + (text_loss / 5000)

    if training:
        loss.backward()#retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        if type(model) is EnhancedAgentBrain:
            model.soft_reset()
    
    if printing:
        print(f"Total loss: {loss.item()}; that's {l1.item()} task and {l2.item()} recon and {text_loss.item()} total text\n\n")

    if reset_model and (type(model) is EnhancedAgentBrain):
        model.reset()

    return loss.item(), l1.item(), l2.item(), tl1.item(), tl2.item()

def imagineAfterMove_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _imagineAfterMove_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _imagineAfterMove_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)

