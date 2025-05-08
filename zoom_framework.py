from general_framework import *
from generalQA_framework import *
from game_logic_solver import *
from copy import deepcopy

# task to imagine the room without gold present at all.

prompts_zoomAgent = [ \
    "Zoom in on just the agent, please.", \
    "Take a closer look at yourself", \
    "Woah, what's that on your face?", \
    "Please zoom in on yourself.", \
    "Zoom in on the agent."
]

prompts_zoomAgent_tensor = tensorify_list(prompts_zoomAgent)

prompts_zoomGold = [ \
    "Zoom in on just the gold, please.", \
    "Take a closer look at the gold", \
    "Please zoom the gold", \
    "Zoom in on gold, please."
]

prompts_zoomGold_tensor = tensorify_list(prompts_zoomGold)

prompts_zoomHalfway = [ \
    "Zoom in on the path, please.", \
    "Take a closer look at the path for a second", \
    "Please zoom in on the path.", \
    "Zoom in on the path."
]

prompts_zoomHalfway_tensor = tensorify_list(prompts_zoomHalfway)

########

def get_zoomAgent_numpy(s):
    G2 = discrteGame(s)
    center = (s.agent_x, s.agent_y)
    factor = 2
    return G2.zoom([center], [factor])[0] # 0 index to make the 'batch' coordinate go away.

def get_zoomGold_numpy(s):
    G2 = discrteGame(s)
    center = (s.gold[0][0], s.gold[0][1])
    factor = 3
    return G2.zoom([center], [factor])[0] # 0 index to make the 'batch' coordinate go away.

def get_zoomHalfway_numpy(s):
    G2 = discrteGame(s)
    center = (0.5*(s.agent_x + s.gold[0][0]), 0.5*(s.agent_y + s.gold[0][1]))
    factor = 2
    return G2.zoom([center], [factor])[0] # 0 index to make the 'batch' coordinate go away.

# 0 for agent, 1 for gold, 2 for halfwar
def zoom_data(batch_size, task_ind=0):
    S = get_settings_batch(batch_size)
    imgs_in = get_images(S)
    funcDict = {0:get_zoomAgent_numpy, 1:get_zoomGold_numpy, 2:get_zoomHalfway_numpy}
    func = funcDict[task_ind]
    imgs_out = torch.zeros(num_sample, 224, 224, 3)
    for i in range(batch_size):
        imgs_out[i] = torch.tensor(func(S[i]))
    imgs_out = torch.permute(imgs_out, (0, 3, 1, 2)).contiguous().to(device)

    tensorDict = {0:prompts_zoomAgent_tensor, 1:prompts_zoomHalfway_tensor, 2:prompts_zoomHalfway_tensor}
    texts = simple_sample(batch_size, tensorDict[task_ind])
    return texts, imgs_in, imgs_out

# Adding task_ind here; short wrapping functions for each task provided below
def _zoom_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, task_ind=0):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.eval()

    if training:
        model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    task_texts, inp, out = zoom_data(batch_size, task_ind)
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

def _zoomAgent_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    return _zoom_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, 0):

def _zoomGold_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    return _zoom_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, 1):

def _zoomHalfway_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    return _zoom_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, 2):


def zoom_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, task_ind=0):
    if compute_grad:
        return _zoom_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, task_ind)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _zoom_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, task_ind)

def zoomAgent_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _zoom_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, 0)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _zoom_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, 0)

def zoomGold_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _zoom_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, 1)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _zoom_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, 1)

def zoomHalfway_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _zoom_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, 2)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _zoom_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, 2)


