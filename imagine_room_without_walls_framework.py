from general_framework import *
from generalQA_framework import *

# task to imagine the room without gold present at all.

prompts_imagineWithoutWalls = [ \
    "Imagine this room without walls.", \
    "What would this look like with nothing in your way.", \
    "Woah, what if there were no walls here?", \
    "How would this look without walls?", \
    "Can you imagine how this would look with no barriers at all?" \
]

prompts_imagineWithoutWalls_tensor = tensorify_list(prompts_imagineWithoutWalls)

########

def imagineWithoutWalls_data(batch_size):
    S = get_settings_batch(batch_size)
    imgs_in = get_images(S)
    imgs_out = get_images(S, ignore_walls=True)
    texts = simple_sample(batch_size, prompts_imagineWithoutWalls_tensor)
    return texts, imgs_in, imgs_out

def _imagineWithoutWalls_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.eval()

    if training:
        model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    task_texts, inp, out = imagineWithoutWalls_data(batch_size)
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

def imagineWithoutWalls_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _imagineWithoutWalls_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _imagineWithoutWalls_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)

    
