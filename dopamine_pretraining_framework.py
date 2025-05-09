from general_framework import *
from game_logic_solver import *

# Defaults to just discount expected reward from consuming gold, unnormalized
# THis is just an initialization; this will get more involved later on.

default_discount_rate = 0.98 # so that after 32 steps, 1.0 becomes roughly 0.5

# only works for bare settings (1 gold)
def score(s):
    steps_to_solve = len(_trace_forward(s))
    return default_discount_rate ** steps_to_solve

def score_tensor(settings_batch, device=device):
    return torch.tensor([score(s) for s in settings_batch], dtype=torch.float, device=device)

def _dopamine_pretraining_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.eval()

    if training:
        model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    S = get_settings_batch(batch_size)
    scores = score_tensor(S)
    img_tensor = get_images(S)

    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = sdt[ind:ind + batch_size].to(device)

    control_probs, control_recon, control_dopamine = model(control_texts, img_tensor, ret_imgs=True, ret_dopamine=True)

    img_loss = img_criterion(conrol_recon, img_tensor)
    dopamine_loss = img_criterion(contro_dopamine, scores)
    text_loss = get_text_loss(control_probs, control_texts)

    # pick coefficients for dopamine better; important
    loss = img_loss + dopamine_loss + (text_loss / 1000) 

    if training:
        loss.backward()#retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        if type(model) is EnhancedAgentBrain:
            model.soft_reset()

     if printing:
        print(f"Total recon loss: {loss.item()}; that's {text_loss.item()} text, {img_loss.item()} img, and {dopamine_loss.item()} dopamine \n\n")

    if reset_model and (type(model) is EnhancedAgentBrain):
        model.reset()

    return loss.item(), text_loss.item(), img_loss.item()



def dopamine_pretraining_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _dopamine_pretraining_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _dopamine_pretraining_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)

