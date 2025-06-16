from general_framework import *

# This framework is for recalling (reconstructing) images from between 1 and N elements ago
# The first 3 are probably stored in canvases (and that will be the first version); the others
# will require difficult reconstruction from memory.
#
# This is focused only on images for now. Text recollection may come, later.
# RAW. Needs calibration; may or may not work. THe 'context' tensor may need better markers for what comes from which input

N_lookback = 4#3 # just canvases for now
min_lookback = 0

current_image_prompts = [\
    "Hey, recall the current image again?", \
    "Focus on the present view, please.", \
    "What do you see right now?", \
    "Think about the present game, for a moment."\
]

prev_image_prompts = [\
    "Hey, recall the last image, again?", \
    "Focus on the last view, please.", \
    "What did you see 1 image ago?", \
    "Think about the last game, for a moment.", \
    "Woah! What was that 1 image ago, again???" \
]

def get_image_prompts(ind):
    return [\
        f"Hey, could you recall the image {ind} ago, again?", \
        f"Focus on the view {ind} ago, please.", \
        f"Hey, recall the game {ind} ago for me, will you?", \
        f"Woah! What was that {ind} images ago, again??", \
        f"Think about the game from {ind} ago for me, will you?", \
        f"What did you see {ind} games ago, again.", \
        f"Think about the view {ind} steps ago, for a moment." \
    ]

lookback_prompts = [current_image_prompts, prev_image_prompts]

for n in range(2, N_lookback):
    lookback_prompts.append(get_image_prompts(n))

lookback_prompts = lookback_prompts[min_lookback:N_lookback]

def mem_task_img_sample(num_sample=40):
    img_in = torch.zeros(num_sample, 224, 224, 3)
    #arrow_drawers = np.random.randint(0, 2, size=num_sample)
    for i in range(num_sample):
        bare_settings = G.random_bare_settings(gameSize=224, max_agent_offset=0.5)
        G2 = discreteGame(bare_settings)
        #if arrow_drawers[i]:
        #    G2.G2.bare_draw_arrow_at_gold()
        img_in[i] = torch.tensor(G2.getData())
    img_in = torch.permute(img_in, (0, 3, 1, 2)).contiguous().to(device)
    return img_in

# newest to oldest, possible targets
def get_prompts(img_tensor_list):
    batch_size = img_tensor_list[0].size()[0]
    lookback_vals = np.random.randint(0, N_lookback - min_lookback, (batch_size,))

    target = torch.zeros_like(img_tensor_list[0])
    prompts = []

    for ind in range(batch_size):
        lookback = lookback_vals[ind]
        target[ind] = img_tensor_list[lookback][ind].detach() 
        prompts.append(random.choice(lookback_prompts[lookback + min_lookback]))

    target = target.contiguous()

    prompt_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(prompts)]).contiguous().to(device)
    padded_prompt_tensor = torch.zeros((batch_size, 32), dtype=prompt_tensor.dtype, device=prompt_tensor.device)
    padded_prompt_tensor[:, :prompt_tensor.size()[1]] += prompt_tensor
    return padded_prompt_tensor, target

def _mem_canvas_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.eval()

    if training:
        model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    if N_lookback - min_lookback < 2:
        raise ValueError("Go back and adjust min_lookback and N_lookback; must be at least 2 apart")

    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = sdt[ind:ind + batch_size].to(device) # will be kept the same throughout the task. Is this wise? Unsure

    img_tensor_list = []
    prompt_tensor_list = []

    for step in range(N_lookback):
        imgs = mem_task_img_sample(batch_size)
        img_tensor_list.append(imgs)
        if step == N_lookback - 1:
            L = img_tensor_list[min_lookback:]
            L.reverse() # img_tensor_list should be unharmed by this
            prompt_tensor, target = get_prompts(L)
            prompt_tensor_list.append(prompt_tensor)
        else:
            prompt_tensor_list.append(control_texts) # not a copy operation, I checked, just many pointers to the same object

    # now, with all things prepared, let's run the model
    for step in range(N_lookback):
        if step < N_lookback - 2:
            _ = model(prompt_tensor_list[step], img_tensor_list[step], ret_imgs=False)
        elif step == N_lookback - 2:
            control_probs, control_recon = model(prompt_tensor_list[step], img_tensor_list[step], ret_imgs=True)
        else: # if step == N_lookback - 1
            text_probs, recon = model(prompt_tensor_list[step], img_tensor_list[step], ret_imgs=True)
    
    task_img_loss = img_criterion(recon, target)
    task_text_loss = get_text_loss(text_probs, prompt_tensor_list[N_lookback - 1])
    
    control_img_loss = img_criterion(control_recon, img_tensor_list[-2])
    control_text_loss = get_text_loss(control_probs, control_texts)

    img_loss = task_img_loss + control_img_loss
    text_loss = task_text_loss + control_text_loss
 
    loss = img_loss + (text_loss / 1000)

    if training:
        loss.backward()#retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        if type(model) is EnhancedAgentBrain:
            model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()}; that's {task_img_loss.item()} for recalled reconstructions, {control_img_loss.item()} for normal images, and {text_loss.item()} total text\n\n")

    if reset_model and (type(model) is EnhancedAgentBrain):
        model.reset()

    return loss.item(), task_img_loss.item(), task_text_loss.item(), control_img_loss.item(), control_text_loss.item()


def mem_canvas_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _mem_canvas_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _mem_canvas_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)

   






