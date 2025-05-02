from general_framework import *
from game_logic_solver import *

# This framework is similar to rel_positionQA_framework and uses much of generalQA_framework.
# However, unlike rel_positionQA_framework, the responses are sequences of actual action tokens:
# <forward> (token 1, action 1), <clock> (turn clockwise; token 3, action 3), <anticlock> (turn anticlockwise; token 4, action 4) 
# Moving backwards will not be attempted for the time being. Maybe I'll introduce that action later.
#
# NOTE!! This is a very rough framework. many off-by-one errors may be present. Debug before running

special_symbols = set([1, 3, 4])
symbol_action_map = { 1:1, 3:3, 4:4 } # map from text tokens to action indeces
action_symbol_map = { 1:1, 3:3, 4:4 } # map from action indeces to text tokens

# given a batch of settings and a length, produce the set of text tokens corresponding to the best sequence of moves up to length.
def get_action_text_tensors(settings_batch, length):
    N = len(settings_batch)
    text_reponse = torch.tensor((N, length), dtype=torch.long) # on CPU for now; will only transfer to device at the end, faster this way.
    for ind in rand(N):
        setting = settings_batch[ind]
        game = discreteGame(setting)
        finished = False

        for step in range(length):
            if finished:
                continue
            move = best_move_forward(game)
            reward = game.actions[move]()
            text_response[ind, step] = action_symbol_map[move]
            if reward > 0:
                finished = True
                if step < length - 1:
                    text_response[ind, step + 1] = 2
    return text_response.to(device), text_response # it's useful to have the CPU version too, just as a guide for downstream CPU tasks.


prompts_to_action = [ \
    "Please take the best move in this position:", \
    "Solve this.", \
    "What is the move here?", \
    "Move", \
    "Use default action", \
    "Get to the gold."
]

prompts_to_action_tensor = tensorify_list(prompts_to_action)
prompts_to_action_lens = get_lens(prompts_to_action_tensor)

# generate tensor full of prompts to action, followed by move tokens up to trace_length
def initial_text_generator(batchsize, prompts, prompt_lengths, moves, trace_length, device=device):
    prompt_num, prompt_size = prompts.size()
    output_tensor = torch.zeros((batchsize, prompt_size + trace_length), device=device, dtype=prompts.dtype)
    output_lens_tensor = torch.zeros((batchsize,), device='cpu', dtype=torch.long)
    finished_tensor = torch.zeros_like(output_lens_tensor, dtype=torch.bool)
    
    for i in range(batchsize):
        ind = torch.randint(0, prompt_num, (1,)).item()
        prompt = prompts[ind]
        length = prompt_lengths[ind]
        output_tensor[i, :length - 1] = prompt[:-1]
#        output_tensor[i, length] = 255 # code for space
        output_tensor[i, length:length + trace_length] = moves[i] # corresponds to settings, not prompt
    
        output_lens_tensor[i] = length-1 # take into account the erasure of stop token

    return output_tensor.contiguous(), output_lens_tensor.contiguous(), finished_tensor

# given current step, the full text tensor, and a batch of games, produce current img tensor, input text, and output text tensor, and update finished_tensor
def tensors_for_step_full_move_framework(games, full_text_tensor, prompt_lens_tensor, finished_tensor, step, trace_length, device=device):
    if step >= trace_length:
        raise ValueError("step cannot be greater than trace_length. Check full_move_framework.")
    img_tensor = get_images([game.settings for game in games]) # moves before the game
    input_tensor = torch.zeros_like(full_text_tensor)
    output_tensor = torch.zeros_like(full_text_tensor)
    # will need both, in main loop, finished before and after the move
    finished_output_tensor = torch.zeros_like(finished_tensor, dtype=torch.bool)
    batchsize = len(games)
    for ind in range(batchsize):
        if not finished_tensor[ind]:
            prompt_length = prompt_lens_tensor[ind]
            game = games[ind]
            # prompt and trace before the step:
            input_tensor[ind, :prompt_length + step] = full_text_tensor[ind, :prompt_length + step]
            # prompt and trace right after the step; target:
            output_tensor[ind, :prompt_length + step + 1] = full_text_tensor[ind, :prompt_length + step + 1]
            # let's update the game and the finished tensor
            move = best_move_forward(game)  # faster to recreate on cpu than copy from cuda, I think (could be wrong)
            reward = game.actions[move]() # updates game object and tells if gold was eaten
            finished_output_tensor[ind] = (reward > 0)
        else:
            finished_output_tensor[ind] = True # no need to forget that
    return img_tensor, input_tensor, output_tensor, finished_output_tensor

# custom text loss only looks at the prompt losses themselves, and the last move to be generated, present in the output tensor
def full_move_text_loss(probs, output_tensor, prompt_lens_tensor, finished_tensor, step, trace_length, device=device):
    loss = torch.tensor(0, device=device)
    batch_size = output_tensor.size()[0]
    for ind in range(batch_size):
        length = prompt_lens_tensor[ind]
        # add the loss from the prompt entropy (should learn the prompts themselves)
        # off by one because 'probs' is naturally shifted by one to the left
        loss += ent_criterion(probs[ind, :, :length-1], inputs[ind, 1:length])
        # add the loss from the move itself; using slices instead of index for correct dimensionality
        loss += ent_criterion(probs[ind, :, length-1+step:length-1+step+1], inputs[ind, length+step:length+step+1])
    return loss
    

full_move_trace_length = 4 # up to 4 moves from the one game, during training; longer than the canvases, and the batchsize constraints not too extreme
def _full_move_batch(batch_size, model, optimizer=None, batch_num=0, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.eval()

    if training:
        model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")
        
    S = get_settings_batch(batch_size)
    # just the optimum traces, nothing more
    full_trace_tensor = get_action_text_tensors(S, full_move_trace_length)
    # get the full text tensor and the lens and initialize the finished tensor
    full_text_tensor, prompt_lens_tensor, finished_tensor = initial_text_generator(batch_size, \
                                                                                   prompts_to_action_tensor, \
                                                                                   prompts_to_action_lens, \
                                                                                   full_trace_tensor, \
                                                                                   full_move_trace_length, \
                                                                                   device=device)
    # initialize games and loss nums
    games = [discreteGame(setting) for setting in S)
    reconstruction_loss = torch.tensor(0, device=device)
    custom_text_loss = torch.tensor(0, device=device)
    
    for step in range(full_move_trace_length):
        img_tensor, input_tensor, output_tensor, finished_output_tensor = tensors_for_step_full_move_framework(games, \
                                                                                                               full_text_tensor, \
                                                                                                               prompt_lens_tensor, \
                                                                                                               finished_tensor, \
                                                                                                               step, \
                                                                                                               full_move_trace_length, \
                                                                                                               device=device)

        probs, recon_imgs = model(input_tensor, imgs, ret_imgs=True, create_context=True) # compute the likelihoods and the reconstructions
        reconstruction_loss += img_criterion(img_tensor, recon_imgs)
        # custom text loss only looks at the prompt losses themselves, and the last move to be generated, present in the output tensor
        custom_text_loss += full_move_text_loss(probs, output_tensor, prompt_lens_tensor, finished_tensor, step, full_move_trace_length, device=device)
        # model.soft_reset() # necessary?? try both ways

    loss = custom_text_loss + 1000 * reconstruction_loss # finetune the relative weights here.
    
    if training:
        loss.backward()#retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        if type(model) is EnhancedAgentBrain:
            model.soft_reset()
    
    if printing:
        print(f"Total loss: {loss.item()}; that's {reconstruction_loss.item()} on the images and {custom_text_loss.item()} for the text and the moves\n\n")

    if reset_model and (type(model) is EnhancedAgentBrain):
        model.reset()

    return loss.item(), reconstruction_loss.item(), custom_text_loss.item()


