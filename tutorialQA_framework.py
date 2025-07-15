from general_framework import *
from generalQA_framework import *

############
# On to the functions for the task at hand

# Left-right (interchangeable)
task2_prompts_lrgold = [ \
    "Is the gold to the left or to the right of you?", \
    "Which side is it on?", \
    "Is it to the left or right of the agent?", \
    "Do you need to go left or right to get the gold?", \
    "Please tell me whether the gold is left or right.", \
    "Please tell me which side is the gold on.", \
    "Which side to you need to go to get it?", \
    "Which side has gold?", \
    "On which side is the gold?"
]

task2_Lreplies_lrgold = [ \
    "Left", \
    "It's to the left.", \
    "It's on the left.", \
    "Go left."
]

task2_Rreplies_lrgold = [ \
    "Right", \
    "It's to the right.", \
    "It's on the right.", \
    "Go right."
]

task2_prompts_udgold = [ \
    "Is the gold above or below you?", \
    "Is it up or down from the agent?", \
    "Do you need to go up or down to get the gold?", \
    "Please tell me whether the gold is above or below you.", \
    "Please tell me whether the gold is up or down.", \
    "Do you need to go up or down to get it?", \
    "Which side has gold?", \
    "On which side is the gold?"
]

task2_Ureplies_udgold = [ \
    "Up", \
    "Above", \
    "It's up.", \
    "It's above me.", \
    "Go up."
]

task2_Dreplies_udgold = [ \
    "Down", \
    "Below", \
    "It's down.", \
    "It's below me.", \
    "Go down."
]

task2_prompts_lragent = [ \
    "Are you to the left or right of the gold?", \
    "Which side is the gold on?", \
    "Is the agent to the left or right of the gold?", \
    "Please tell me whether you are right or left of the gold.", \
    "Please tell me which side you are relative to the gold.", \
    "On which side of the gold are you?"
]

task2_Lreplies_lragent = [ \
    "Left", \
    "I'm to the left.", \
    "The agent is on the left."
]

task2_Rreplies_lragent = [ \
    "Right", \
    "I'm to the right.", \
    "The agent is on the right."
]

task2_prompts_udagent = [ \
    "Are you below or above the gold?", \
    "Is the agent above or below the gold?", \
    "Please tell me whether you are up or down from the gold.", \
    "Please tell me whether you are above or below the gold."
]

task2_Ureplies_udagent = [ \
    "Up", \
    "I'm above it.", \
    "The agent is above the gold."
]

task2_Dreplies_udagent = [ \
    "Down", \
    "I'm below it.", \
    "The agent is below the gold."
]

########

task2_prompts_lrgold_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_prompts_lrgold)]).contiguous().to(device)
task2_Lreplies_lrgold_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_Lreplies_lrgold)]).contiguous().to(device)
task2_Rreplies_lrgold_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_Rreplies_lrgold)]).contiguous().to(device)

task2_prompts_udgold_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_prompts_udgold)]).contiguous().to(device)
task2_Ureplies_udgold_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_Ureplies_udgold)]).contiguous().to(device)
task2_Dreplies_udgold_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_Dreplies_udgold)]).contiguous().to(device)

task2_prompts_lragent_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_prompts_lragent)]).contiguous().to(device)
task2_Lreplies_lragent_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_Lreplies_lragent)]).contiguous().to(device)
task2_Rreplies_lragent_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_Rreplies_lragent)]).contiguous().to(device)

task2_prompts_udagent_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_prompts_udagent)]).contiguous().to(device)
task2_Ureplies_udagent_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_Ureplies_udagent)]).contiguous().to(device)
task2_Dreplies_udagent_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task2_Dreplies_udagent)]).contiguous().to(device)

########

task2_prompts_lrgold_lens = get_lens(task2_prompts_lrgold_tensor)
task2_prompts_udgold_lens = get_lens(task2_prompts_udgold_tensor)
task2_prompts_lragent_lens = get_lens(task2_prompts_lragent_tensor)
task2_prompts_udagent_lens = get_lens(task2_prompts_udagent_tensor)

########

# uninituitive, but pygame flips these
# this is 'left' and 'right' relative to the game setup, not the agent
is_gold_left = (lambda settings: settings.agent_y > settings.gold[0][1])
is_gold_up = (lambda settings: settings.agent_x > settings.gold[0][0])

is_agent_left = (lambda settings: not is_gold_left(settings))
is_agent_up = (lambda settings: not is_gold_up(settings))

########

task2_lrgold_generator = (lambda settings_batch: text_generator(settings_batch, \
                                                                task2_prompts_lrgold_tensor, \
                                                                task2_Lreplies_lrgold_tensor, \
                                                                task2_Rreplies_lrgold_tensor, \
                                                                task2_prompts_lrgold_lens, \
                                                                is_gold_left, \
                                                                device \
                                                               ))

task2_udgold_generator = (lambda settings_batch: text_generator(settings_batch, \
                                                                task2_prompts_udgold_tensor, \
                                                                task2_Ureplies_udgold_tensor, \
                                                                task2_Dreplies_udgold_tensor, \
                                                                task2_prompts_udgold_lens, \
                                                                is_gold_up, \
                                                                device \
                                                               ))

task2_lragent_generator = (lambda settings_batch: text_generator(settings_batch, \
                                                                 task2_prompts_lragent_tensor, \
                                                                 task2_Lreplies_lragent_tensor, \
                                                                 task2_Rreplies_lragent_tensor, \
                                                                 task2_prompts_lragent_lens, \
                                                                 is_agent_left, \
                                                                 device \
                                                                ))

task2_udagent_generator = (lambda settings_batch: text_generator(settings_batch, \
                                                                 task2_prompts_udagent_tensor, \
                                                                 task2_Ureplies_udagent_tensor, \
                                                                 task2_Dreplies_udagent_tensor, \
                                                                 task2_prompts_udagent_lens, \
                                                                 is_agent_up, \
                                                                 device \
                                                                ))

text_generators = [task2_lrgold_generator, task2_udgold_generator, task2_lragent_generator, task2_udagent_generator]

########

task2_lrgold_generator_simple = (lambda settings_batch: text_generator_simple(settings_batch, \
                                                                              task2_prompts_lrgold_tensor, \
                                                                              task2_Lreplies_lrgold_tensor, \
                                                                              task2_Rreplies_lrgold_tensor, \
                                                                              task2_prompts_lrgold_lens, \
                                                                              is_gold_left, \
                                                                              device \
                                                                             ))

task2_udgold_generator_simple = (lambda settings_batch: text_generator_simple(settings_batch, \
                                                                              task2_prompts_udgold_tensor, \
                                                                              task2_Ureplies_udgold_tensor, \
                                                                              task2_Dreplies_udgold_tensor, \
                                                                              task2_prompts_udgold_lens, \
                                                                              is_gold_up, \
                                                                              device \
                                                                             ))

task2_lragent_generator_simple = (lambda settings_batch: text_generator_simple(settings_batch, \
                                                                               task2_prompts_lragent_tensor, \
                                                                               task2_Lreplies_lragent_tensor, \
                                                                               task2_Rreplies_lragent_tensor, \
                                                                               task2_prompts_lragent_lens, \
                                                                               is_agent_left, \
                                                                               device \
                                                                              ))

task2_udagent_generator_simple = (lambda settings_batch: text_generator_simple(settings_batch, \
                                                                               task2_prompts_udagent_tensor, \
                                                                               task2_Ureplies_udagent_tensor, \
                                                                               task2_Dreplies_udagent_tensor, \
                                                                               task2_prompts_udagent_lens, \
                                                                               is_agent_up, \
                                                                               device \
                                                                              ))

text_generators_simple = [task2_lrgold_generator_simple, \
                          task2_udgold_generator_simple, \
                          task2_lragent_generator_simple, \
                          task2_udagent_generator_simple]

########

# THis is modified from the original. This function can be used to train or test. For the train loop, run this function with training and compute_grad and provide the optimizer

def _qa_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
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
    
    texts_lrg = task2_lrgold_generator_simple(S)
    texts_udg = task2_udgold_generator_simple(S)
    texts_lra = task2_lragent_generator_simple(S)
    texts_uda = task2_udagent_generator_simple(S)

    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = sdt[ind:ind + batch_size].to(device)

    all_texts = [control_texts, texts_lrg, texts_udg, texts_lra, texts_uda]
    text_inds = list(range(5))
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
            all_probs[t_ind] = (model(all_texts[t_ind], imgs, ret_imgs=False, create_context=False))
        else:
            all_probs[t_ind] = (model(all_texts[t_ind], imgs, ret_imgs=False))
    #now, all_probs, despite random execution order, has the corresponding output for every element of all_texts

    all_losses = [get_text_loss(all_probs[i], all_texts[i]) for i in range(5)]
    
    loss = sum(all_losses)

    if training:
        loss.backward()#retain_graph=True) # needed? consider.
        optimizer.step()
        optimizer.zero_grad()
        if type(model) is EnhancedAgentBrain:
            model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()}:\n{all_losses[0].item()} control,\n{all_losses[1].item()} lrg,\n{all_losses[2].item()} udg,\n{all_losses[3].item()} lra,\n{all_losses[4].item()} uda\n\n")

    if reset_model and (type(model) is EnhancedAgentBrain):
        model.reset()

    return (loss.item(), all_losses[0].item(), all_losses[1].item(), all_losses[2].item(), all_losses[3].item(), all_losses[4].item())

def qa_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _qa_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _qa_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)


