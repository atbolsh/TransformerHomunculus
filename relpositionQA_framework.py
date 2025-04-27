from general_framework import *
from game_logic_solver import *

# This framewok is very much like tutorialQA_framework, but instead of absolute position, it focuses on relative position
# Framework is RAW. Should not have deep issues (trained tutorialQA successfully before, after all), but it needs testing for syntax, etc.

prompts_willIntersectForward = [ \
    "If you go forward, will you eat?", \
    "Is the gold in your path?", \
    "How do you figure, will you intersect it just by going forward?", \
    "Is going forward enough?", \
    "Can you get the gold without turning?", \
    "You don't need to turn, right?" \
]

Yreplies_willIntersectForward = [ \
    "Yep", \
    "Absolutely.", \
    "Certainly", \
    "I think so.", \
    "Uh-huh.", \
    "Sure" \
]

Nreplies_willIntersectForward = [ \
    "Nuh-uh", \
    "No", \
    "I don't think so.", \
    "Certainly not", \
    "Absolutely not", \
    "Nah" \
]

###

prompts_whichWayTurn = [ \
    "Which way should you turn, do you figure?", \
    "Damn, how can I twist in the right direction?", \
    "Which way to fix our direction?", \
    "How should you turn?" \
]

CWreplies_whichWayTurn = [ \
    "Clockwise", \
    "I should turn clockwise", \
    "CW", \
    "Clockwise, sir!" \
]

CCWreplies_whichWayTurn = [ \
    "Counter-clockwise", \
    "I should turn counter-clockwise", \
    "CCW", \
    "Counter-clockwise, sir!" \
]

###

prompts_whatNextMove = [ \
    "Damn it, what's the move here, partner?", \
    "What should you do here?", \
    "In this position, what should you do?", \
    "How do you figure, what's the move for us?", \
    "What's the move?" \
]

Freplies_whatNextMove = [ \
    "Just go straight.", \
    "We just go straight", \
    "Full speed ahead!" \
]

CWreplies_whatNextMove = CWreplies_whichWayTurn # I don't want to write more
CCWreplies_whatNextMove = CCWreplies_whichWayTurn

########

prompts_willIntersectForward_tensor = tensorify_list(prompts_willIntersectForward)
Yreplies_willIntersectForward_tensor = tensorify_list(Yreplies_willIntersectForward)
Nreplies_willIntersectForward_tensor = tensorify_list(Nreplies_willIntersectForward)

prompts_whichWayTurn_tensor = tensorify_list(prompts_whichWayTurn)
CWreplies_whichWayTurn_tensor = tensorify_list(CWreplies_whichWayTurn)
CCWreplies_whichWayTurn_tensor = tensorify_list(CCWreplies_whichWayTurn)

prompts_whatNextMove_tensor = tensorify_list(prompts_whatNextMove)
Freplies_whatNextMove_tensor = tensorify_list(Freplies_whatNextMove)
CWreplies_whatNextMove_tensor = tensorify_list(CWreplies_whatNextMove)
CCWreplies_whatNextMove_tensor = tensorify_list(CCWreplies_whatNextMove)

########

prompts_willIntersectForward_lens = get_lens(prompts_willIntersectForward_tensor)
prompts_whichWayTurn_lens = get_lens(prompts_whichWayTurn_tensor)
prompts_whatNextMove_lens = get_lens(prompts_whatNextMove_tensor)

########

# True for Y, False for N
will_intersect_forward = (lambda settings: willIntersectForward(discreteEngine(deepcopy(settings))))

# True fro CW, False for CCW
best_turn_cw = (lambda setings: not should_turn_anticlockwise_forward(discreteEngine(deepcopy(settings))))

# 0 for forward, 1 for CW, 2 for CCW
throwaway_index_helper = {1:0, 3:1, 4:2}
best_move = (lambda settings: throwaway_index_helper[best_move_forward(discreteEngine(deepcopy(settings)))])

########

willIntersectForward_generator_simple = (lambda settings_batch: text_generator_simple(settings_batch, \
                                                                                      prompts_willIntersectForward_tensor, \
                                                                                      Yreplies_willIntersectForward_tensor, \
                                                                                      Nreplies_willIntersectForward_tensor, \
                                                                                      prompts_willIntersectForward_lens, \
                                                                                      will_intersect_forward, \
                                                                                      device \
                                                                                     ))

whichWayTurn_generator_simple = (lambda settings_batch: text_generator_simple(settings_batch, \
                                                                              prompts_whichWayTurn_tensor, \
                                                                              CWreplies_whichWayTurn_tensor, \
                                                                              CCWreplies_whichWayTurn_tensor, \
                                                                              best_turn_cw, \
                                                                              device \
                                                                             ))

whatNextMove_generator_simple = (lambda settings_batch: text_generator_simple_GENERAL(settings_batch, \
                                                                                      prompts_whatNextMove_tensor, \
                                                                                      [Freplies_whatNextMove_tensor, CWreplies_whatNextMove_tensor, CCWreplies_whatNextMove_tensor], \
                                                                                      best_move, \
                                                                                      device \
                                                                                     ))

text_generators_simple = [willIntersectForward_generator_simple, whichWayTurn_generator_simple, whatNextMove_generator_simple]

########

def _relposion_qa_batch(batch_size, model, optimizer=None, batch_num=0, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
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
    
    texts_wif = willIntersectForward_generator_simple(S)
    texts_wwt = whichWayTurn_generator_simple(S)
    texts_wnm = whatNextMove_generator_simple(S)

    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = sdt[ind:ind + batch_size].to(device)

    all_texts = [control_texts, texts_wif, texts_wwt, texts_wnm]
    text_inds = list(range(4))
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

    all_losses = [get_text_loss(all_probs[i], all_texts[i]) for i in range(4)]
    
    loss = sum(all_losses)

    if training:
        loss.backward()#retain_graph=True) # needed? consider.
        optimizer.step()
        optimizer.zero_grad()
        if type(model) is EnhancedAgentBrain:
            model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()}:\n{all_losses[0].item()} control,\n{all_losses[1].item()} willIntersectForward,\n{all_losses[2].item()} whichWayTurn,\n{all_losses[3].item()} whatNextMove\n\n")

    if reset_model and (type(model) is EnhancedAgentBrain):
        model.reset()

    return (loss.item(), all_losses[0].item(), all_losses[1].item(), all_losses[2].item(), all_losses[3].item())

def relposion_qa_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _relposion_qa_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _relposion_qa_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)


