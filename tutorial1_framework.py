# Easier this way; both tutorials share some common tools, like the tokenizer
from tutorialQA_framework import *

task1_prompts = [ \
    "Imagine the line from the agent to the nearest gold.", \
    "What's the straight path from the agent to the gold?", \
    "Please draw the straight line to the gold from the agent.", \
    "How would you move from the agent to the gold?", \
    "What's a direct path from the agent to the gold?", \
    "From the agent to the nearest coin, please draw a path." ]

task1_text_tensor = torch.tensor([x.ids for x in tokenizer.encode_batch(task1_prompts)]).contiguous().to(device)

########

# output format: image in, image out, text context
# no output text for now

def task1_img_sample(num_sample=40):
    img_in = torch.zeros(num_sample, 224, 224, 3)
    img_out = torch.zeros(num_sample, 224, 224, 3)
    for i in range(num_sample):
        bare_settings = G.random_bare_settings(gameSize=224, max_agent_offset=0.5)
        G2 = discreteGame(bare_settings)
        img_in[i] = torch.tensor(G2.getData())
        G2.bare_draw_arrow_at_gold()
        img_out[i] = torch.tensor(G2.getData())
    img_in = torch.permute(img_in, (0, 3, 1, 2)).contiguous().to(device)
    img_out = torch.permute(img_out, (0, 3, 1, 2)).contiguous().to(device)
    num_texts = task1_text_tensor.size()[0]
    text_inds = torch.randint(0, num_texts, size=(num_sample,), device=device)
    texts = task1_text_tensor[text_inds]
    return img_in, img_out, texts

########

img_criterion = nn.MSELoss()

########


