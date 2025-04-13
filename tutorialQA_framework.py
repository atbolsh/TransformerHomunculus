# should have all the torch libraries I need
from visual_transformer import *

device = torch.device('cuda:0') # can be changed after import

# from viusual_transformer.enhanced_model import *

from game import *

game_settings = BIG_tool_use_advanced_2_5
game_settings.gameSize = 224 # for compatibility with brain's expected size
G = discreteGame(game_settings)

########

vocab_size = 10000
# tokenizer.save_model(".", "tokenizer/eng_sentences_tokenizer_vc10000")
tokenizer = ByteLevelBPETokenizer(
    "./text_pretraining_tokenizer/eng_sentences_tokenizer_vc10000_v2-vocab.json",
    "./text_pretraining_tokenizer/eng_sentences_tokenizer_vc10000_v2-merges.txt",
)   
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)   
tokenizer.enable_truncation(max_length=32)
tokenizer.enable_padding()


## Dataset
class SampleDataset(Dataset):
    def __init__(self, seq_length = 32, evaluate: bool = False, tokenizer=None, device = None):
        if device is None:
            device = 'cpu'
        self.device = device
        self.seq_length = seq_length
        if tokenizer is None:
            tokenizer = ByteLevelBPETokenizer(
                "./text_pretraining_tokenizer/eng_sentences_tokenizer_v2-vocab.json",
                "./text_pretraining_tokenizer/eng_sentences_tokenizer_v2-merges.txt",
            )   
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )   
        tokenizer.enable_truncation(max_length=self.seq_length)
        tokenizer.enable_padding()#length=seq_length)
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []

        src_files = Path("./text_pretraining_data/").glob("*-eval.txt") if evaluate else Path("./text_pretraining_data/").glob("*-train.txt")
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i): 
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i]).to(self.device)

sdt = SampleDataset(tokenizer=tokenizer)
sdv = SampleDataset(tokenizer=tokenizer, evaluate=True)


ent_criterion = nn.CrossEntropyLoss(ignore_index=0)

def get_text_loss(res, inputs):
    return torch.sum(ent_criterion(res[:, :, :-1], inputs[:, 1:])

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
    "Are you to the left or right of the gold?", \
    "Which side is the gold on?", \
    "Is the agent to the left or right of the gold?", \
    "Please tell me whether you are right or left of the gold.", \
    "Please tell me which side you are relative to the gold.", \
    "On which side of the gold are you?"
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

def get_lens(prompt_tensor):
    lens = []
    max_len = prompt_tensor.size()[1]
    for prompt in prompt_tensor:
        for ending in range(max_len):
            if prompt[ending] == 2:
                lens.append(ending) # want to skip the final </s>
                break
    return lens

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

def _stitch(prompt, reply, container, length):
    container[:prompt.size()[0]] = prompt
    container[length] = 225 # tokenizer.encode(' ')
    reply_len = reply.size()[0] - 1
    container[length+1:length+reply_len+1] = reply[1:]
    return container

def text_generator(settings_batch, prompts, yes_responses, no_responses, prompt_lengths, func, device=device):
    batchsize = len(settings_batch)
    prompt_num, prompt_size = prompts.size()
    reply_size = max(yes_responses.size()[1], no_responses.size()[1])# - 1 # skip initial <s> in reply
    yes_num = yes_responses.size()[0]
    no_num = no_responses.size()[0]
    input_tensor = torch.zeros((batchsize, reply_size + prompt_size), device=device, dtype=prompts.dtype)
    output_tensor = torch.zeros((batchsize, reply_size + prompt_size), device=device, dtype=prompts.dtype)
    
    for i in range(batchsize):
        ind = torch.randint(0, prompt_num, (1,)).item()
        prompt = prompts[ind]
        length = prompt_lengths[ind]
        if func(settings_batch[i]):
            reply = yes_responses[torch.randint(0, yes_num, (1,)).item()]
        else:
            reply = no_responses[torch.randint(0, no_num, (1,)).item()]
        input_tensor[i] = prompt
        _stitch(prompt, reply, output_tensor[i], length)

    return input_tensor.contiguous(), output_tensor.contiguous()

def text_generator_simple(settings_batch, prompts, yes_responses, no_responses, prompt_lengths, func, device=device):
    batchsize = len(settings_batch)
    prompt_num, prompt_size = prompts.size()
    reply_size = max(yes_responses.size()[1], no_responses.size()[1])
    yes_num = yes_responses.size()[0]
    no_num = no_responses.size()[0]
    #input_tensor = torch.zeros((batchsize, reply_size + prompt_size), device=device, dtype=prompts.dtype)
    output_tensor = torch.zeros((batchsize, reply_size + prompt_size), device=device, dtype=prompts.dtype)
    
    for i in range(batchsize):
        ind = torch.randint(0, prompt_num, (1,)).item()
        prompt = prompts[ind]
        length = prompt_lengths[ind]
        if func(settings_batch[i]):
            reply = yes_responses[torch.randint(0, yes_num, (1,)).item()]
        else:
            reply = no_responses[torch.randint(0, no_num, (1,)).item()]
        #input_tensor[i] = prompt
        _stitch(prompt, reply, output_tensor[i], length)

    return output_tensor.contiguous()

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

def get_images(settings_batch, device=device):
    batch_size = len(settings_batch)
    img = torch.zeros(batch_size, 224, 224, 3)
    for i in range(batch_size):
        G2 = discreteGame(settings_batch[i])
        img[i] = torch.tensor(G2.getData())
    img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(device)
    return img

def get_settings_batch(batch_size):
    return [G.random_bare_settings(gameSize=224, max_agent_offset=0.5) for i in range(batch_size)]

########

# The train loop is below this; look to the original in order to recreate that and adjust to your particular desires
