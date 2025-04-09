from model import *

# The below is copied over from model.py

## Memory processing

# class Memory Processor
# consumes the memory tensor, uses positional encoding on it, and uses the Decoder architecture (maybe 1 encoder layer?) to produce a vector based on the text and image encodigns
# THis vector is then used as context for the others (eg for answering questions or recalling saved images).
class MemoryProcessor(nn.Module):
    def __init__(self, sequence_length=128, embed_dim=768, num_heads=6, num_layers=8, dropout=0.1, norm_first=False):
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.sqrt_embed_dim = math.sqrt(embed_dim)
        self.embed = nn.Sequential(
            PositionalEncoding(embed_dim, sequence_length),
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.1),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=norm_first,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.text_img_marker = PositionalEncoding(embed_dim, 2) # extend to num_canvases? Or some other processing like that?

        # Really necessary? Think
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Convenient tensor:
        self.consecutive_indeces = torch.LongTensor(list(range(self.sequence_length))).to(self.get_device())

    def get_device(self):
        return self._modules['fc'][0].weight.device # this function should be part of nn.Module, honestly
   
    # Accepts already encoded inputs (dimension 768)
    def forward(self, x, text_context=None, img_context=None):
        x = self.embed(x)
        if (text_context is None) and (img_context is None):
            context = x
        elif (not (text_context is None)) and (img_context is None):
            context = text_context
        elif (text_context is None) and (not (img_context is None)):
            context = img_context
        else:
            context = torch.cat((text_context + self.text_img_marker.pe[:, :1], img_context + self.text_img_marker.pe[:, 1:]), dim=1) # default mode
        x = self.decoder(x, context)
        return self.fc(x)


# class Memory Encoder
# consumes arbitrary input (standardized to one text and 3 canvases) + memory; probably uses markers for 1st tensor, 2nd tensor, 3rd tensor, etc.
# Produces 4 vectors of length 768 (or 1?? consier)
# These are fed through the 'forget' gates and added to existing memory
# This class will have the infamous 'saved forget gate'
class MemoryEncoder(nn.Module):
    # I made it have fewer heads and layers for compactness; check later if this is enough
    def __init__(self, new_tokens=1, embed_dim=768, num_heads=3, num_layers=4, dropout=0.1, norm_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.new_tokens = new_tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=norm_first,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        
        self.fc_prep = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout)
        ) 
        self.fcs = nn.Sequential(*[nn.Linear(4*embed_dim, embed_dim) for i in range(self.new_tokens)]) # different final layer for each token used

    def get_device(self):
        return self._modules['decoder']._modules['layers'][0].linear1.weight.device # this function should be part of nn.Module, honestly
    
    # x is the text; context is the images
    def forward(self, x, context=None):
        if context is None:
            context = x
        x = self.decoder(x, context)
        prep = self.fc_prep(x) # 4 times larger internal dim
        results = torch.zeros((x.size()[0], self.new_tokens, x.size()[2]), dtype=x.dtype, device=x.device)
        for i in range(self.new_tokens):
            results[:, i, :] = torch.sum(self.fcs[i](prep), dim=1)
        return results


# class Memory
# Tensor storage
# initialization, addition of the new 1-token thing
# new_tokens MUST factor neatly into mem_size
class Memory:
    def __init__(self, mem_size=128, new_tokens=1):
        self.mem_size = mem_size
        self.new_tokens = new_tokens
        self.repetitions = int(mem_size // new_tokens)

        starter = 1.0 / torch.arange(1, self.repetitions + 1, 1)
        self.scales = starter.unsqueeze(0).repeat(new_tokens, 1).T.flatten().contiguous()
        self.scales = self.scales.unsqueeze(0).unsqueeze(2).contiguous()

        self.memory = None

    def is_empty(self):
        return (self.memory is None)

    def get_device(self):
        return self.scales.device

    def to(self, device):
        self.scales = self.scales.to(device)
        if not (self.memory is None):
            self.memory = self.memory.to(device)
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self):
        return self.to('cuda')

    # tokens has shape (batches, new_tokens, 768). Even if batches is 1
    def remember(self, tokens):
        # last layer forgot to unsqueeze
        if len(tokens.size()) == 2:
            tokens = tokens.unsqueeze(1).repeat(1, self.new_tokens, 1)
        tokens = tokens.repeat(1, self.repetitions, 1)
        if self.memory is None:
            self.memory = tokens # have to start somewhere
        else:
            self.memory = self.memory * (1.0 - self.scales) + (self.scales * tokens)


##################
class VisionCanvases:
    def __init__(self, num_canvases=3, num_channels=3, width=224, height=224):
        self.num_canvases=num_canvases
        self.num_channels=num_channels
        self.width=width
        self.height=height

        self.canvases=None
        self.device='cpu'

    def get_device(self):
        if self.canvases is None:
            return self.device
        else:
            return self.canvases[0].device

    def is_empty(self):
        return (self.canvases is None)

    def to(self, device):
        self.device=device
        if not (self.canvases is None):
            for i in range(self.num_canvases):
                self.canvases[i].to(device)
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self):
        return self.to('cuda')

    def store(self, img):
        b = img.size()[0]      
        if self.canvases is None:
            self.device = img.device
            self.canvases = [torch.zeros(b, self.num_channels, self.width, self.height, device=img.device) for i in range(self.num_canvases)]
            for i in range(self.num_canvases):
                self.canvases[i][:, :, :, :] = img[:, :, :, :] # this notation ensures a copy rather than pass by reference, I think
            self.canvases.contiguous()
        else:
            self.cancases = self.canvases[1:] # drop oldest
            self.canvases.append(torch.zeros(b, self.num_channels, self.width, self.heith, device=img.device))
            self.canvases[-1][:, :, :, :] = img[:, :, :, :]

##################

# Additions to the brain (or new brain class):
# 1) fixed slots for the vision canvases, possibly with their own 'signatures' to tell them apart
# 2) Memory processing on every input
# 3) Standardized 'forward' call that uses the memory and the canvases and the text
# 4) Everything else more or less carried over

# The new Brain type
# Code copied over from model.py before modification
# I may delete that file (and merge these two files) in a later commit
class EnhancedAgentBrain(nn.Module):
    def __init__(self, vocab_size=10000, mem_size=128, new_tokens=1):
        super().__init__()
        self.img_enc = ImageTransformerEncoder()
        self.img_dec = ImageTransformerDecoder()
        self.text_enc = SentenceTransformerEncoder(num_embed=vocab_size)
        self.text_dec = SentenceTransformerDecoder(num_embed=vocab_size)
        self.dopamine = IntermediateTransformerScorer() # for RL; not yet tested, use later

        # 6 inputs; 3 img canvases, 1 img input, text input, memory input
        self.context_tagging = nn.Parameter(torch.empty((1, 6, 768)))
        nn.init.uniform_(self.context_tagging, -1.0/math.sqrt(768), 1.0/math.sqrt(768)) 

        # Memory processing
        self.memory = Memory(mem_size, new_tokens)
        self.memenc = MemoryEncoder(new_tokens=new_tokens)
#        self.memproc = MemoryProcessor(sequence_length=mem_size) # canceling memory processing for now

        # image canvases
        self.canvases = VisionCanvases(3)

    def get_device(self):
        return self.img_enc.get_device()

    def get_masks(self, text_batch, use_masks=True):
        if use_masks:
            src_attention_mask = generate_src_mask(text_batch.size(1), text_batch.device)
            src_key_padding_mask = generate_src_padding_mask(text_batch)
        else:
            src_attention_mask = None
            src_key_padding_mask = None
        return src_attention_mask, src_key_padding_mask

    def get_text_encoding(self, text_batch, src_attention_mask, src_key_padding_mask):
        return self.text_enc(text_batch, src_attention_mask=src_attention_mask, src_key_padding_mask=src_key_padding_mask)

    def get_text_decoding(self, text_encoding, src_attention_mask, src_key_padding_mask, context=None, return_full=True):
        return self.text_dec(text_encoding, context, return_full=return_full, tgt_mask=src_attention_mask, tgt_key_padding_mask=src_key_padding_mask)

    # Unlike the below, whether or not ret_imgs is marked, img_dec will be called and the reconstruction saved.
    def forward(self, text_batch, img_batch, ret_imgs=False, return_full=True, use_masks=True):
        if self.canvases.is_empty():
            self.canvases.store(img_batch)

        context = []
        b = text_batch.size()[0]

        real_img_context = self.img_enc(img_batch)
        context.append(real_img_context)

        for i in range(self.canvases.num_canvases):
            context.append(self.img_enc(self.canvases[i]))

        src_attention_mask, src_key_padding_mask = self.get_masks(text_batch, use_masks)
        text_encoding = self.get_text_encoding(text_batch, src_attention_mask, src_key_padding_mask)
#        context.append(text_encoding) # This cannot be done! this would let the text output have direct access to full text input

        if self.memory.is_empty():
            context.append(torch.zeros(b, 128, 768, device=text_batch.device))
        else:
            context.append(self.memory.memory)

        for i in range(len(context)):
            context[i] += self.tagging[i]
        tensor_context = torch.cat(context, dim=1)

        # now that we have built the shared representation, we use it
        text_probs = self.get_text_decoding(text_encoding, src_attention_mask, src_key_padding_mask, tensor_context, return_full)
        # For images and memory, the text_encoding can be added to context (in fact, *must* be)
        context.append(text_encoding + self.tagging[-1])
        full_context = torch.cat(tensor_context, context[-1], dim=1)

        img_recon = self.img_dec(real_image_context, full_context)
        self.canvases.store(img_recon)

        self.memory.remember(self.mem_enc(text_encoding, full_context))

        if ret_imgs:
            return text_probs, img_recon
        else:
            return text_probs
    
    # Default Brain operation. Preserved for legacy execution (and perhaps to retrain correct answers using transferred weights)
    def old_forward(self, text_batch, img_batch=None, ret_imgs=False, return_full=True, use_masks=True):
        src_attention_mask, src_key_padding_mask = self.get_masks(text_batch, use_masks)
        text_encoding = self.get_text_encoding(text_batch, src_attention_mask, src_key_padding_mask)
        if img_batch is None:
            img_context = text_encoding # just feed the text features back to itself.
        else:
            img_context = self.img_enc(img_batch)
        text_probs = self.get_text_decoding(text_encoding, src_attention_mask, src_key_padding_mask, img_context, return_full)
        if not ret_imgs:
            return text_probs
        else:
            if img_batch is None:
                batches = text_batch.size()[0]
                img_encoding = torch.zeros((batches, 256, 768), device=self.get_device())
            else:
                img_encoding = img_context
            img_reconstruction = self.img_dec(img_encoding, text_encoding)
            return text_probs, img_reconstruction

    # used for training question-answering; text_batch_in has SAME shape, but answers zero-ed out.
    # slower than straight-up text training. ONly use if you are ALSO training image reconstruction.
    # otherwise, just use the old 'forward' and only compute the gradient with respect to the text transformers
    def qa_forward(self, text_batch_in, text_batch_out, img_batch=None, ret_imgs=False, return_full=True, use_masks=True):
        src_attention_mask_in, src_key_padding_mask_in = self.get_masks(text_batch_in, use_masks)
        src_attention_mask_out, src_key_padding_mask_out = self.get_masks(text_batch_out, use_masks)
        text_encoding_in = self.get_text_encoding(text_batch_in, src_attention_mask_in, src_key_padding_mask_in)
        text_encoding_out = self.get_text_encoding(text_batch_out, src_attention_mask_out, src_key_padding_mask_out)
        if img_batch is None:
            img_context = text_encoding_in # just feed the text features back to itself.
        else:
            img_context = self.img_enc(img_batch)
        # when generating the answer string, each token still sees all the preceding tokens, and NO others
        # this is valid; the answer token (eg 'left') will still have to be inferred
        # I will reassess whether this is valid for longer answers
        text_probs = self.get_text_decoding(text_encoding_out, src_attention_mask_out, src_key_padding_mask_out, img_context, return_full)
        if not ret_imgs:
            return text_probs
        else:
            if img_batch is None:
                batches = text_batch.size()[0]
                img_encoding = torch.zeros((batches, 256, 768), device=self.get_device())
            else:
                img_encoding = img_context
            img_reconstruction = self.img_dec(img_encoding, text_encoding_in) # CRITICAL! Only the 'question', not the 'answer', is used when computing the image!
            return text_probs, img_reconstruction

    def img_autoencoder(self, img_batch, context = None):
        img_encoding = self.img_enc(img_batch)
        if context is None:
            context = img_encoding
        return self.img_dec(img_encoding, context)

    def sentence_autoencoder(self, text_batch, context = None, return_full=True, use_masks=False):
        src_attention_mask, src_key_padding_mask = self.get_masks(text_batch, use_masks)
        text_encoding = self.get_text_encoding(text_batch, src_attention_mask, src_key_padding_mask)
#        print(text_encoding)
        # These lines not needed; this chance is covered more smoothly by the conditions within text_dec
#        if context is None:
#            context = text_encoding
        return self.get_text_decoding(text_encoding, src_attention_mask, src_key_padding_mask, context, return_full)

    # I will keep this as the default dopamine signal; may spin some other 'raw default' types later.
    # can sometimes train the dopamine layer alone, without any text or image gradients.
    # Must then set the img_gradient and text_gradient settings to False, or you get a memory leak
    def evaluate_text(self, text_batch, img_batch=None, img_gradient=True, text_gradient=True):
        if text_gradient:
            text_encoding = self.text_enc(text_batch)
        else:
            with torch.no_grad():
                text_encoding = self.text_enc(text_batch)
        if img_batch is None:
            return self.dopamine(text_encoding)
        else:
            if img_gradient:
                context = self.img_enc(img_batch)
            else:
                with torch.no_grad():
                    context = self.img_enc(img_batch)
            return self.dopamine(text_encoding, context)

    # useful for comparing images
    # I may later switch to using this by default
    def two_img_context(self, img1_batch, img2_batch):
        batches = img1_batch.size()[0]
        context = torch.zeros((batches, 256*2), device = self.get_device())
        img1_encoding = self.img_enc(img1_batch)
        img2_encoding = self.img_enc(img2_batch)
        context[:, :256] += (img1_encoding + self.img_tagging[0])
        context[:, 256:] += (img2_encoding + self.img_tagging[1])
        return context

    def select(self, logits, temp=0.0, ret_all = True, temp_eps = 1e-4):
        # logits is a vector batch x vocab size
        # ret_all means return log probs and entropy; that's the assumed behavior.
        # This is a different input than model_declarations_v2: I am assuming batch generation, and the logits come from different batches
        if temp < temp_eps:
            preds = torch.argmax(logits, dim=1)
            if not ret_all:
                return preds
            log_probs = torch.max(F.log_softmax(logits, dim=1), dim=1)
            dist = Categorical(logits = logits)
            entropy = dist.entropy() # should be length batch
            return preds, log_probs, entropy
        else: # very convenient. Maybe add a torch.no_grad()? Dunno
            dist = Categorical(logits = logits / temp)
            preds = dist.sample()
            if not ret_all:
                return preds
            entropy = dist.entropy()
            log_probs = dist.log_prob(preds)
            return preds, log_probs, entropy

    def extend(self, seed, is_terminated, context=None, temp=1.0, ret_all=True, temp_eps = 1e-4):
        s = seed.size()
        output = torch.zeros((s[0], s[1] +1), dtype = torch.long, device = seed.device)
        output[:, :-1] += seed
        # Process and return only the logits for the final character. Use the provided (visual) context
        # the final zero is ignored in the computation due to the mask
        logits = self.sentence_autoencoder(output, context, use_masks=True, return_full=False)
        if not ret_all:
            preds = self.select(logits, temp, ret_all, temp_eps)
        else:
            preds, log_probs, entropy = self.select(logits, temp, ret_all, temp_eps)
        preds = preds * torch.logical_not(is_terminated) # set all past-terminated actions / tokens to 0
        output[:, -1] += preds
        #print(preds==2)
        is_terminated = torch.logical_or(is_terminated, (preds==2))
        if not ret_all:
            return output, preds, is_terminated
        else:
            return output, preds, log_probs, entropy, is_terminated # maybe also multiply by is_terminated? Dunno.

    def generate(self, x=None, context=None, maxlen = None, temp=1.0, ret_all=True, temp_eps = 1e-4, default_batches = 1):
        if maxlen is None:
            maxlen = self.text_enc.sequence_length
        if x is None:
            x = torch.zeros((default_batches, 1), device=self.get_device(), dtype=torch.long)
        if ret_all:
            lp = torch.zeros((default_batches, 1), device=self.get_device()) # default dtype
            ent = torch.zeros((default_batches, 1), device=self.get_device())
        batches, _ = x.size()
        is_terminated = torch.zeros(batches, dtype=torch.bool, device=self.get_device()) # none are terminated initially
        if ret_all:
            lp = torch.zeros((batches, 1), device=self.get_device()) # default dtype
            ent = torch.zeros((batches, 1), device=self.get_device())
        firstGone = False
        while (x.size()[1] < maxlen) and (not torch.all(is_terminated)):
            if ret_all:
                x, _, newlp, newent, is_terminated = self.extend(x, is_terminated, context, temp, ret_all, temp_eps)
                if firstGone: # so, in all cases except the first value
                    lp = F.pad(lp, (0, 1))
                    ent = F.pad(ent, (0, 1))
                else:
                    firstGone=True
                lp[:, -1] += newlp
                ent[:, -1] += newent
            else:
                x, _, is_terminated = self.extend(x, is_terminated, context, temp, ret_all, temp_eps)
        if ret_all:
            return x, lp, ent
        else:
            return x

#    def compute_probabilities(self, x, seed_offset=0, context=None, temp=1.0):
#        """Given sentences x, possibly computed by another model, compute the logpas and entropies for all the values chosen.
#           (Notice that we are talking about the values ALREADY chosen, not the choices this model would make.)"""
#        batches, total_len = x.size()
#        gen_len = total_len - seed_offset
#        logpas = torch.zeros(batches, gen_len, device=x.device)
#        entropies = torch.zeros(batches, gen_len, device=x.device)
#        for i in range(gen_len):
#            cutoff = i + seed_offset
#            logits = self.sentence_autoencoder(x[:, :cutoff], context, use_masks=True, return_full=False)
#            dist = Categorical(logits = logits / temp)
#            entropies[:, i] = dist.entropy()
#            logpas[:, i] = dist.log_prob(x[:, cutoff])
#        return logpas, entropies

    # SINGLE only returns probs for final val; multi returns all probs (assumes identical contexts)
    def compute_probabilities(self, x, seed_offset=1, context=None, temp=1.0, single=False):
        if single:
            return self._compute_probabilities_SINGLE(x, context, temp)
        else:
            return self._compute_probabilities_MULTI(x, seed_offset, context, temp)

    def _compute_probabilities_MULTI(self, x, seed_offset, context=None, temp=1.0):
        """Given sentences x, possibly computed by another model, compute the logpas and entropies for all the values chosen.
           (Notice that we are talking about the values ALREADY chosen, not the choices this model would make.)"""
        batches, total_len = x.size()
        gen_len = total_len - seed_offset
        logits = self.sentence_autoencoder(x, context=context, use_masks=True, return_full=True)[:, :, seed_offset-1:-1] # should be batches x tokens x genlen
        logits = logits.transpose(1, 2) # bathes x genlen x tokens
        logits = logits.reshape((batches*gen_len, self.text_dec.vocab_size)) # (batches * genlen x tokens)
        dist = Categorical(logits = logits / temp)

        y = x[:, seed_offset:].reshape((batches * gen_len)) # single vector

        logpas = dist.log_prob(y).reshape((batches, gen_len))
        entropies = dist.entropy().reshape((batches, gen_len))
        return logpas, entropies

    def _compute_probabilities_SINGLE(self, x, context=None, temp=1.0):
        """Like the above, but only returns the value for the final token"""
#        batches, total_len = x.size()
#        gen_len = total_len - seed_offset
        # I think this is finally it. All the non-terminal tokens are input; the terminal one is index.
        logits = self.sentence_autoencoder(x[:, :-1], context=context, use_masks=True, return_full=False) # should be batches x tokens
        dist = Categorical(logits = logits / temp)

        inds = x[:, -1]

        logpas = dist.log_prob(inds)
        entropies = dist.entropy()
        return logpas, entropies

    # Experimental use of self.generate for a particular type of input question
    def answer_image_comparison_question(self, text_seed, img1_batch, img2_batch, maxlen = None, temp=1.0, ret_all=True, temp_eps = 1e-4):
        return self.generate(text_seed, self.two_img_context(img1_batch, img2_batch), maxlen, temp, ret_all, temp_eps)


