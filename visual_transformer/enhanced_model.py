from .model import *
from .memory import *
from .vision_canvas import *

##################
# The below is copied over from model.py

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
    # can be called with create_context=False to just produce the next token, in an otherwise static scene
    def forward(self, text_batch, img_batch=None, ret_imgs=False, return_full=True, use_masks=True, create_context=True):
        if (img_batch is None) and create_context:
            raise ValueError("Must provide img_batch to create new context")
        if ret_imgs and (not create_context):
            raise ValueError("to generatre new images, create_context must be true")

        b = text_batch.size()[0]
        src_attention_mask, src_key_padding_mask = self.get_masks(text_batch, use_masks)
        text_encoding = self.get_text_encoding(text_batch, src_attention_mask, src_key_padding_mask)
#        context.append(text_encoding) # This cannot be done! this would let the text output have direct access to full text input

        if create_context:
            if self.canvases.is_empty():
                self.canvases.store(img_batch)
    
            context = []
    
            real_img_context = self.img_enc(img_batch)
            context.append(real_img_context)

            for i in range(self.canvases.num_canvases):
                context.append(self.img_enc(self.canvases[i].canvas))
    
            if self.memory.is_empty():
                context.append(torch.zeros(b, 128, 768, device=text_batch.device))
            else:
                context.append(self.memory.memory)
    
            for i in range(len(context)):
                context[i] += self.tagging[i]
            tensor_context = torch.cat(context, dim=1)
    
            self.context = tensor_context
    
        # now that we have built the shared representation, we use it
        text_probs = self.get_text_decoding(text_encoding, src_attention_mask, src_key_padding_mask, self.context, return_full)
        self.memory.remember(self.mem_enc(text_encoding, self.context)) # always store it in memory; no step should be forgotten

        if create_context:
            # For images and memory, the text_encoding can be added to context (in fact, *must* be)
            context.append(text_encoding + self.tagging[-1])
            full_context = torch.cat(tensor_context, context[-1], dim=1)
    
            img_recon = self.img_dec(real_image_context, full_context)
            self.canvases.store(img_recon)
       
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

    def sentence_autoencoder(self, text_batch, context = None, return_full=True, use_masks=False, store_memory=False):
        src_attention_mask, src_key_padding_mask = self.get_masks(text_batch, use_masks)
        text_encoding = self.get_text_encoding(text_batch, src_attention_mask, src_key_padding_mask)
        if store_memory:
            self.memory.remember(self.mem_enc(text_encoding, context)) # The case context==None is smoothly handled by mem_enc
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

    def extend(self, seed, is_terminated, context=None, temp=1.0, ret_all=True, temp_eps = 1e-4, store_memory=True):
        s = seed.size()
        output = torch.zeros((s[0], s[1] +1), dtype = torch.long, device = seed.device)
        output[:, :-1] += seed
        # Process and return only the logits for the final character. Use the provided (visual) context
        # the final zero is ignored in the computation due to the mask
        logits = self.sentence_autoencoder(output, context, use_masks=True, return_full=False, store_memory=store_memory)
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

    # run 'forward' once first, to compute self.context correctly. the text_probs that result can be discarded
    def generate(self, x=None, context=None, maxlen = None, temp=1.0, ret_all=True, temp_eps = 1e-4, default_batches = 1, store_memory=True):
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
                x, _, newlp, newent, is_terminated = self.extend(x, is_terminated, context, temp, ret_all, temp_eps, store_memory)
                if firstGone: # so, in all cases except the first value
                    lp = F.pad(lp, (0, 1))
                    ent = F.pad(ent, (0, 1))
                else:
                    firstGone=True
                lp[:, -1] += newlp
                ent[:, -1] += newent
            else:
                x, _, is_terminated = self.extend(x, is_terminated, context, temp, ret_all, temp_eps, store_memory)
        if ret_all:
            return x, lp, ent
        else:
            return x

    # SINGLE only returns probs for final val; multi returns all probs (assumes identical contexts)
    def compute_probabilities(self, x, seed_offset=1, context=None, temp=1.0, single=False, store_memory=False):
        if single:
            return self._compute_probabilities_SINGLE(x, context, temp, store_memory)
        else:
            return self._compute_probabilities_MULTI(x, seed_offset, context, temp, store_memory)

    def _compute_probabilities_MULTI(self, x, seed_offset, context=None, temp=1.0):
        """Given sentences x, possibly computed by another model, compute the logpas and entropies for all the values chosen.
           (Notice that we are talking about the values ALREADY chosen, not the choices this model would make.)"""
        batches, total_len = x.size()
        gen_len = total_len - seed_offset
        logits = self.sentence_autoencoder(x, context=context, use_masks=True, return_full=True, store_memory=store_memory)[:, :, seed_offset-1:-1] # should be batches x tokens x genlen
        logits = logits.transpose(1, 2) # bathes x genlen x tokens
        logits = logits.reshape((batches*gen_len, self.text_dec.vocab_size)) # (batches * genlen x tokens)
        dist = Categorical(logits = logits / temp)

        y = x[:, seed_offset:].reshape((batches * gen_len)) # single vector

        logpas = dist.log_prob(y).reshape((batches, gen_len))
        entropies = dist.entropy().reshape((batches, gen_len))
        return logpas, entropies

    def _compute_probabilities_SINGLE(self, x, context=None, temp=1.0, store_memory=False):
        """Like the above, but only returns the value for the final token"""
#        batches, total_len = x.size()
#        gen_len = total_len - seed_offset
        # I think this is finally it. All the non-terminal tokens are input; the terminal one is index.
        logits = self.sentence_autoencoder(x[:, :-1], context=context, use_masks=True, return_full=False, store_memory=store_memory) # should be batches x tokens
        dist = Categorical(logits = logits / temp)

        inds = x[:, -1]

        logpas = dist.log_prob(inds)
        entropies = dist.entropy()
        return logpas, entropies


