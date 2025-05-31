from mem_canvas_use_framework import *

# This framework is for pretraining mem_enc as an autoencoder in the first place.
# given inputs and contexts, it's supposed to create an encoding which can be used to re-create context
# RAW. This is untested, and needs calibration. May or may not work. THe 'context' object may need better placemarkers to tell the different inputs apart.

# 1152 is the image context: 256 for the input, 3*256 for the canvases, and 128 for the memory from the previous timestep
class MemoryDecContext(nn.Module):
    def __init__(self, sequence_length=1152, embed_dim=768, num_heads=6, num_layers=8, dropout=0.1, norm_first=False):
        super(MemoryDecContext, self).__init__()
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

    def get_device(self):
        return self._modules['fc'][0].weight.device # this function should be part of nn.Module, honestly
   
    # Accepts 0 inputs and uses memory to reproduce the context
    def forward(self, x, mem_context):
        x = self.embed(x)
        x = self.decoder(x, context)
        return self.fc(x)

context_dec = MemoryDecContext().to(device)
text_dec = MemoryDecContext(32).to(device) # sequence length corresponding to text input. Will only use sdt for now.

# context_dec.load_state_dict(torch.load('brain_checkpoints/context_dec_mem_training_helper.pth', weights_only=True, map_location=device))
# text_dec.load_state_dict(torch.load('brain_checkpoints/text_dec_mem_training_helper.pth', weights_only=True, map_location=device))

mem_criterion = nn.MSELoss()


N_lookback = 4

# MAKE SURE THE OPTIMIZER INCLUDES CONTExT_DEC AND TEXT_DEC!!!! OTHERWISE, THIS WILL NOT WORK AT ALL!!!!!!!!!
def _mem_pretraining_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.eval()

    if training:
        model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = sdt[ind:ind + batch_size].to(device) # will be kept the same throughout the task. Is this wise? Unsure

    img_tensor_list = []
    prompt_tensor_list = []

    for step in range(N_lookback):
        imgs = mem_task_img_sample(batch_num)
        img_tensor_list.append(imgs)
        prompt_tensor_list.append(control_texts) # not a copy operation, I checked, just many pointers to the same object

    for step in range(N_lookback):
        _, _ = model(prompt_tensor_list[step], img_tensor_list[step])

    model.soft_reset() # detach all relevant gradients
    model.detach()

    src_attention_mask, src_key_padding_mask = model.get_masks(control_texts, use_masks)
    text_encoding = model.get_text_encoding(control_texts, src_attention_mask, src_key_padding_mask)

    text_encoding.detach()
    model.detach()

    # here comes the part we actually care about. model.context should be full, and we have text encodings.

    mem_encoding = model.mem_enc(text_encoding, model.context) # single point only for now.

    text_recon = text_dec(torch.zeros_like(text_encoding), mem_encoding)
    context_recon = context_dec(torch.zeros_like(model.context), mem_encoding)

    full_context = torch.cat((model.context, text_encoding), dim=1) # context including text (not 'remembered', not reconstructed)

    img_recon_real = model.img_dec(context_recon[:, :256, :], full_context)
    img_recon_canvas_0 = model.img_dec(context_recon[:, 256:512, :], full_context)
    img_recon_canvas_1 = model.img_dec(context_recon[:, 512:768, :], full_context)
    img_recon_canvas_2 = model.img_dec(context_recon[:, 768:1024, :], full_context)

    text_loss = mem_criterion(text_recon, text_encoding)
    context_loss = mem_criterion(context_recon, model.context)

    img_loss = img_criterion(img_tensor_list[-1], img_recon_real) + \
               img_criterion(img_tensor_list[-2], img_recon_canvas_2) + \
               img_criterion(img_tensor_list[-3], img_recon_canvas_1) + \
               img_criterion(img_tensor_list[-4], img_recon_canvas_0)

    loss = text_loss + (context_loss / 32) + img_loss

    if training:
        
        loss.backward()#retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        if type(model) is EnhancedAgentBrain:
            model.soft_reset()

    if printing:
        # may need to comment out the torch save portions, later
        torch.save(text_dec.state_dict(), "brain_checkpoints/text_dec_mem_training_helper.pth")
        torch.save(context_dec.state_dict(), "brain_checkpoints/context_dec_mem_training_helper.pth")
        print(f"Total loss: {loss.item()}; that's {text_loss.item()} for reconstructing text, {context_loss.item()} for reconstructing context.\n\n")

    if reset_model and (type(model) is EnhancedAgentBrain):
        model.reset()

    return loss.item(), text_loss.item(), context_loss.item()


def mem_pretraining_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order = True, model_eval=True, reset_model=True, printing=True, training=False):
    if compute_grad:
        return _mem_pretraining_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _mem_pretraining_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training)

