May 30, 2025

I think I can modify the memory to be usable.

This is a guide: https://arxiv.org/pdf/2404.05726

I will need an explicit image-to-single-vector reducer, and same with text.
 -- the quality of this encoding can also be improved if I train standard QA frameworks using it; that is, using the small embedding instead of the large
 -- that may be a pain to set up; I will see.
Possibly, instead, this can be done with the full context.

The catch is that instead of pretraining with dumb MSE loss like the current mem_enc_pretraining network,
iI can train with entropy on the text and MSE on the image (not feature vector), getting much better compression (later).

The current generator of memory can be replaced with a several-step thing that does this.

This is an initial stage.

Around phase 2, I can do this: use GPU single-vector 'tags', and store loadable full 'contexts' (images? vectors?) in CPU memory or disk.

That can be an explicit skill / trick that the agent learns around the same time as it stores task lists for itself.

~~~~~~~~

THe 'tag' thing can be done later.

I don't want to rewrite the entire EnhancedBrain. Most of the training should transfer very smoothly to the next version.
THis is especially true because the vast majority of frameworks do not use the memory tensor anyway.

~~~~~~~~

So, plan:
 -- all current frameworks trained. MSE pretraining for memory happens, with the understanding that this module will be discarded
 -- start phase 2.
 -- then, possibly in parallel, rewrite how memory works. 
 -- store the vectors in the same way as now (or not? Or start multi-vector compression?), but the Mem encoder is re-written


