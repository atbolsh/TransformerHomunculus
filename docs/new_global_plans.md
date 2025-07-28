July 28th, 2025

From the progress on the frameworks, it's becoming quite apparent that I may need to shift my approach very soon.
I'm still not near the advanced tasks yet, and that's the only original part of my plan.

Here are several paths that may work:
 -- pay for cloud training, including for RL (use that as a test)
 -- switch to new architecture and MUCH smaller visual inputs / intermediaries
    -- probably not much different, but 64x64 visual input
    -- other options, especially for memory, are very possible
 -- pick a pre-trained network and run with it (modify, retrain, etc)

I will do the research for a shift (both training and pretrained model) in August, as long as last-minute experiments (eg testing frameworks)
The ideal would be a large-batch long train loop that handles ALL of the frameworks.

~~~~~~~~

Important lessons:
1) The plan is basically sound, the architecture *can* do all the tasks I want, but it takes a lot of massaging, and possibly not all at once in its current state.
2) RL is hard and not great
3) Jury is still out on the complex loss func (test me)
4) Current memory setup is inadequate (test something)
5) Others, I'm sure: EXPAND THIS LIST

~~~~~~~~

*Potential solution*

Train on low-res images.
When I want it to start exploring using more imagination, train 'translators' from high-res to low-res.
These can be further trained to 'mark' textures (just automatically). FOr example: teach a pattern distinction whcih you KNOW disappears at standard under-sampling,
then set it up so the autoencoder simply has to mark it, somehow.
  -- that 'high res' training would come much later, after 'stage 2' in the 'full plan', etc.

I'm liking this more and more.
Basically, mostly current setup (rewritten for speed?), plus a memory system I lift from someone and maybe some solid RL / video autoencoder pretraining,
followed by all the frameworks, and if luck, followed by the elusive 'phase 2'

