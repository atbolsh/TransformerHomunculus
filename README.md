Jan 24, 2025

This is the repo that is going to become AGI, or at least the first repo that is seriously going to be trained on cognitive tasks using my game.

I started clean, copying over only what I needed from the Florence repo: the game, the code for the agent, and some code for RL.

That last is the messiest, and won't be needed the longest. It's also the only one where I kept the demonstration ipynb notebooks, since 
in that case, they still encode some crucial RL algorithm information.

Env details: 
llava-florence on penguins.farm
player on penguins.army

~~~~
~~~~

I had meant to make this on the penguins.army server, but I have technical difficulties with it, so I am sticking with penguins.farm while I fix those.

For a demonstration of the code (in case I forget and haven't made .ipynb's for training in the main space of this folder), look to the Florence repo.

~~~~
~~~~

Next task: pretraining, especially on images. I need to see the capacity of the image transformer autoencoder I have chosen. This can be done on the smaller
penguins.farm server.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jan 25, 2025


Using Florence notebooks as precedent, I wrote 3 python notebooks to feed game snapshots, text, or both to the brain.

** Note: For text, I'm using the totoeba small English dataset, and the tokenizer created in Florence.
** I will save the tokenizer but not the lang data in git
** The English data can be found here: https://tatoeba.org/en/downloads 

I will delete the text and game alone notebooks, but BrainMultimodal remains as the template for further work.

Some changes to the model itself were necessary; the Florence code turned out to be imperfect (masks unused; weird defaults)

~~~~

Tmr: modify the multimodal notebook into pretraining, and launch it before your trip back.

create 40 images at a time, but only load them onto the device in sets of 10.

batchsize 10 for the text.

I will probably be done with visual autoencoder training by the time I'm half done with the sentence pretraining.

Copy epoch number, lr's, etc, from lang_experiments.

If those turn out to be a problem, experiment.

~~~~

This is secondary to hardware debugging. I really need my main server back.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jan 27, 2025

No hardware debugging

Started text pretraining on the small server. Will take several days.

Some more code modifications for ease of use.

Will do image pretraining next (besides hardware debugging).

Using randn for image input, unfortunately; will also use randn for text input when training images.

Teaching them to talk together will be finetuning, if it's needed.

I had an idea how to make the two talk to each other during training (see Brain Multimodal), but it ended up being too slow.

I will really need the large server.

~~~~

Next steps: definitely hardware debugging.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jan 28, 2025

Checked on training: it was slow. At 5.6ish, instead of 3.0ish for the smaller model back when it was at the same epoch.

Relaunching training using a scaled lr and no Adam betas. Will see if it's better.

~~~~

Lots of hardware debugging.

I got the damn thing to see both GPUs, but this is going to be a pain. Each GPU is going to keep falling off

In addition, I have to log in without seeing the password getting typed out.
 -- display is a mess, with two Xorg sessions.
 -- I should be able to fix this with xorg.conf, but I tried once and it bored me. Will maybe retry only later.

Frequent saving is in order.

Will launch a train sequence for the image portion, on penguins.army.

In addition, will ask dad to make penguins.arm visible on the web.

I think most work on that monstrosity will come from the laptop.

~~~~

Tmr:
X -- check on this training sequence. End it? Restart?
     -- I'll let it finish. Why not? I may let the big server run one more time with it. Will see.
X -- launch an image training sequence, this time on the big server.
     -- neatly fits on the big GPU, all alone. Will use that one a lot, I think.
     -- as expected, the GPU memory is well-used, but the temperature isn't that high.
     -- bottleneck is probably numpy-based image generation (CPU); that will need to change. I may rewrite my game in JAX at some point.
 -- think about external backups for the brain weights. Consider a dedicated external drive. Check your HW supplies, dig one out and test.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jan 29, 2025

Simple, clean day. Continued text pretraining on penguins.farm; launched image pretraining on the workhorse.

Quick brainstorm about infrastructure and how to make my life easier; file infrastructure_ideas. Will implement those on some weekend.

General plan for next several weeks:
 -- pretraining
 -- pretraining with both together
 -- question-answer pairs of various sorts, mixed in with 'repeat' epochs to prevent catastrophic forgetting
 -- don't focus on making it make sense, yet; give it the raw skills (like left / right, straight path, corner id, playing game, simple logic, etc.)
 -- step back from skill teaching; find a way to connect all of these things together.

~~~~

UPDATE:
nothing works. Some auto-update broke my ubuntu drivers, which is a damn shame.
X Tomorrow will involve fixing the driver before I can do anything intelligent.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jan 30, 2025

Driver fixed. Using 560 now. After some apt games (autoremove, fix-broken, etc), the damn thing actually worked. I didn't even have to reinstall torch.

Relaunched image pretraining. Seems to be twice as slow as before. No explanation I can find. Will run more tests later.

Tmr:

X -- temporary interruption, look at the output quality.
X -- possibly interrupt the penguins.farm pretraining.
   -- interrupted, no need for the marginal returns. Will test it and use this pretraining for the next task.
 -- No other big tasks. Extra time? Read the Deepseek paper and play with it. Try to understand their approach.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jan 31, 2025

The autoencoder did not learn at all (beyond smearing the average color on the entire output image).

I changed positional embedding to have more of a punch (so each patch knows where it is in the sequence).
I then restarted training with a higher lr.

It will train the entire weekend.

If no luck, next week will basically just be debugging the image autoencoder portion.
There are many ways to do this, starting with fake datasets that include only one image (black square in corner), looking up autoencoders for 224x224 images in general, etc.


Also: synced the two folders on the two machines using git.

Added a 'global_plans' doc for the next several months (while working on the game).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 3, 2025

Damn thing did fail. Debugging this week.

Plan:
X learn to ssh into this at all (even from your own network)
  -- 192.168.5.116; will do something more with this IP address later
X use the wifi mesh
 backup all the stuff
 start debugging? Or do tmr?


