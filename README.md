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





