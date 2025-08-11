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
  -- will do something more with this IP address later
X use the wifi mesh
X backup all the stuff
  -- for some reason, the device does not appear automatically in /media/atbolsh on penguins.army
  -- In that case, run dmesg
  -- should see line like
            [2952.552623]  sdc: sdc1
  -- from there, run 
            sudo mount sdc1 KINGSTON_mountpoint
  -- note that KINGSTON_mountpoint is a folder in ~
  -- when done, run
            sudo umount KINGSTON_mountpoint
  -- a bit tedious; I may make a script for this. FOr now, a funny quirk.
X start debugging? Or do tmr?

~~~~~

Continuing on penguins.farm

In the interests of min noise, running debugging here.
Will return to the behemoth when it is appropriate.

~~~~~

Debugging began, in ImageAutoDebugging.

Results are as expected so far.

Next steps involve going through step-by-step, then trying to copy someone else's approach (if own approach fails).
 -- after this works, I will play with other ways to indicate position
 -- I may also try the 'different scale feature' embeddings, to make zooming later easier. I can have some fun with this.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 3, 2025

Debugging continues.

It appears that most uses of the transformerDecoder assume input masks - that is, they assume autoregressive behavior.

No version of the decoder appears to be able to differentiate the position, no matter what happens.

THis is even true of the original decoder input as described in the first paper (Attention is All You Need).

~~~~~~~~

Next steps, over days:

X 1) Check if transformerEncoder has the same issue or not (my money's on 'not')
     -- it does!
X 2) Look carefully at the math and the code. Understand exactly why (I think I know why, but get the exact value)
     -- I think it *can* detect things only from itself, *but* it's not great at it.
     -- It's best at combined influence of all other tokens.
X 3) Make a custom decoder with a pass-through layer, like those old visual systems.
     -- pass through added for the entire encoder and entire decoder
     -- the cross terms will come in useful later.
     -- I may play with the exact way this is encoded next
~ 4) Redesign input / output, possibly with only one input / output and unique masks (though I'd rather have separate output decoders).
     -- unnecessary; pass through used instead
X 5) Run training.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 4, 2025

See above for notes
Added pass through path
Launched training


PS I'm going to have issues using the transformer to modify the correct patch input point.
It will have great difficulty knowing where exactly it is.

Maybe I should try training the thing to recreate the image, but with a dead square, or a flipped color agent, or something.

~~~~~~

Priorities:
X 1) Check that it can train at all
2) Look up the 'missing patches' visual autoencoders one more time. Add some version of those structures
3) Try adding some conditional dead pixels or something.
4) Once that works, think about image embedding one more time. Maybe add the 'multiple scales' things after all (16x16, 4x4, and 2x2 all queued up).

Damn shame, but it looks like this business will still need some more architectural input before I move on to the main task.
I familiarized myself with the standard transformer, but not a visual transformer, and for sure not a visuarl autoencoder.

This shouldn't take long, though. I think I'll have at least one of the 'fun tasks' (eg draw a path) starting training before Argentina


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 6, 2025

It trained!

*v3* is the correct (minimal) functional version for the image autoencoder

Saving on the behemoth, and doing some more work on the small machine to make sure that it can learn information from neighboring image blocks.
From there, I may need to retrain, or may need to finetune, depending on the architecture changes.

Possible changes:
1) Better 'location' codes
2) More skip connections
3) Location codes that reflect 2D space.

~~~~~~~~

On small machine:
clearly it's still failing at it.
WIll rewrite the image autoencoder code followign this guide: 
https://medium.com/thedeephub/building-vision-transformer-from-scratch-using-pytorch-an-image-worth-16x16-words-24db5f159e27
https://medium.com/thedeephub/building-mae-vision-transformer-from-scratch-using-pytorch-masked-autoencoders-are-scalable-2c2e78e0be02

~~~~~~~~

***I FINALLY HAVE A VERSION THAT WORKS***

Sort of . . . I killed the decoder and had to reimplement the encoder from scratch.

Tomorrow:
1) Try with the built-in encoder and same training parameters, just in case
2) Rewrite the decoder from scratch; use it.


Weekend:
Save everything, merge all branches, rerun all the pretraining. Plan the first task.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 7, 2025

It worked

DEBUG_MOVING2000 works, and works with this version of the repo.

Will now try to keep modifying / experimenting

~~~~~~~~

Transformer encoder works!

On to adding a minimal functioning decoder. May God have mercy.

~~~~~~~~

1 decoder layer works, but that's really only one additional encoder layer (with how it's written).

Trying now:
X 1) 8 encoder layers, 8 decoder layers, bigger heads (6 heads instead of 12; 2 times bigger).
X 2) 12 encoder layers, positional encoding v2, 12 decoder layers.

~~~~~~~~

I tried that. Neither worked.

It seems I really should understand exactly how the decoders work, and figure out the training program.
 -- only ones that worked: 12enc 0dec; 12enc 1dec (worse).
 -- all others got to at least 3000 batches with no results at all.

*Tomorrow:*
1) Custom Decoder; Custom decoder testing (at least once)
X 2) Experiment until something works with the decoder. Keep it super shallow if you must.
 -- solution found: 6 layers enc, 3 layers dec. v8 from the braing debug saga.
3) Think about how exactly you want to do this. 
 -- how to pretrain? Pretrain each feature, then only train the encoder / decoder stuff for the interactions
 -- train end-to-end in standard style?
4) Pick, then train.

~~~~~~~~

Some general notes:

I do think, piecemeal, I will be able to get each trick to work, training either end-to-end or on a curriculum
I may have trouble later gluing everything together. THis is not the order in which kids learn these things: they learn tasks before terms.

Another submodule / approach may involve teaching it to play the game, with RL.
 -- best approach here is to start with straight-line games only at first. Do it with just 1 gold and no walls first, and keep it near the agent.
 -- then, build up the curriculum, teaching it the 'turn and chase' task.
 -- may teach task first, then description, or reversed. Either approach is fine.

These can be done in parallel or together.

~~~~~~~~

HELL YES! I remembered that the decoder has 2 MHA layers for every decoder layer. So, I aimed for 12 total MHA layers. I left the number of heads alone.

That fixed things: 6 emcoder layers, 3 decoder layers, trains like the old encoder-alone layers.
This is *v7*.

X Next: testing with text context as randn's; we will see what that does, whether it can still learn the right thing.
     -- complete victory.

~~~~~~~~

Final thoughts:
~ 1) Do try the custom decoder block. Good exercise to understand exactly what is going on.
    -- Kinda. Look at CustomTransformer_Experiment if you ever get confused again
    -- Just remember: the key and value dimensions must be the same (gets cancelled). The Q must be the same as output.
2) Make this neat. Merge the branches, clean up code.
X 3) THink about training procedure (end-to-end vs stages).
    -- Look at the ending of ImageAuto_MovingSquare and the v9 img_enc and img_dec
    -- Final approach takes longer. Not worth it. May freeze layers later, but not yet.
4) Launch pretraining(s)

I have a great system, moving forward.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 8, 2025

Kind of a lazy day, but enough to close the chapter on the 'debugging' phase.

I understand the decoder know. I understand why cross-attention is so weird: all the 'values' in the final version come from the second vector (eg text values); 
the main input only produces the queries (the keys are also from the text).

However, the decoder has both a self-attention and a cross-attention stage, and also skip connections. Therefore, this will do.

This 'weirdness' is probably another good reason to keep the image decoder shallower than the image encoder.

~~~~~

I have determined that pretraining the patch encoder / decoder layers and then freezing them does not necessarily make things faster / easier / better.

I will stick to end-to-end for now, with possible layer-freezing tricks in the future to avoid catastrophic forgetting.

~~~~~

Tmr:

X 1) Actually make this neat.
X    -- Merge branches, 
X    -- merge all weight recordings.
X 2) Launch both pretrainings alone (possibly all on behemoth). Autoencoder only for valid position.
X 3) Start thinking about which next to set up and how to set it up. If pretraining is done, start that.

Monday: 
Check performance; start setting up next task.

Tuesday: launch next training.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 9, 2025

Launched both pretrainings, both on behemoth (though I'll have to keep the other one on. Too bad).

Modified the text decoder to be a bit more shallow, perhaps also transferring the lessons from the image transformer saga.

Manually walking and mounting the KINGSTON disk is annoying. I think I may write some scp tasks

Next big task: any of the tricks (detect corners, move left, move right, looking direction, path to gold in empty game, anything).
 -- I will do the naive approach first: no additional pretraining, only throw the task at it, maybe with some 'reinforce the old stuff' batches mixed in.
 -- Fails / takes too long? Will think of smarter things (teach the shapes themselves as part of autoencoder first, etc).
 -- Task will reproduce the image in most cases, but make it modified in some cases.

~~~~~~~~

This week:
1) Tests for the corner task on penguins.farm
2) Look at pretraining results; draw conclusions.
3) Make a position-sensitive 'moving box equivalent' test for the text encoder? Run it on penguins.farm?
4) Starlink, etc.

~~~~~~~~

Tmr: 
X Pretraining will still be running. Check on it, but don't sweat it.
X Maybe focus on the boat thing and chess instead, then actually work? Break from hobby? Food for thought.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 10, 2025

Text pretraining going as expected. Will kill it at the end of the day, or maybe tomorrow mornign.

Image pretraining again failed to learn positions.
I'll give it another 24 hours. If it still fails, I will retry, training the conv layers first, and using the full encoder second.
 -- restarted it with lr 0.0001, training v5.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 11, 2025

Image Pretraining had actually produced good results, but it suffered some lapse in judgement and fell back to high losses.
Retraing, saving batch numbers as well in case this happens again.

~~~~

The 2080 isn't reachable right now. I think penguins.army needs another hardware debug, or at least a power cycle. Will do after the image pretraining is complete.

~~~~

X Running quick test on the text module, on penguins.farm
  -- test successful; the brain easily learns to use positional encoding

X Will probably rerun text pretraining on penguins.farm (the 2080 death had killed that).
  -- running now

~~~~

Tomorrow:
X -- check on all pretraining
 -- done pretraining? Sync all network weights and git versions.
 -- set up corner task

Once image pretraining is done, I will look at the results and decid if it's good enough to continue.
I may spend all day Monday (day off!!) to set up that task properly (maybe in several versions) and start running it

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 12, 2025

The pretrainings are going decently, but I want to see if I can get more out of it.
I restarted the img pretraining, starting with a decently-performing batch and continuing with a lower lr
I interrupted the text pretraining, but let it continue with a new moniker. I'll have both an overfit and just-right version after today.

That is all for now. Syncing representations and code can wait until these stages are done.
Same with the necessary power cycle of penguins.army.

~~~~~~~~

Evening:

pretrainings still moving decently.
Will let everything go till morning. May relaunch the text pretraining stuff with a lower lr.

THere is a lot of room for improvement, especially for the text encoder.
The '32 symbol' thing is really going to be limiting in the future, but for now, it is freeing: can pretrain with absurd batch sizes.

Alternatives involve different text databases, or padding out to some minimal size like 512 or 1024.
But the pretraining already done can be useful.

In the future: I may try to transfer the learning of this network to one with a seqlength of 256 by just chaning how the positional encoding works, 
and teaching it to imitate the output of the current network.

THAT IS LATER, THOUGH.

Initial work will be the 'tricks'. Especially navigating / drawing lines / describing scenes with no walls, or drawing corners.

Other later work: retrain by adding the RL element as a separate stream (actions). I like that a lot.
Add some way of gathering state (embedding timestep and adding into some shared pool LSTM style, perhaps? Or just copy whatever Foerster did?)
Then plans, then following plans, etc.

But tricks come first. I have a good curriculum for 'detecting reward and navigating towards it'.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 13, 2025

Pretraining continues, slowly. Image loss is not going down past 0.004, but that may be enough. Or I may need to switch to a larger resolution.

Same for text pretraining.

I'll let it got for another day.

i also wrote random_bare_settings in discreteEngine, for the first 'trick'. Will debug and test that tmr / next week.

~~~~~~~

Tmr:
~ 1) Kill all pretraining no matter what
     -- I changed my mind. I'll let it go to the end of this weekend. See how low it can actually go.
2) Sync representations (code and weights)
3) Power cycle penguins.army. 
   -- if the 2080 does not come back, do another hardware debug.
4) If time: Make another jupyter notebook. Debug the new game code; write the new task, or most of it.
   -- this can wait till Monday / next week.

~~~~~~~

I killed the Text Pretraining.
Image Pretraining keeps inching down, though much more slowly than I'd like. Against my better judgement I'm giving it another several hours to impress me; will kill it in the morning.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

V-Day, 2025

Good day. I'm letting ImagePretraining continue over the weekend just in case.
I debugged the random 'bare game' generation, and added a way to create a line from the agent to the gold.

I will write the full 'trick' code right before I launch the training. That's first priority Monday. Everything else can wait.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 17th, 2025

Finished image pretraining.

Use either v6 or v10 for image autoencoder (v10 may be sensitive, v6 may be too fuzzy).

Simple 'bare game' with text outputs and simple modifications (arrow, etc) are next.

Synced checkpoints and git accross machines.

~~~~~~~~

UPDATE: penguins.army rebooted correctly, with both GPUs. No need for HW debug tmr (probably)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 18th, 2025

Took forever, but I launched the first 'trick' tutorial: drawing a line to the gold, when prompted.

Only image outputs for now. Next is to train image and text outputs.

Need to get the LR's done correctly for this to really work.

Will get this to work, then duplicate for the other skills that I want. I think I will have a dozen solid tricks within a month.

~~~~~~~~

Next:
X check results
X change printout (show both errors separately, for calibrating the constants)
~ success? write next trick (or several)
  -- mostly. Next trick waits, retraining hoping for better results
X failure? think of solutions (simple, like different pretrained version / loss ratio / learning rate, OR complex, like architecture change / identify blocks / additive outputs / more complex pretraining steps)
  -- thankfully this is avoided.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 19th, 2025

The 'v2' version did great. It basically looks like a child's scribble.

I moved on to v3, with more printouts and some losses to make sure that text prediction isn't destroyed either.

I'll probably kill this tonight. I may relaunch this as a training script and monitor it during my trip.

I'll write the next several tricks when I'm back. I now have a structure and a proven method; the rest *should* be straightforward.

I think next week is going to have great results. A trick a day, or a trick every two days.

I'll also look into training separate things on the different GPUs. I have 3 available, after all.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 21st, 2025

I accidentally deleted brain_v2 :(

I can't do much while I'm far from my instruments. I can't look at the jupyter notebook.

For now, I will only delete older versions of the v3 brain. I won't kill the process at all; worst case, I may have to rerun this all, but i'm not scared.

When I get back:
1) Turn the last jupyter notebook into a script to launch / monitor remotely.
2) Launch the next thing, with 'is it right / is it left' and maybe 'zoom in please'
   -- I will think exactly how to do this. May use additional input masks.

Cool, plan is definite

~~~~~~~~

Sidenote: I'm letting this tutorial train for a very long time, but it's worth remembering that I had decent results within a day.
Future tutorials will be more optimized; each one won't take this long.
 -- use separate lr's for diff components; keep parts frozen
 -- try to use bigger batch sizes for any text task. Maybe multiple text batches for the same image batch? Write thinking of the compilation.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 22nd, 2025

Not much. Glanced at progress. Wrote 'business_plans'

Tomorrow:
~ -- write the 'extra mask' code, no matter what it takes.
     -- looked at it and found a better way.
 -- if time, start writing the 'train text output' code.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 23rd, 2025

I looked at the brain 'forward' method, and looked at how masks work.

I've decided on the technique for the Q-A tasks: I will modify the 'forward' method
to allow for different text encodings to be used for image reconstruction (only the questions) vs 
text reconstruction (full input including answer, but causal mask helps that).

I *might* be able to do this instead with fancy masks (different src mask vs target mask) but I don't think it's needed.

So: tomorrow, 
I kill the current experiment, and 
write some of the code for the Q-A tasks, ideally in a script, not ipynb.

From there, I'll either launch the new task (once the code is done) or play with the results of the first tutorial, whichever fance strikes me.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 24th, 2025

Killed firstTutorial, backed up last checkpoint to shared disk.
Results are pretty good. The blue line is still slightly wavy, but very nearly usable now.

Tomorrow / this week: I will stress-test this checkpoint (text generation, text prompt versions, add a wall in just for fun, etc).
 -- delayin this by at least a day

Next step right now: write code for tutorial 2 (answering image questions)

~~~~~~~~

I wrote qa_forward for tutorial 2 (accepts text with and without 'answer').

An alternative would be to train tutorial 2 using RL; I will get to that, too, after a simpler RL system.

Tomorrow:
Debug qa_forward
Write, launch tutorial 2 (can be jupyter again, fine).

Other days:
1) Debug firstTutorial results; examine
2) Write some RL thing; launch.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 25th, 2025

Wrote most of the code for tutorial 2.

Use the 'simple' text generators that only create output, and only train the text encoder / decoder elements; leave the img weights unchanged.
 -- only use the qa_forward technique when you are also reconstructing images (and need only part of the input available to the reconstruction)

Tmr:
X 1) Learn how to load the state dict even if it was saved on the wrong device
     -- map_location = device
2) Write the training loop using the 'simple' method and without any image reconstructions. Choose which GPU to run it on, and start it.
3) Done!
4) If you want, write tutorial 3
5) If you want, look up the ssh stuff and port opening; come to dad ready to talk on Thursday.

Rest of week: think of how to launch an RL task on a different GPU.
Maybe start a new git branch, with a separate 'action' buffer.

~~~~~~~~

Minor edits and code testing in the evening.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 26th, 2025

penguinsarmy, launched tutorialQA on the 2080.

Will check / kill this evening.

Saving, will write tutorial_RLnaive over on penguinsfarm.

~~~~~~~~

Switched to new branch on penguinsfarm, RLmania.

Unfortunately, this will really take a few days.

I *could* probably write an RL system in an inspired morning, but I failed to have an inspired morning this morning, so I'm only going to split this into smaller tasks.

Next several days:
X 0) Modify tokenizer, tokens 1, 3, and 4 (make them left, right, and forward; avoid 'backward' for now) 
X 1) make PPO helper work with the game (call it 'game wrapper' or something, in a new file):
X    -- 'seed' is just the Settings file; copy is stored at every interval (but not full image, that's obscene)
X    -- token is generated conditional on all that came before
X    -- all the normal calculations, just like PPO_helper
2) Rewrite the RL notebook in the main place, which works with this setup (it won't train; just make the shell work).
   -- have the book open in front of you as a reminder.
3) Function to make fake traces (brain won't stumble towards the correct actions any other way)
4) Start playing with the *simplest* setup, and see if it can learn to crack it.

This will be the 'naive' RL. This ignores text replies and other sophisticated interactions. This will just be "see, turn towards, chase, eat", learned through RL (multi-stage, enforced by me).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 27th, 2025

Finished training.

Will check results later. Full focus on RL for now.

~~~~~

Slow day. Basically had to rewrite the bufffer code

Wrote the RL_helper file, fully.

Tmr:
X debug this file, using an untrained brain (random outputs).
X Then, find a way to write fake traces and see how that goes.
From there, launch a real training funciton.

Delete PPO_helper when done; a legacy copy exists in the Florence folder anyway.

Should launch real training tomorrow.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feb 28th, 2025

Busy day

Fully debugged the RL helper and solved the 'fake trace' issue quickly.

Monday, I'll 
double-check the trace generator in RL_fake_traces, then 
choose some fake log_prob's and entropies (pick one from a 'dist'?), then 
write and launch the RL code on top of a brain sample.

Notes:
1) May need custom value function to avoid propagating gradients through img_enc (wasteful; too many images)
2) Remember that generate_log_probabilities_and_entropies needs to be the new one, not the old one
3) Otherwise, can mostly copy the PPO notebooks, and launch the loops
    -- may reduce gamma and tau.
4) Launches and runs? Results? Delete PPO_helper folder
Will launch for a day or two with pretrained brains, see how it goes

~~~~~~~~

At least for the hobby, this was a very good day

Next week, after this launches, I will largely sit back and run a bunch of tests on all the trained systems.
I will run 3 tests on all systems (visual on the P40, linguistic on the 2080, and RL on the 1080, at least while it's still a toy).

Then I may retrain / pick specific next tasks / etc. But coding will be slightly more relaxed again for a couple of weeks, probably including the Argentina trip


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 3rd, 2025

Bad day

Launched a 'fake traces' version for RL, that's it.

Not really even using RL, unfortunately . . .

Tmr:
 -- kill it
 -- launch 'actual RL' with no fake traces on the big machine
 -- maybe launch a version of this one with no 'action memory', only pure image processing
    (I'm afraid of an error mode where it will fail to start turning in the right way and fail to switch to the forward action, but will know to keep an action going)

That's it.

The rest of this week is looking at what I've produced so far and recalibrating.
Maybe I'll write a small 'helper' script with all the most useful functions / the fake trace stuff / the code for the language training, to reduce code in notebooks.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 4th, 2025

Fake traces had an error. Launched a better version.

Launching a 'real' version over on penguinsarmy now. 

Tmr: kill, switch gears into debugging the several results so far.
     -- for RL, a good first start will be just creating a single, universal prompt, not feeding a trace. On both machines. Maaaaybe.
     -- UPDATE: the traditional approach has a memory leak. Fixing it comes first.

Then, sync all these, merge branches, debug the other tutorials, etc.

~~~~

Once it's debugged and I have a single brain that does every silly trick I've tried so far, I'll make a short list of the next tricks.
(no corners, but add things like 'facing toward', 'facing away', 'to the left', 'to the right', and a couple others).
ALso, I will standardize the code from the notebooks and make several re-importable python files to use.
 -- objective for March: convincing NB which shows the agent 'knows' how to find gold in an empty room, and can talk about it reasonably.

THe goal after those is to have a system that usually chases the gold, and can explain a few things about it.
From there the goal is to imagine different counter-factuals and maybe using them intellectually.
I will also train up the ability to 'summarize' at this point. I will possibly modify the brain.

The big reveal will be 'corners'. That comes next. ONce I have that task, I can introduce 'plan' and the code structure needed for it.
From there, it can move to other rooms and discuss them.

THat's a *very* advanced robot. ONce it's there, I can absolutely move on to 'teach yourself new images' by remembering images and running small training loops.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 6th, 2025

Finally fixed damn memory leaks (I think) for RL_traditional

Launched *a* version today. If, by tmr it doesn't finish training, will launch a better version tmr (on the 1080),
and launch a version with fake traces as a guide over on the 2080 or the P40.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 7th, 2025

RL_traditional launched correctly, but later failed for some reason.

Relaunching with higher batch size (16 instead of 6) and high hopes. We will see.

I am also launching a version of it (still on penguins army) with one 'fake buffer' per buffer-buffer, and seeing what that does as a guide.

~~~~

Merged a version onto penguins farm
Also launching a notebook on penguinsfarm, letting the radius grow over time

~~~~

I did the hard work of writing the RL scripts, but *really* I need to debug them as carefully as I debugged the vision system. I don't think any of my variants are actually training right now.

I'll do it after I look at all the other results, in great detail, however. Next week is "examine the agents" week. I'm a bit sick of RL, especially RL learning alone.

There is a lot of conceptual work to do in order to get the RL to behave, but it's good that right now, I have all the code down.

The value function is particularly bad for some reason. I'll check later, but I bet I also have unusualy low entropy.

I will need to check out how exactly I fixed the RL system back when I was only testing it out.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 10th, 2025

I'm back from my trip!

Against all hope, it looks like the 'curriculum' version did learn a little. I will let it run for 24 more hours.
I suspect it's a matter of 'the transformer needed initial pre-training to get anywhere'.
THe NLP and vision systems benefitted from pretraining in a way the dopamine module did not, after all.

I'll have to learn how to use words like 'good' and 'bad' to automatically train the dopamine portion. But that's a long way off.
That's in 'make your own training loops' territory, and that has to wait until I have a single, standard OOD loop.

~~~~

Who knows what's going on on the monster. Firefox chose to be awful and 'crashed my tabs'. That could mean anything.

I'll maintain the status quo for 24 hours, then reboot. The P40 is still fire-breathing which suggests that the notebook is still training, though with what results, who can tell.

~~~~

Tmr: housekeeping.
X Kill all notebooks, maybe glance at traces on the 1080 but nowhere else.
X Reboot Frankenstein and see what the status is.
  -- it recovered fine. I just have to keep in mind that GPU 0 is unreliable
  -- will reboot at least once more before the big trip.
X Open / close firefox.
~ Backup best versions to the the disk. Sync branch representations.
  -- for that, I need to find the best versions. This waits
X ***merge the RL branch into the main branch***
X -- delete the 'PPO helper' folder, recreate with RL notebooks / files (use ln -s to make links)

Week:
1) Check all results from all notebooks.
   -- language
   -- ALL RL (supervised, curriculum, semi-supervised)
2) Debug RL 
   -- maybe make alternative system which uses the entire buffer-buffer for signals on how to behave. Use averages, deepseek-style, instead of learning a value function.

~~~~

I was slow today, but that's because the goals were vague and firefox / penguinsarmy were uncooperative. Tomorrow, the goals are specific: housekeeping.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 11th, 2025

Housekeeping done.

Tmr:
Examine the 'language' notebook (Tutorial QA, on the behemoth), and draw conclusions.
Maybe also examine one or two of the RL notebooks.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 12th, 2025

TutorialQA did very well. Supervised learning does just fine with varieties of outputs, especially when the concept it needs to detect is so straightforward.

I could have ended this after only 10k batches, by the way, or even 2500.

I am keeping batch7600, however, as the best overall, and most trusted.

Keeping 2500 and 7600, deleting all others as unnecessary
 -- AAAAAND I accidentally deleted it. FML. Will rerun.

~~~~~~~~

Wrote pseudo-GRPO code instead of PPO, to avoid problems with value functions
Will see whether or not it trains in the future. THis is how I'll learn RL / what I'll use as a default. Value funcs can come much later.

~~~~~~~~

TutorialsQA relaunched. GPU troubles again. It seems I can only use one GPU at a time.

I lost the evaluation code in TutorialsQA, but it's easy to rewrite.

It seems that using both GPUs at capacity on the behemoth will not work. I will only train on the P40 during my Argentina trip; will fix upon return.

Tonight: rerun TutorialsQA
Tmr: relaunch the new pseudoGRPO code; check on it during the trip. Maybe launch it as a script, not as an ipynb.

~~~~~~~~

Made evaluation notebooks for Image (some capacity lost; will need to be restored, perhaps in stages) and for RL_fake_traces.
Caught a bug in RL_semi_traditional

Tmr, when launching RL_semi_traditional: be super careful. Perhaps launch from a python script instead of nb, but that's not strictly necessary.

~~~~~~~~

Tmr:
X -- pick the TutuorialsQA result you want. Share onto both machines.
~ -- reboot both machines in preparation
     -- no need, I just shut down the TutorialQA train loop
X -- launch pseudoGRPO, v1, carefully. THis will be the last time you touch it before the trip.
     -- yep, in new file, copied exclusively from the python notebook

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 13th, 2025

Reran TUtorialQA. I kept batch 6100.

Lesson: this approach will work for other questions. Also, there's no need to 'baby' the agent when it comes to exact answer format, etc.

Launched the pseudo-GRPO with even more simplifications, on the big machine (P40). Will see how it goes.

Tmr / Saturday:
1) Check progress. Maybe debug if does not work (do step by step in new notebook, like what I did back for the vision tutorial)
2) Sketchbook work (Mermaid editor?). Pick a structure / brain behavior system without too-long logs. Stuff where details can be trained, but a default 'interaction with env' mode is very clear.
X    -- this 'interaction with env' mode can get more bells and whistles later. I want, right now, a basic system with only neural (and maybe persisten visual canvas) memory. Text banks and files come later.
X    -- reread the Foerster RL paper. Try to use as a guide?
            -- does not work. He uses the Markov property to make reactions a function of present state only. No memory.

~~~~~~~~

Looking more closely at fake_traces, it really should have worked.
Debugging it may yield the highest return on investment. Make sure you're computing what you think you're computing.
But this can wait for the other projects.

I don't think that the 'semi-traditional pseudo-GRPO' thing will work. The off-policy instructions from the guide are too different from the on-policy distribution.
THough maybe the probabilities for the correct actions will go up . . . will see. It has a couple of days.

Fake_traces must be debugged first, however.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 18th, 2025

"Semi-supervised RL" finished (or rather, broke; ran out of disk space). I deleted a lot of the intermediate save points, now order is restored.

Will not test it right now due to physical weakness (food poisoning). Will do that as one of the 'tasks that neeed to get done' once I am sane again.

Good progress creating final brain design with memory. I like the 'fixed remember / fixed forget' gate idea. It's probably not optimal, but it'll do the trick
and it's fully understandable by humble old me.

The papers mention that I am writing a 'full list of tasks' for the agent before I move on to bigger tasks (like self-curated lists, special actions to store 
plands / info tidbits / load files, corner learning using all the tools so far, etc). This file is called 'full_task_list.md'

Quick note: 
A lot of those tasks will require 'creative loss functions'. It will call for imagining something in  arough, not exact, position.
One thing that could really help would be 'get settings from image': use things like center of mass of yellow, red, and gold to extrapolate the Settings object.
This can be a '1 gold' or 'multi-gold' task.

I think that writing some helper functions for that will be useful this trip, dunno with which priority.

Big projects this trip. How much I complete depends on physical health, primarily.
1) Debug RL, relaunch
2) Make a GUI visualizer for the current state in the agent (except memory, maybe): all canvases, current text, input image.
3) Make the 'settings from image' func. Test speed / accuracy. Use it for 'creative loss funcs' later.
      -- if it's 'visualize gold near upper left corner', then make the loss 0 within a chunk of area, and reward variety in output batch.
4) Lots of notebooks (NOT RUN) written in a mixture of python and pseudo for each of the tasks on the other page.
5) ONLY THEN, write the new brain (maybe new repo) and start training memory (probably use transfer learning from current brain).

Tmr / Thursday:
1) Double-check the list of tasks. Double-check the structure, and that it will be able to do everything you are asking of it.
2) Find ways to do GUI (too sick of RL, unless I am in perfect physical health).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 19th, 2025

I made some notes in full_task_list. Won't repeat here.

At best, this looks like 6 months to a year before I'm done with the game and move on to the 3d robot and self-guided exploration.
That's at current paces, however. Buying more hardware for training and quiting to work full-time may make this go faster.

Good news is, I'm fully recovered physically.

~~~~~~~~

Afternoon / morning: make the 'image to settings' script. Test this week.
Then the other tasks.

Alternatively, test out some other options, like retrainiing lava, in another repo. A 'refocusing time' like this break is the right time to do this.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 20th, 2025

I wrote a python notebook called DoNotSave_PreludeImageToSettings

There, that task is mostly done, except for anything to do with walls.

Tmr, I will finish this, except for walls:
X 1) Test the 'multiple gold centers' thing
X   -- probably each step needs checking, what's x and what's y. THink through it.
X 2) Write the funcs to process image batch instead of image alone (only matters at the end).
X 3) Write the actual python file (use bare_settings=True and copy_walls=True as default, and default device as 'cpu' but can be set easily).
X 4) Delete the ipynb.
~ 5) Test this on some agent reconstructions. Play with epsilon. Pick one. Probably 0.01 is best, no?
    -- canceled this; this can wait until this file is used for loss funcs

That's probably it for Friday. Plus or minus serious planning.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 21st, 2025

FInished the image_to_settings fun. May do several other tasks if today is an intensive.

Updated business plans with conviction.com info

~~~~~~~~

Started widget exploration. Not done. File WidgetExploration.ipynb

Buttons often do not print, but I finally found code where they do print. They can also control the Output widget (eg by clearingit; see notebook)

Widget tutorial sequence: https://ipywidgets.readthedocs.io/en/8.1.2/examples/Widget%20Basics.html
Widget list: https://ipywidgets.readthedocs.io/en/8.1.2/examples/Widget%20List.html
Wdiget Output documentation: https://ipywidgets.readthedocs.io/en/latest/examples/Output%20Widget.html

Tmr:
X 1) Opent the widget list link above
X 2) Run the ipynb
X 3) add image output with refreshing and the game object
X 4) Build this:
X    -- text box
X    -- button to 'submit'
~    -- pseudo-agent present.
~    -- if "FORWARD", "LEFT", "RIGHT" or "BACKWARD" are present in the submitted text, that action is taken in the game
X    -- The output is cleared, the game display is updated
X    -- a randomly pixelated image is generated and also displayed by the pseudoagent
X    -- whatever the input text was, it's reversed and printed by the pseudo-agent.

That's enough for 1-2 mornings. This system, almost bit-for-bit, will work for the agent (user input, one game step, agent output and print, several image canvases)
This will actually let me see the agent in its env
Some sort of 'keep stepping for N steps' button can be added if I feel like it.

This is either tmr or Monday finishing. Then I will be done with widgets.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 22nd, 2025

Amazing. Literally 5 minutes of work.

I skipped the use of the game and the agent brain, but it's easy to input text, display text, and display matplotlib plots as much as I want. All of that is in the notebook.

Right now, the sequential display of buttons is ugly, but once I make the system, I could change that if I want:
https://ipywidgets.readthedocs.io/en/7.6.5/examples/Widget%20Styling.html

Done with widgets. Done with 'image to settings', whether I use that as a differentiable loss or for RL.

Monday (tmr?): lots of RL. Full days. Go through the training scripts. Understand everything that happened. Print critical values and gradients. Make sure they're all correct.
RL debugging week is nigh.

If I am done with RL debugging week, I could make an NN version of image2settings, or test image2settings on brain outputs.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 23rd, 2025

Started debugging. Most notes are in the notebook.
Running notebooks on penguinsarmy? remember to use 'penguinsfarmForwardPorts 8888 8888' on the laptop.

Main error I see so far: fake_data_fill does not store the correct settings (only multiple copies of the initial state).
I will fix this first, but then make sure everything else is as I expect, too (don't just rerun right away).
X  -- fixed! Next step, tmr, is adding options to skip GAE computation in the logic solver, followed by 
X  -- going through all the other steps to see if they make sense.
X  -- once I fix any other egregious errors, optimistically, I can retry training Tuesday evening or Wednesday morning.
  -- if *that* goes well, I will also fix PPO and rerun it, and / or traing the dopamine func.

One of the traces round 10 showed an attempt to turn then go forward. This suggests that, with the correct data saved, this will actually work.

I will basically go through and test every element of the training loop, making sure it's what I expect, then relaunch training Tuesday or Wednesday.

Done early? Maybe also carefully test the images_to_settings code (at most one day).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 24th, 2025

Debugging goes apace
There's the question of the "+1" in log prob computation (compare with HEAD).
Weird results in the 'guide' case. the masks should have removed the issue but they didn't.
Fix, then run.

~~~~~~~~

I think I finally figured it out. I am getting log probs that look correct in all cases. I'm pretty sure I fixed it.
Rerunning at v3. Will check in the morning.

If good:
 -- still check past_terminated
 -- launch non-guided or even PPO (or both; two machines after all)

If not good, keep debugging (except now focus on learning rates, etc).

~~~~~~~~

If satisfied with RL:
1) Recheck image2settings with reconstructions from agent
2) Make differentiable settings2img? For fun?
   -- I have *something* like this. Will check
3) make version with memory? Start making next repo? Basically, act on the other plans, now that RL is finally also in the toolkit.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 24th, 2025

Better, but it still didn't learn.
It learned to spin rapidly, then go forward.

The later rounds were worse than the earlier rounds because it kept trying to kill the round by pressing '2':
the rewards were misaligned in a way that made this look like a rewarded action.

I relaunched with correctly aligned gae's (v4). Hopefully this is enough.
If not, I will consider not feeding the trace in (to prevent it from learning 'spam the action you took last move', which is not the lesson I want).

If works: see above for next steps.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 25th, 2025

It was not enough.

Oddly, self-terminating did not fully go away as a habit, even though its GAE was always strictly negative.
Maybe, in certain contexts, the 2 action was not pushed down as much as the others? Say, after you've spun and it's already hopeless, the '1' and '3' actions
got pushed down 10 times per (non-guide) round, but the 2 action only once?

It doesn't really matter. I relaunched it without feeding in the trace, hopefully forcing it to *properly* use the rich image information.
If that doesn't work, I'll do a post-mortem and move one. This is no longer a mysterious black box; I understand what RL can and can't do and why it's doing 
what it's doing.

It'll be *so* much fun to get a full agent to go for the gold after talking through all the reasons (I need to turn toward it, which is left right now, then go forward).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 26th, 2025

I almost killed it, but round 230 had slightly-above-average returns (like 0.128), so I'll let it run for another day
I'm almost certain I see a spike in performance before a drop-off.

Next things:
1) Debug image2settings. Take that RL break
2) Rerun, but save the traces. Try to see if it 'starts learning then fails' or not. I suspect it might be
3) Debug from there? Flip a coin, I have other priorities.

On the trip and probably next week, I don't have much else to do.
Big day (or a happy morning):
 -- new branch, build agent with memory
 -- transfer the weights somehow; save
 -- train some recall and see how it works

Once I have that, I'll make a proper notebook with widgets that uses this agent.
Then I will make that new repo and start knocking out a tutorial a day (see full_task_list.
 -- remember to make them in a way that can be imported. Remember to run train loops that include prev. seen tricks, maybe not as often as the new trick
 -- now that I know what I can / can't do, I can take more time here and not train right away. I need to write a ton of these and not 'shake from fright' after each one during training. Take several days.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 29th, 2025

It still did not really learn, though now the failure modes are less bad, and success is like 1 in 16 or 32 traces (not rigourous sampling).
At both early and late epochs, v5 can't decide which way to turn, and wobbles back and forth.
Looking at a succesful trace, one almost thinks it learned the right steps, but looking at a failed trace, it clearly doesn't know what to do.

Round 5 looks slightly better than all others (early and late), but the data sampling was not rigourous.
Later rounds seem more 'sure of themselves' (consistently turn in one direction, longer), but seemingly without great jumps in performance.

Maybe running longer, or with different time discounts, or on a curriculum, or lower lr, could have resolved this, but I don't need that right now.
** I'm most curious about repeating this after training it on words, with concepts like 'looking at', etc. I think there will be some transfer learning, 
even if I use RL.

~~~~~~~~

I'm rerunning with output to v6_training_trace (or some such filename) so I can check whether or not there is an 'early spike' in performance followed by a dropoff

I will kill this tonight or tomorrow morning, I do not need a full trace.

After that, if I rerun, it'll be a 'fake traces' style learning ONLY. Sanity check.

Next steps, as described before:
0) Fake traces run?
1) On penguinsfarm, debug image2settings
2) Big day (Tuesday? Wednesday?), make that full memory agent and start on that new branch /  repo.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 30th, 2025

v6 trace: there really does seem to be an early peak. Lots of sucess rates close to 16%, for a while, whereas later ones go down to 9-10.
Round 20 seems to have been the peak.

~~~~~~~~

Launched fake_traces script. Will see what that yields. Script written quickly, may need debugging.

Plans:
1) Monday, launch fake_traces, write / launch DDQN
2) Debug image2settings. Sleep in Tuesday.
3) Wed, BIG DAY! Day off work, but fully home. Write the memory agent. Maybe launch some training.

Various other algos for RL (DDQN, PPO, REINFOCE, A2C) are on the menu for mornings where I can't do memory-agent work.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 31st, 2025

Naive fake traces managed to fail. I used only logic-solver traces, and this led to only the most common action geting rewarded.

I launched a new fake_traces, using proper cross-entropy loss. Hopefully that is enough.

Next steps:
1) Check in on fake_traces
2) image2settings!! 
3) Tuesday: sleep in, DQN if up for it.
4) Wed, big day planned as before. Memory agent.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 1st, 2025

The new cross-entropy is close to 0.1 but is not 0. I will give it another day to run since I don't have any more time to work on it today.

Tmr plan:
X -- check in
X -- image2settings
X -- cleanup / syncing / records of progress
 -- DQN if fake-traces worked (new folder)
 -- other machine: finally, the big project

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 2nd, 2025

Failed day.

Killed training (it failed)
Same plan tmr.

Try to start the second branch, but don't cut work for it.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 3rd, 2025

Finally, fake_traces, the improved version (proper supervised training), learned this task.

Every RL and RL-like approach failed, however.

It learned by epoch 30,000 or so, so it should have taken a little under 4 hours. Still slightly worse performance than the guide, but what can you do.

RL fine-tuning might fix that. I don't care enough to try.

I do not want to set up DQN at this time. I have seen how PPO and GRPO start moving in the right direction but get stopped; I'm sure I could get them to work with some
tricks / guides / initial-burn-in / FEWER DECISIONS.

I won't touch this again until I need it.

~~~~~~~~

Deleted all except round400000 (best version) of the improved_fake_traces

Deleted all the *EXPERIMENTAL* files except for that one. RL failed, no need to store all of these different versions.

GRPO *does* slowly improve the correct weights, and can be used. I'm sure that *explaining* the RL mechanism, once 
the beast knows enough words, will yield far better results. (THis is also something to try once I try finetuning existing models).

~~~~~~~~

FINAL RL SUMMARY:

PPO, as written, simply does not converge.
If I try value functions again, they will be separately trained, or I'll use value-based systems like DQN (at least first), to have experience with value funcs that converge
pseudo-GRPO almost converged, by means of having a better value baseline. Still, it suffered from the fact that this is a difficult task and probably needed some noise reduction / pretraining / something else.
Fake traces finally converged. I deleted the version that didn't work, and only kept the one with the full, correct supervised learning loop
 -- shows that convergence for these tasks is possible
 -- shows the proper way to learn these situational moves, and how not to be lazy.

Will return to this some other time. First games will probably be 'strung together' from skills learned by supervised learning (call it instinct).

~~~~~~~~

Tested image2settings. Everything I needed seems to work well. Leaving the python notebook as evidence / user guide.

~~~~~~~~

Started branch memoryUnit. This will redo the brain
For now, all the new class details are in comments alone
Tmr, will write them one at a time, debugging code in throw-away notebook (make sure you *do* throw it away)

Should be done in 2 or 3 work days.
Then, transfer weights in some way (new class for old brain??)
Then, test, maybe write the widgets
If I'm at all productive, one - two weeks

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 4th, 2025

Wrote all the main memory classes

Will test their validity tmr, then consider some more, and write the brain 'forward' loop.

THen consider some more and write it with full img canvases and everything.

Sat / next week: make brain forward loop with canvases and widgets and things. Transfer weights.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 8th, 2025

Tested everything and it works.

Next:
X 1) Add this to a new brain class; save it with transferred weights
X 2) Write the forward loop
3) Begin transferring the other stuff

Biggest job will be this weekend.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 9th, 2025

First draft of new class written.

Next steps:

X 0) Separate file for the memory code
X 1) make Memory and Canvases into proper nn.Parameters, so they transfer devices along with the brain
X 2) change forward to allow either creating the entire context or not creating, with small steps for text-only generation (allows 'generate')
X 3) Properly write the 'generate' code as well
4) test all of this
X 5) transfer weights

This will be several days. Probably much of Saturday, too.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 10th, 2025

Good progress. Did most things except write 'generate' code. Will do that for the rest of the week. That and testing.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 11th, 2025

Bad day.

Only read this:
https://www.anthropic.com/research/tracing-thoughts-language-model

Associated paper:
https://transformer-circuits.pub/2025/attribution-graphs/methods.html#graphs-constructing

Interesting 'neuroscience' on how LLMs arrive at conclusions, and which input / intermediate / output features they use.
Pretty cool. The conclusions about how Claude works are to be expected.

Perhaps I will bring this in, later, to use for later-stage debugging, BUT this is not a time investment I want to use now.
Cool to see it in use for foundation models. Perhaps I will check the state of the art again this summer, and see if I can replace my model with something finetuned.
COntinuing current track because all the experience and much of the code I use will transfer quite well.

~~~~

Tmr:
X -- finish writing / testing new model, and transfer weights (see checklist above)
 -- write first 'memory use' task; run it (somewhere)
 -- time? start writing the others

Sunday:
 -- taxes
 -- drive around, find a boat place, use the boat place


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 12th, 2025

Bad day, but got some things done. FInished coding, transferred weights. Testing may need some work

Notes on generate:
 -- run 'forward' once in each context first, to generate self.context correctly
 -- run note the 'store memory' field. Notice it won't affect the self.context variable. Keep it in mind, set it properly depending on application

Tmr morning / this week:
1) Run some 'test img autoencoder' and 'test sentence autoencoder' code. Maybe run a training loop.
2) Done? Write proper widget wireframe for the enhanced model.
   -- OR, write some code to train the memory in elementary ways. (This could wait for later this week).

Goals this week:
 -- repeat in Enhanced Brain all the skills there were before
 -- train the memory for recall and recall only
 -- maaaaybe start the other tutorials in the list? Food for thought

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 13th, 2025

I copied the FirstTutorial and TutorialQA codes into 'framework' files, which can be re-imported and re-used.

Next several days: I will write a short loop that will both test and later be used to train EnhancedBrain (with memory) on these tasks.

Details:
X -- randomize the order of inputs in the TutorialQA inputs 
X -- only use the 'full forward' on the first input; the other four get to share context
X -- intersperse (include?) the firstTutorial tasks
X -- intersperse (include?) the original image autoencoder stuff (with control texts)
 -- don't be in a rush to train everything. A good day can be just knocking one of these tasks off this checklist

Plan for week / weekend:
X 1) Above 'sanity check' for EnhancedBrain
2) EnhancedBrain widget-framework. Full and complete.
3) Merge into the 'master' branch; make new git repo.
4) (probably next week) Knock off more items off full_task_list.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 14th, 2025

Started writing TestingEnhancedBrain python notebook
Tmr:
X -- use it for eval purposes
X -- add training capabilities; use it to run some finetuning on EnhancedBrain

Not much done thanks to taxes, but progress is solid

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 15th, 2025

FInally debugged yesterday's code. Wasted day. There were faster ways to do this. Should have done that, first. But I just couldn't wait, huh?

Tmr:
 -- check how other systems with memory are written. briefly. I'm doing something crooked, I think.
 -- add training to the functions in TestingEnhancedBrain, then save them with the 'framework' files
 -- no need for img_autoencoder or text_autoencoder special pleading. Those losses are part of the QA and firstTutorial losses.
 -- everything on-time? move to other server and launch a training loop.

Plan for other tutorials:
X 1) Write new training functions like the ones in this file
2) During actual training, cycle through them (or sample randomly), and chase all the losses together.
  -- later, I can add code to weigh likelihoods, but that's far from today.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 16th, 2025

Added training code but did not run

Tmr:
X 0) COnsider how to run (when to reset?)
     -- considered. NEed different remembered images, so allowed non-reset execution. We will see if this produces memory leaks.
X 1) Move to penguins.army
X 2) Run training, overnight

Rest of week: depending on results, either debugging or testing the results

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 17th, 2025

Noticed we must change the device in general_framework. Oh well. Bit of a pain.
Removed Tests4Memunit; ugly scratchwork I only needed for development. 

~~~~

We're back to VisionCanvases dropping canvases for no reason.

I think I'm done with using nn.Sequential.
Instead, I will use a new class (just a wrapper around a list of tensors), and try to use 'register_buffer' for the new class.
Will see if it works.

If not . . . I'll use external memory? And be careful to transport it between devices with the brain? Dunno, man, this is frustrating.

~~~~

Tmr: fix the canvas thing. Try rerunning training.
Next week: 
X -- fix training and run; 
 -- widgets for display; 
 -- next several frameworks, in this order:
     -- image retraining (don't forget how to zoom)
     -- using the memory directly (recollection task)
     -- then the other tasks from the list, in any order.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 18th, 2025

Two new versions of VisionCanvases attempted. Neither works.

New plan: Monday, I'm rewriting all this to include an external VisionCanvas at initialization, and moving both between devices manually.
Kinda ugly. Oh well. I might add a custom 'move_to' instead of 'to' which also sends the vision canvases. Some ugly solution like that.

I don't like that, but it's not very avoidable. Perhaps it's time to start working with external objects anyway.

NEXT WEEK GOAL: make the damn thing train.
After that, train the old skills, then basic memory, then new tricks and widget display.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 21st, 2025

Slow day.

So close to grad computations actually working.
Try:
X -- retain_graph again
X -- strategic ".detach" calls (make it part of 'reset'? Or as a backward hook?)

One of those two should work. Then launch proper training, finally.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 22nd, 2025

Added 'soft_reset' (detach) between batches (after loss.backward is called).

Damn thing finally training. Will check results later today or tomorrow.

FINALLY IT TRAINS!!

~~~~

For the next tasks, here is the checklist from April 13th.
I'm recopying it here and adding an element, to avoid scrolling up too much:

X 1) Above 'sanity check' for EnhancedBrain
2) Look at results. Retrain? Fix
3) EnhancedBrain widget-framework. Full and complete.
3.5) add training code to store the entire learning curve
4) Merge into the 'master' branch; make new git repo.
5) More frameworks from full_task_list
   -- start with more finetuning for image_autoencoder
   -- next, train the memory and vision-canvas using abilities
   -- start writing the other frameworks one at a time after that.


~~~~

Tmr plan, specifically:
X 1) look at training; see what went wrong / right.
X   -- failed to train? try different 'labeling' for vision canvas (or bigger amplitude, or limited to only last several dimensions, or something).
X   -- think about the losses. Estimate how much / what you'll need for specific results.
2) everything good? think about the first 'memory' task. Look at notes, look at full_task_list, etc. Pick an approach; maybe start coding.

Other stuff from the list (widgets? add code for full learning curve? think) can substitute #2, or complement it.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 23rd, 2025

Training was slower than desired; 10000 batches was not enough. Retraining with more opportunity to learn, based on last night's results. Will check tmr.

Done? Will write:
X -- basic memory training framework
X -- img autoencoder framework
 -- widget interaction ipynb.

~~~~~~~~

Made some new business plans.

Will make a decision about BASE club in Miami sometime in August, when it's more professionally relevant and when I have a better idea of my exact starting capital.
 -- for now, saving, not extravagant payments, are the law. I need to learn how to save.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 24th, 2025

It's still training. It's making progress, though it's slow, so I will give it another several days.

Training these two together is probably not the correct call, because the lr's are so different.
For next train sessions, will alter their relative frequencies, and generally focus on one training at a time.

Wrote two more frameworks: control and tutorial_mem_canvas. Both are untested so far.
 -- control for the standard recall of normal and zoomed games
 -- tutorial_mem_canvas for recall.

If tutorial_mem_canvas proves very difficult, I will have to change which vectors I use to 'label' the canvases (and start all Enhanced training over again).

I may need to restart exactly once more, for memory use (requires better autoencoders), but hopefully after that never again.

~~~~

All VC stuff can be left until a later date. I need my product.

Tmr / next stuff:
 -- check on training. Interrupt?
X -- widget stuff
 -- intelligent 'device' loading (force specification before general_training import)
 -- try a task with 'settings from output image' (two losses: 'gameness' of image and correctness of result)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 25th, 2025

Can't do widget work on penguinsarmy, pulling progress onto penguinsfarm and branching (widgetBranch).
 -- update the 'player' env on penguinsarmy.

~~~~

This is on the memUnit branch (should be master).
No progress here today, moved slow.

System still training, number going down, so I'm letting it keep going for now, though patience is for sure wearing thin,

Next time: 
X -- check progress; make training decisions
X -- knock out some more frameworks, especially any that require image-to-settings loss funcs.

~~~~

This version is on widgetBranch

Made a rough draft of all the functions. Moved slow.
Next tasks:
X 1) Make the buttons and test the functions
2) Make the buttons pretty.
   -- takes too long
? 3) Try to make an importable 'widget framework'. No sweat if doesn't work.
     -- not necessary
~ 4) Clean version, with only one mode of operation (import from the game, and updating the game when the right move is detected).
     -- not necessary

That is weekend / next week.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 26th, 2025

Training VERY slow. Weirdly, it's slower for the QA task.

I won't train 2 ssytems together again. There will be one main one, and the others will just be there to not lose the skill. One task at a time.
I had simply hoped for transfer learning, but that hope had been misplaced.

Tmr: if the losses are not below 0.0035 for the task on trial 1, and one of the QA entropies goes below 0.3, I will terminate this and switch to the '1 at a time' version anyway 
(and test the performance so far, qualitatively).

~~~~

Knocked out mem_enc pretraining framework (make it an autoencoder before anything else). Weird framework took mental effort to write, so I'm excusing that being my only achievement.

Next time I'm here: 
 -- a couple of 'image to settings' frameworks
 -- some other random frameworks, easy ones. More prompts to memorize ("learn").

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 27th, 2025

I figured out why tutorialQA is not working at all: tutorialQA.py has create_context=False for all but the first image. This is not correct: that causes the input img_batch to be ignored.
Training has been going slow for both tasks, anyway.
 -- or is that the case? I'm actually not sure. The img batch should be the same so . . . why re-encode it more than once?
 -- leaving it to train for now.

I've decided I'm not going to kill it until I want to check on the progress. Ideally, it can keep going for 2 more weeks or until I'm satisfied with at least one set of loss outputs.

In the meantime, I will keep writing new frameworks and also making the widget debugger.

~~~~

Progress: two more frameworks, one to use complex loss functions (extracting info from images), one to answer questions about relative (not absolute) position.

I marked all framework files that haven't been tested and need finetuning as 'RAW'. grep-ing for this will easily yield some framework files to try and fix when I want to try that task.

~~~~

Tmr:
No frameworks! Only widgets!
 -- finish rough draft
 -- put final version together, including a single game object and constantly showing game results.

Iterate on the GUI for several days. Make it pretty and intuitive and capable of answering all the questions you may have later. This is important.
A version of the GUI will later be the 'waking hours' anyway. It will create 'subtasks for training' for itself, then run that overnight (in addition with some mem maintenance scripts).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 28th, 2025

Rough widget version complete. It has text inputs, changing game depending on the action chosen, and everything else you could want.

The GUI is still ugly, not gonna lie. I could make it much prettier. I won't, not until I have models to test on this.

I will push this and merge the branches back together on penguins.army

Tmr:
X 1) Merge branches
2) Another framework (zooming?)

One more week of training, then kill the damn thing.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 29th, 2025

I killed training. Once I have my free weekend, I will return to this difficult problem.

~~~~

I fixed the tokenizer today. Now, it includes the special characters, and if skip_special_tokens=False is set when tokenizer.decode is called, then the special characters are present in decoding, too.
Took longer than it should have; read docs a lot.

~~~~

I want to finish a lot before retraining. I don't want to retrain off the cuff because it takes a lot of time and effort and I don't want it wasted.

Small things like:
X -- transferring the saved weights
X -- examining the trained networks, whatever happened to them
X -- merging all the branches.
X    -- widgets into this
X    -- this into master
 -- knocking out ALL remaining frameworks 
    -- start with zooming; easy enough, and takes a little creativity.
    -- continue with things related to actually moving.
    -- make sure everything in full_task_list is covered and then some.

Works well? Run the frameworks in one-off notebooks, debug them.

THis all should take 2 weeks. By then, I will be ready to dedicate a full weekend to understanding what exactly I did so wrong last time.

~~~~

Minor but important change to widgets: 'skip special tokens' set to False whenever tokenizer.decode is called.
This matters; we need our '<forward>' and other tokens.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 30th, 2025

No frameworks, unfortunately, but I knocked out all the other small tasks. Weights transferred, branches merged.

The 'trained' networks clearly need retraining, and with one task at a time. I think the confused joint training is what really killed them.

Also, I might add visual loss to TutorialQA; the reconstruction there was just too funny (putting agent shadows in the walls made them look like they 
were made of watermelon).

Tutorial1 did not learn the arrow at all. I think I needed to retrain the 'control' task before teaching it to be modified.

That's ok. I will do all of these things after / during my training weekend.

~~~~

Funny note: the reconstruction was much *worse* when I tested tutorialQA tasks than tutorial1 tasks. That's a strong argument for including the vision loss in that loss func.

Basically, depending on the prompt, the agent chooses how much mental effort to devote to reconstruction, lazy bastard.
I think this can be partially fixed by just training these one at a time and making sure each works.

~~~~

Widgets clumsy, feel more like a core dump than a simple interface. HOwever, I'll improve them over time. It's a good quick way to test something without writing custom ipynb notebooks.
Good general view.

~~~~

Next days / week: lots of simple frameworks. Run nothing.

Retraining in general: start with control framework. Add in the other frameworks one at a time, with one optimizer only.
Most frameworks will run in some sort of eval mode. SOme sort of alert if loss degrades.
Keep careful track of all losses; save them.

Basically, before the next training, I will completely rewrite GeneralTraining. Though its present form is a good inspiration.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 2nd, 2025

Made full_move_framework for teaching default behavior (making the correct moves to get to the gold piece).
Good start, but the framework is probably full of off-by-one errors and may or may not need a soft_reset between the moves; debug this before running.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 5th, 2025

Wrote direction_names framework for teaching the names of the special character actions it can take.
Wrote blue_lineQA framework for the first 'using blue line as a guide' tutorial

~~~~~~~~

Writing the frameworks isn't too bad. Today's were easy, but harder ones are possible.
Next frameworks, each about a day:
X -- 'please turn toward / away from the gold / blue line'. 4 subtasks, but that can all happen within a single framework. Big one though; full morning.
X -- the dopamine framework (which picture to do you prefer?)
~ -- adding image recon losses to all the QA frameworks (after getting burned at the first training, this is important)
     -- skipping this for now; will add in only if problems persist once I fix the training issues and use correct optimizers
X -- all the zooming frameworks, should be about a day
X -- all the 'imagine' frameworks ('imagine this room without you'), should be about a day.
X -- 'imagine this room after X moves'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 6th, 2025

blu_line_QA amended; please_turnQA written (handles several tasks from full_task_list).
Imagination frameworks next, all of them. Then, maybe a solid day on the dopamine and another on the zooming.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 7th, 2025

Wrote several nearly identical frameworks for imagining the room without an agent, or gold, or walls, or gold and agent.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 8th, 2025

Many many fun frameworks.
Only the comparison / dopamine frameworks are left. I will have to actually think tomorrow, decide if I want to use the dopamine module, 
rewrite forward, how to set up that task, etc.

Almost at the end for first order frameworks.

Tmr: dopamine frameworks, all
Saturday: rewrite training; train control framework.

The next several weeks will be adding framework after framework. I will need to think about how to use the different optimizers intelligently, but I'll get there.

I've written a short list of tasks to do in case the training takes too long in side_quests.md

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 9th, 2025

I changed the model again and added the dopamine layer to execution. It'll be a pain loading saved weights, but what can you do.
I finished the first order frameworks, at least more than enough to get started.
 -- I wrote dopamine_pretraining
 -- I wrote a framework to compare 2 images and pick the more desirable one.

Tmr / week Plan:
X -- load the updates onto big server
X -- transfer weights onto changed model architecture. Time it?
X -- think about how to train in the future. Remember to always use (almost) the same optimizer for all models
X -- get started on a 'control' training session
 -- if that works, try the other 'initialization' stuff, like dopamine and the other simple tasks.
 -- works? move on to other QA and imagination tasks.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 11th, 2025

Debugged new enhanced_model and control_framework code
Running General_training, but this training will probably be interrupted: I'm also running a pseudo-training using TutorialQA.
THis is because, when training happens both on the P40 and the 2080, an error is sometimes triggered; I want to trigger then debug this error.

Plan for week:
X -- spend 2 days trying to debug that error. No dice? Move on
X -- full control training
 -- full tutorial1 and then tutorialQA training (since I know those tasks can be done).
 -- then, with those as part of the buffer of tasks available, run the other pretraining stuff (memory, dopamine, canvases, etc)
 -- move on to other frameworks; 
    -- think before each training. 
    -- Change up which optimizers you use. See how fast this is. 
    -- Try other training options (cloud, etc)?

We're in hardcore training mode now. While that happens, if I'm ever bored, it'll be sidequest time.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 12th, 2025

Control framework training is ok. The image score will get there in a day or two. If the text score doesn't, I'll kill the system and restart with a new text-only framework.

~~~~~~~~

On the 2080, I did generate the issue I had wanted to. I found no good logs or anything else. I tried debugging using online resources; no luck.

I have posted my problem on the forums. I will wait.

The best suggestion is that this may be a hardware connection issue.
X I doubt that very much, but before the next training, I will try a different connection and try to generate the same error.
We will see.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 13th, 2025

Restarting control training with v2.

I tried opening up the server, but no luck with it.
THe only other connections available are all inconvenient to reach for the large GPU.

I'll build a cheap new server around the 2080, I think. And *then* I will buy / rent more training equipment.

I'm leaving this task for now. I will only follow it on the NVIDIA developer forums, where I posted my issue.

~~~~~~~

Next days:
X check training
X restart stuff?
READ PAPERS.
reorganize the directory

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 14th, 2025

Training continues ok. The img loss is almost where I want it. THe text loss is slow. I'll give it till the end of the week.

I restarted triggering the GPU0 shutdown. ONline commenters suggested this is a temperature issue, so I'm running this with a temperature logger.

Look at folder temp_recorder on this machine, tmr, to see the results.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 15th, 2025

Based on the losses, it looks like the training did complete its job.

Unfortunately, it looks like I killed the entire setup in the remote office.
THe wifi extension is dead
I can't boot up the server, either.
This also means I cannot access the remote disk.

~~~~~~~~

On the small machine, I added temp_recorder to *this* repo, and added a monitor_stage function which will cool the device if it ever gets too hot.
Once I get the server working again, I will add this to all the training notebook headers:
    import temp_recorder as tr
I'll also add this to all the train loops:
    tr.monitor_temp(device)

If that works, I will simply include it in all future loops, and forget about it.

Tmr plan: fix the remote wifi. Small goals.
Weekend plan: 
fix the entire server, or at least save the snapshots from training.
Server fixed?
X -- evaluate training
X -- evalute the temp curve and whether that was the issue
     -- maybe? cutoff should be more like 75, it seems
X -- save all the snapshots
X -- next stage of training launched, with temp monitoring.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 16th, 2025

Nothing today.

The wifi extender fixed itself, all good.

Tmr, I will fix the big server and then continue with various training tasks (see above).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 17th, 2025

Big server turns on fine, it turns out. Need to make sure to use the HDMI connection only, not the others.

It refuses to connect to the internet on the extender. I thought I had it fixed.

Tmr: make that work. Check internet with the behemoth dragged back into the living room. THen check network card (open carapace).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 18th, 2025

Big machine turns on fine and connects fine in the living room.

And connects fine in the server room, too.

And can be ssh'd into (with new IP).

Damn, 40 minutes wasted on what turned out to be just a transient error. Should have reproduced before debugging.

Oh well. Back to normal business.

~~~~~~~~

Official model after control training will be enhanced_brain_control_training_v2_batch55900.pth
 -- basically, looks good enough in vision when I try it with the Widget Interface

I'm keeping a couple others around. That one will be on the USB.

Relaunching with tutorial1 training. Hopefully won't be more than a day or two.
Also launching on small device, as per usual. Hoping the temp controller fixed the issues

~~~~~~~~

Training has started.
THis week:
 -- continue training according to curriculum (check above for details)
 -- pick some papers and read them.
 -- Maybe pick a list of models / 3d envs / robot systems / other sidequests to try and explore

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 19th, 2025

Temperature was not it. Shelving that debugging.
Restarted arrow task training with v2.

Tmr: maybe launch new training, but more importantly, start a side quest, probably some papers and blogs and maybe X posts.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 20th, 2025

It draws the path, but can't use the language cues to tell when it *should* draw the path.

Loss is still about 3 times too high (in task recon) than if the firstTutorial notebook.

Training another day or two, with 3 times as many arrow_task runs as control runs (to maximize drilling in this lesson).

If this doesn't work in another day or two, I'll try to remedy the situation with some text-only drilling (new framework).

~~~~~~~~

Also made directory 'research' and phrased some organizing questions.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 21st, 2025

Loss going down again, properly.

I think I'll let it run through my vacation before re-evaluating it.

When I'm back: definitely run a 'text only' training framework for a bit, to improve that absurd number.
From there, I move on to the tutorialQA stuff, then a lot of the pretrainings (memory, canvases, dopamine, etc).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 27th, 2025

Training for tutorial 1 done.
Loss size promising.

Tmr: Xtest it, 
if works launch next thing (probably memory and / or Tutorial QA, though there are options; consider).

~~~~~~~~

Quick fact: I found the size of my model. This is the pytorch snippet that gets that for you:
sum(p.numel() for p in model.parameters() if p.requires_grad)

Using this result on my current network yields 244 M parameters.
THis is probably too little for intelligence, but all the other networks are known to be too large, so maybe I'm not wrong after all.

~~~~~~~~

Finished paper-reading and pure research. Will try to download / modify Florence 2 in a new repo, hoping for transfer learning.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 28th, 2025

Today was actually pretty fun. It *did* learn how to draw the line on command. However, it used the wrong text trigger: sentence length, not content.

Luckily, this was easy to fix with a few short training sessions. They clicked very quickly, too, which was lucky.

The best version of the brain is now enhanced_brain_arrow_task_v5_batch133

I don't want to spend more time orchestrating a train sequence today.

Tmr:
X Launch a text-based or pretraining-based (memory, canvases) session tomorrow.
Feeling good? new repo on penguins.farm; learn to torchify / extend / finetune Florence 2 and other hf models.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 29th, 2025

Lots of delays

Finally launched next training: mem_canvas_use_framework.

Will do the other memory thing next, then a week straight of only text stuff.

~~~~~~~~

For the huggingface model, use this:

https://huggingface.co/docs/transformers/en/custom_models

Steps for it:
1) NB with loaded Florence model
2) Check 'forward' results (when passed text and imgs). Find way to get text embeddings and img embeddings
3) Inheritor class that uses those embeddings
4) Develop from there.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 30th, 2025

It needs several more days, needs another order of magnitude.

Hopefully the text-based ones go faster.

Restarted.

Day continues on penguins.farm

~~~~~~~~

I tried to steal Florence 2, but it's just so boring and such a pain.
Small repo called Florence_Proper created, but I'm not doing anything with it.
Project abandanoned.

~~~~~~~~

I think I found a good paper on how to use memory, longer memory: https://arxiv.org/pdf/2404.05726

Most of the stuff they describe is actually doable. I will do, about, 20 % of it, and hopefully end up with a usable memory system.

I don't want to redesign / retrain the brain again, at least not yet. Luckily for me, most of this stuff can wait until around phase 2.

Making memory_phase2.md to discuss these ideas (this README gets swamped).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 31st, 2025

Still more training necessary.

It definitely isn't picking up on the 'use the other canvas' idea at all.

Not sure how to change that.

~~~~~~~~

Added proper image reconstructions for the mem_enc module. That will be useful for debugging (can see exact recon quality).

Before launching it (hopefully tmr), run it through a custom ipynb, check for loss scales (maybe comment some losses out), etc.
Don't use general optimizer on it; optimize only the mem_enc part (or else it'll overfit by destroying the rest of itself).


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 1st, 2025

Bad day.

No progress in training.

If that continues, this week will be debugging this training module.
Steps:
X 1) nb duplicate of the canvas_use framework. Make sure the task is set correctly at all.
2) Debug, like I did back when I started on vision. Do it on new branch, possibly with new class name
   -- new architecture with only 4 inputs and varying output.
   -- try to teach it from scratch or from other inputs
   -- DO NOT change lr; too low as it is
   -- try different ways of signaling which input is which (that is, position encoding).
   -- don't start with it, but try different decoder architectures: start with the 'real image' features? Somehow choose / mix among the inputs with a dynamic vector? Start with zeros?
      -- the 'dynamic mixing weight' idea is probably best, but it will require the strongest restructuring for the main task.
   -- while I am here, I *may* also pretrain some memory. Will see.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 2nd, 2025

No glaring errors in mem_canvas_use_dupe

Switched to new git branch. Will run the harder experiments here.

To start, I'm restarting General_training, but I'm doing it de novo, no control task, even; I am only interested in the results from this new task.

If that doesn't work in a day or two, I will start changing the architecture, as described above (especiall changing how img_dec is called)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 3rd, 2025

No luck training de novo.
I'll let it run tonight, then kill it.
Morning did not leave any time to change anything.
Other things to do:
 -- next training, whenever it comes, will place even less weight on the text (something like dividing by 10k or 20k)
 -- next approach to this needs to try the other architectures discussed.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 4th, 2025

Draft of new code written. I don't feel like testing a ton of things. The retraining really *should* be fast this time.

Tmr: test. If ready, transfer weights. Train on control and the line (no context use yet). If it works well, switch to using the context.

~~~~~~~~

If that experiment fails (early next week), try teaching it the context at the same time as the control (and maybe even waiting on tutorial1 stuff).

This is an annoying distraction, but I expect the QA stuff really will be fast, so . . .

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 5th, 2025

Good start. New model seems to have the correct format.

X Will start control / line drawing training right away.

Will check on it tmr.

~~~~~~~~

Transferred weights from enhanced_brain_arrow_task_v5_batch133.pth
 -- NOTE: the version on the penguins.army server is now CORRUPTED! Next time KINGSINGTON is plugged in, replace with the backup

Using the name 'super_brain' in the save files. NOt my favorite. Will think of another one.

Retraining control and arrow_task for a day.
Works? Great, move on to the canvases again.

If the canvases don't work again, I will try to train these in a different order, maybe control and canvases together.

I'll figure this out. The QA tasks are next.

~~~~~~~~

UPDATE: this change forces me to go from a default 16 to a default 8 images per batch. This is a limitiation.

After making sure that this current system works with the canvases, I may go back and try a more lightweight vision attention module, to help with training down the road.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 6th, 2025

Dumb mistake cost me a day. Stray comment made it so taht I didn't load the saved weights into the new brain.
Redoing control / arrow training. Canvas training in the morning.

If that doesn't do it: train only the img_weight layer to produce 1, 0, 0, 0 by default (for an hour?), before then testing inference for old tasks and then switching to canvases.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 7th, 2025

I think it's getting there. For some reason, it needs a little more time to learn this.

If it's not done tmr - or even if it is - I may experiment with this a little more.
For example, could I work off of the original (transferred_weights) example and train up a img_weight separately? Combining them together after only an hour of work?

Will try tmr or during the week. Unfortunately, I now have to go.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 8th, 2025

Lost another day, damn. It hadn't loaded.

I'm letting it train over night. No dice? Easy training in the morning (img_weight alone), followed by canvas training Tuesday.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 9th, 2025

No luck. Checking weights by hand, they always produce results around 0.25

I realize that I'm doing something wrong: the input tokens need location markers. The input cannot be 0s.

I'm trying the 'just train' approach once more, but only over night. In my understanding, the system *should* be able to learn this.

If that fails, I'll do the quick hand-training in the morning, and check inference performance after.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 10th, 2025

Very strange day

I saw that the new system had failed to retrain fully, but img_weights were turning out correct. This was very strange.

I tried frankensteining the new img_weight layer onto old systems, and the img_dec outputs were complete garbage.

I could just let it keep training, but this is an interesting puzzle that goes against my intuition.

tmr, I will recreate the fwd loop in a new notebook and make sure that everything is correct
(and that the 'frankenstein' brain really has trained weights - though I transferred several times from different old brains)

In short, will investigate. If it really was all correct - unexpectedly - then I will have to let it keep training. Maybe training in a different order 
(control first, then tutorial1, etc).


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 11th, 2025

Jackpot. The 'transfer weights' nb was not transferring weights. I added safeguards to make sure this doesn't happen again.

'frankenstein_transferred' is now correct, as is the old enhanced_brain_arrow_task_v5 (transferred over from storage).

Issue fixed.

Tmr:
X 1) Cleanup
X    -- backup frankenstein_transferred. That's the best version of the new brain architecture that is available.
X    -- delete unnecessary nb's. Consider the 'DELETE_ME' notebook and maybe the 'recreate forward func' (though that one may be useful
X    -- all the saved brains. Probably delete every 'super_brain' except for the 'frankenstein'.

X 2) Launch training
X    -- probably canvases? This can wait if there are any problems / delays.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 12th, 2025

Cleanup done, canvases laucnhed. Fits on alligator GPU.

Next: check results tmr, take next steps.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 13th, 2025

It (almost) worked!!

It learned to use the canvases a little. However, I think it's using the wrong indeces.
Callbacks of '0' and '1' seem identical (which is not good), while, callbacks of '2' go back '1'.

Tmr: recreate (recheckout?) the mem_canvas_use_dupe, and debug it. Make sure the target is what you want it to be.
Maybe rerun, with max_lookback of 4 instead of 3 (so all the canvases are in play).

I should have done this today. Today was wasted.

Still, it's very encouraging to see progress at all.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 14th, 2025

Very odd in general.

I think mem_canvas_use was correct as a rule.
However, the thing hadn't actually learned quite the correct patterns.

Looking at the weights themselves, it seems that it's still learning the correct patter for '2 ago', but already knows the pattern for '1 ago.'

Will launch for another day. No luck? I will change the loss function to directly channel the weights to be correct.

I'm also increasing N_lookback by 1 because it needs to go up.

~~~~~~~~

Next things:
1) maybe change the loss function to train img_weight directly.
X 2) quick small new framework, to zoom in on the agent and to use the 'recall' function to draw lines on older canvases.
X    -- that is, correctly use recall even when the canvas is some arbitrary material.
3) MOVE ON. TutorialQA is really begging to be run.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 15th, 2025

THe training had crashed. ANother day stupidly wasted.

I dropped the batches for the new task down to 6, and added some code that lets most tasks use higher batch sizes. 

I may try to increase the default batch size to 12 (16 is too high for arrow task).

I'll also try high batch sizes on the QA tasks, later (because they can handle it: the loss goes only through text.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 16th, 2025

Training weird. Goes up and down.

I'll let it run for another day before finally hacking it to return canvas weights and train on them directly.

Today, I edite zoom_framework (now actually runs, and zooms in much more), and added a (commented out) option on mem_canvas_use_framework to recall canvases with lines drawn on them.

THis is still going very slowly and much too slowly for my taste :(

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 17th, 2025

It wasn't working, so I wrote a new framework. We will see.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 18th, 2025

Dumb typo cost me a day. Restarting. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 19th, 2025

Canvases finished training. GOod results. Too certain of itself and tripped by small text mistakes, but I can tolerate this for now (not later).

If the hard-coded weights signal is too stubborn, I'll write 'CHEAP' versions for other frameworks.

Launched tutorialQA training in a naive way (using larger lr). Will see if it destroys all other performance.

~~~~~~~~

The more I work, the more I want to use transfer learning from systems that already have rudimentary vision / text comprehension. 
During trips, I will look at that. See the research doc for more info. 

~~~~~~~~

I killed the training early.

QA may have been learned, but the other tasks are showing strong signs of deterioration.

I will look into the qualitative behavior tmr. I will likely retry with some improvements (fewer text runs? Only train the decoder? Dunno).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 20th, 2025

First training was a failure. Even control reconstructions suffered, while the essence of the arrow and the canvas task were forgotten completely.

I will consider retraining once I'm back from Miami. I will focus on reading and plotting while I'm gone, because I'm just a little sick of frameworks.

~~~~~~~~

Next:
1) Retry TutorialQA in 3 more ways until I find something that works. Keep the 'CHEAP' training for the canvases online.
   -- first way: small lr for the text encoder, large lr for the text decoder.
2) Once I find a way, knock out all the QA frameworks. The 'complex loss function stuff' can wait; I want a large fraction of the frameworks under my belt.

~~~~~~~~

This shows me something: I want the 'fundamental' skills not to vanish once I start trying to teach it basics. I don't want to always keep that part of the training data running.

I will use some version of LoRA once I get to phase 2 or beyond.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 23rd, 2025

I looked up 'reasoning' and commented on it in the research folder.

Bad day; slow start. Do better.

Training tomorrow.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 24th, 2025

Launched TutorialQA idea one more time

I chenged the higher-lr text_optimizer to only optimize the text decoder, and added som general_optimizer passes for the tutorialQA_framework.

Hopefully this is enough (most of the other tasks do not heavily rely on the text_decoder; if control is ruined, I might go back and run a few more passes with that feature turned off).

Hopefully this is enough. We shall see tmr.

~~~~~~~~

Tmr: actually take some time to review the videoLLM stuff I've mentioned one too many times already.

And maybe work on the toy, too.
Or make the company, legally.

After the GeneralTraining tasks

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 25th, 2025

Bad day, only training.

I can tell from the losses that the other skills still deteriorated somewhat. HOwever, it looks like the skill was probably learned fully.
Relaunching with the 'text optimizer' part turned off. Maybe a day of normal running can restore the other skills.

Tmr, I will try to actually look at all the performances again, then make a choice about how to proceed.

~~~~~~~~

Tmr: what I had planned for today: toy, videoLLM, maybe register the business

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 26th, 2025

Bad day. Need a good day. Do the other stuff on the good day: toy, videoLLM, maybe register the businesss.

Training did not work; the task seems to have been learned, but lost skills were not restored, and halfway through there was a bizarre regression (later epochs not usable at all).

Restarting from the last part that worked (canvases_v3), using only gen_optimizer. Will see if this works within good time frames.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 27th, 2025

Restarting again, fewer QA runs out of the batch. Looking at the learning curve, I had a catastrophic failure of task 1 again, and that will not be tolerated.

No work on the weekend, hopefully the weekend will be enough. I need better days next week. Here's to hoping.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 30th, 2025

The weekend trainig failed. Tasks decomposed again.

Rerunning with more focus on the arrow task.

I think this is an issue of not using the image loss in tutorialQA. Not sure how to fix that.

~~~~~~~~

I'm letting it run once more in this setup. If it doesn't show catastrophic forgetting, I'll use this time to:
0) read the visionLLM papers,
1) get / start a visionLLM prototype,
2) try to find another LLM I can hack together for transfer learning,
3) open the container for the toy.

~~~~~~~~

The next steps after this? 
1) rewriting all the QA tasks so that they don't shirk on their vision reconstruction capabilities.
2) reading about catastrophic forgetting.

These frameworks do have me feeling like I'm spinning my wheels. I will have a good think about it in 2 weeks.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 1st, 2025

Mostly successful.

arrow task loss collapsed halfway through training, but it was ok by the end.

I'm not loving this sensitive thing. I'm relaunching for a couple days to get the lr down. If I actually wake up on time at least once, I'll look into more long-term solutions.

I may also need to just graduate to multi-GPU cloud training and stop being a baby.

Current best version: frankenstein_tutorialQA_v6_batch10993.pth

~~~~~~~~

Tmr:

look at progress
cleanup disk
side tasks (especially research and robot car)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 2nd, 2025

Typo yesterday cost me a day. Restarting training.

Disease and boredom are severely impacting my progress. In 2 weeks I'll have a chance to do some more research and rethink my approach.
Until then, this stuff is secondary and I'll only launch occasional framework trainings.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

? Jul 11th, 2025 (exact date unknown)

Weird behavior during trainig - lost connection but it still was training.

Killed it a couple days ago but couldn't restore connection.

Restored now (IP address changed).

Tmr: check behavior, save to the thumb drive, make next plans and start next training.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 15th, 2025

v7 did not perform super well.

performance on the other tasks still craters eventually.

I fixed a bug in the tutorialQA task, and I am restarting it.

I'll keep the lo_lr optimizer for now - maybe it will be useful later? - but I think the next task will be trained using general optimizer.

~~~~~~~~

This feels like make-work. I'm not at all certain I can do a QA framework a day.
I need to think about architectures / transfer learning, and also work on the robot.

Hopefully I'll have some proper mornings soon, finally.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 16th, 2025

v8 did ok. The canvas task sometimes fails, but that's it.

I'm keeping an instance of v6 and the final version of v8. Everything else goes.

Launching blue_line_QA today. If technical difficulties, will fix in morning and relaunch.

QA frameworks for several days; I'll find a way that works. If this one doesn't, it'll be lo_lr from here on out, forever.

~~~~~~~~

General thought: one framework at a time with endless backtracking probably isn't the way. The way is probably random batches, especially mixed tasks, all at once, on a big loop.

SOme budget for AI training and just many loops, single big loss, watch multiple little lines go down. Maybe some corrective function which triggers 'more of this framework' if one framework falls behind.

Will look into multi-objective training sometime.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 17th, 2025

blue_line_QA had a bug; fixed and relaunched

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 18th, 2025

More blue_line_QA fixes. If it still vexes me tomorrow, I will go through line by line.

This make work is not enough. Tomorrow will need more serious work, like paper reading, large plans, tests of other systems, Grok tests, and so on.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 19th, 2025

Conclusion:
1) the tutorialQA tasks mostly work
2) the memcanvas tasks mostly work (it can't remember what happened exactly one image ago, on canvas2. It's an issue).
3) the blue_line_QA stuff is a mile away from completion (often gets the wrong answer).

Rerunning, with lo_lr_optimizer, overnight. Adding on some cycles for blue_line_QA and for the mem_canvas tasks.

Tmr:
X -- rerun the tests
X -- triple check that mem_canvas_use is testing the correct thing
X -- launch another training sequence
 -- do the other things (make the bot work, etc).

No going outside tmr. Tmr is just work.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 20th, 2025

blue_line_QA was *better*, but not much better.

the arrow task endures MUCH better with the lo lr.

The last (new) tutorialQA task needs sometimes fails, as does the tough mem-canvas-use case, and the new QA stuff.

I think this will greatly benefit from a week-long run. More recent frameworks do worse than older ones.

I wonder if I can get away with adding all the QA tasks together and running for a week. Any problems can also be debugged all together on the weekend, and 
I can do more fun things during the week. This also removes some of the 'different age' problems (I want the frameworks to be roughly the same quality).


Started that run, will come back after lunch.

Plan today:
X -- debug any QA failures; watch it actually start the run
 -- robot
 -- read papers? Read up / test Grok?

~~~~~~~~

Finally runs without failures

Will not check on it this week.

Tasks for week:
 -- robot
 -- research (papers, Grok testing, other)
 -- look into a network for transfer
 -- look up cloud testing. Do the math. If it's below 1k for decent RL, do it.

Remember the alternative RL system: learn navigation first, rudimentary words only later. Can be trained not with RL but with 'hacks'

~~~~~~~~

Overall, great progress. Now, I'll wait a week before returning to boring framework work.

I also have a better feel for what is learned (arrow on low lr) what is not (arrow on high lr) and what suffers (vaguer questions, more recent questions). 
I no longer feel 'eternally falling behind'; I understand where I am.

This week will be a good one.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 27th, 2025

Good week on robot and cameras.

Disappointing training results.
The blue line was drawn well, but not much else.
The other QA frameworks need to get added visual losses (especially blue line).

There's a very good chance that I'll need to 'abandon and start over' soon. Real cracks that are difficult to overcome are beginning to emerge.
Several things may fix this:
 -- retraining with different relatie weights may work
 -- I could find another way to train this (cloud)

Otherwise, I really need to start thinking about ways to fix my approach, and especially how to take existing networks and use them

I'm also not a huge fan of the reconstruction quality. These are very simple images. Why are they difficult for the network? What better architectures can I use?

~~~~~~~~

Plans for the trip and for August:
1) Research
2) Robot hacked, fully
3) RL version of my agent trained
X 4) "Special loss functions" use attempted. Can be in parallel, on the other machine.
   -- no dice? I can give up on it and switch to using frozen versions of my agent adversarially. This would be useful later on, too.

I *could* go back and make everything work for a much smaller resolution, but I do not think that I will do that.
Human-level intelligence is (probably) full-resolution.
 -- put some more thought into this.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 28th, 2025

On penguins.farm

I'm trying I fixed complex_loss_func_v1_framework, and am launching an experimental training run on the small machine.

I'm expecting to lose a lot of skills in the other areas. I will not transfer this training run.

What I want to check is this: does this complex loss work at all? Or should I focus on using the system to backpropagate feedback signals, later?

Tmr:
 -- check all training progress
 -- ROBOT. ALL
 -- prep for the trip, if time

~~~~~~~~

new_global_plans.md file created in the docs folder.

~~~~~~~~

UPDATE: training loop nan'ed; will debug tmr

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 29th, 2025

I think I've found the issue:
1) in image_to_settings, if the color map is not full, weird things happen (gold may be missing, remember)
2) also, the 'max' in image_to_settings does not seem to be correct, which is annoying me.

using the gold radii may be a problem in general - high gradients on few elements - but I do not think that is enough for nan's

Solution:
Tmr, detailed debug of image_to_settings and the complex_loss_func file:
X -- filter not to use masks that are incomplete (with torch.no_grad)
X -- figure out why the max is close to 0.45 instead of 1 like it should be
X -- use matplotlib.pyplot to see the masks and the max's
X -- try this with other 'backend' mask generators (remember you have 3 differentiable options)

Once I do that, I will rerun training, using LOW LR. 
No luck with that? Different backends.

No luck? Dropping it. The complex loss func will be fully debugged or abandoned before the Chile trip.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jul 31st, 2025

Fixed the masks. No more nan's or other mistakes.

Ran the code yesterday. The task was not learned. THe agent just became 'fuzzier'. Not sure what to do about this.

Most likely, complex losses are just a write-up. When I return, I will focus on the robot, and maybe afterwards on the new system (which I will write over several weekends), plus remote-training RL.

On the next notebook, I'll save the complex loss stuff for the very end.

Off to vacation! THis code will not be touched for a while. Lots of reading will be done, and maybe experimenting with open-source NNs.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aug 11th, 2025

Back from vacation. Killed both training loops.

The 'complex loss func' still eventually went to nan's, but I have no idea whether it trained anything semi-intelligible in the meantime.

Killed the other server, too.

Tmr:
1) Investigate performances on both tasks
2) Decide which versions to keep
3) Sync representations (git branches, files on the disks)
4) Done? Do one of the other tasks:
   -- look up the list of other tasks as reminder; step back into it.
      -- add the 'examine with design concepts' item onto that list
   -- just drive the robot around; all next steps for it
   -- lawn care robot
   -- GPU, server costs
   -- start looking up other models you can use, especially Chinese models.


