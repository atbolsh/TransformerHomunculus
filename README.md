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
    -- this 'interaction with env' mode can get more bells and whistles later. I want, right now, a basic system with only neural (and maybe persisten visual canvas) memory. Text banks and files come later.
    -- reread the Foerster RL paper. Try to use as a guide?

~~~~~~~~

Looking more closely at fake_traces, it really should have worked.
Debugging it may yield the highest return on investment. Make sure you're computing what you think you're computing.
But this can wait for the other projects.

I don't think that the 'semi-traditional pseudo-GRPO' thing will work. The off-policy instructions from the guide are too different from the on-policy distribution.
THough maybe the probabilities for the correct actions will go up . . . will see. It has a couple of days.

Fake_traces must be debugged first, however.

