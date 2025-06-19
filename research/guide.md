May 20th, 2025

Research priorities are a little jumbled.

I have two main goals:
1) Understand the state of the industry and players
2) Look up cool models / concepts to steal.

For point 1, I will
a) watch a Sam ALtman video
b) Play with ChatGPT, Grok for a day (and maybe Claude)
c) Look up what Yann LeCun and Gary Marcus are doing; look up other big names and check recent publication
   -- see next day
d) Scroll through papers from recent conferences, download at least 2 or 3 here.
e) Read up on AutoGPT progress

~~~~~~~~

For point 2, I will

I)	Download Microsoft's Phi 4; play with it
II)	Find at least one more small language / multimodal model I like 
III)	Learn how to finetune Phi 4 and / or Florence 2
IV)	Look up / learn how to add parts to Phi 4 and / or Florence 2
V)	Read their accompanying blogs

~~~~~~~

Questions to answer:
 -- what is being done for multimodal agentic AI?
 -- what are the best examples of that?
 -- what are some of the best 'competitors' to the system I am making?
 -- What is OpenAI up to?
 -- look up that chart for when things should break (2026 or whatever). How does your progress measure up to doomsday?

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 21st, 2025

Short list:

AlphaEvolve (Gary Marcus recommendation)

Papers at the bottom of this FAIR blog (Yann link):
https://ai.meta.com/blog/meta-fair-science-new-open-source-releases/?utm_source=twitter&utm_medium=organic+social&utm_content=video&utm_campaign=fair

NeurIPS links, scroll relevant: https://neurips2024.vizhub.ai/

CVPR preprints, scroll relevant: https://arxiv.org/list/cs.CV/recent

Other conferences; scroll relevant: https://aideadlin.es/?sub=ML,CV,CG,NLP,RO,SP,DM,AP,KR,HCI

Apply here and watch videos: https://altera.ai/

Give these arrogant guys a shot: https://deepagent.abacus.ai/mbh

~~~~~~~~

I refuse to sit down, make a new folder for the papers, read them all and summarize and seek a niche as before. Maybe for memory stuff, but that's it.
I will spend the upcoming trip, in the mornings, reading these papers, and writing a short summary for each.
Maybe in this document.

Once I'm back, THAT'S IT! Switch out of the academic sources. Focus on industry and on non-AI stuff (practical steps to set up a robot env with lego or other platform).

~~~~~~~~

Broader plan:
1) 1 or 2 papers a day for now.
2) Next week: Phi 4 and / or Florence, locally. Find ways to extend them. Find ways to finetune.
3) Industry sidequest
4) Look deep into 3d envs and / or robot interfaces. Make a purchase decision for Lego.

~~~~~~~~

Broader view: 
I want to do this for two reasons: 
a) understand the current market
q) borrow ideas

For the first all of these demos / papers are good

FOr the second, barring flashes of insight, I'll need to probe very very carefully. Or run tech demos on my own.
Here are some things, out of order, that I can do for this:
 -- make sure I can locally run Phi 4 and modify its architecture.
 -- run little tests, like trying a branch of the EnhancedBrain with a 512 character context.
 -- Deliberately look up papers with memory (old, new), including from my own library.
 -- Look up the GPT / Deepseek / other architectures again and see how they handle this issue.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 22nd, 2025

Short list:
1) USAMO paper saved on the USB. The systems clearly still struggle with human proofs, but they can copy math-type reasoning to a scary extent.
2) Grok did not solve the Riemann hypothesis
3) Optimus can learn tasks from watching videos of humans, which is cool and suggests very good grounding.
4) Here are several papers from which I may (but probably won't) steal algorithms, which I skimmed today.
   -- general RL methods: https://neurips.cc/virtual/2024/poster/92983
   -- methods for finetuning which involve splitting the model up into agents (a trend in the research community): https://neurips.cc/virtual/2024/poster/94727

Summary: multi-modal models do seem to be getting quite good. That's not enough for an edge, for me.
Once I have the multi-modal model, I will need to train it how to handle things like "discovery" and then internalizing that discovery.
My current track is viable, but only just. I need to try to use some transfer learning to skip ahead.

~~~~~~~~

Next several days: 
 -- focus on components
 -- focus on Phi 4 and other practical models I might use (theory and demos, no code till I get back).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 25th, 2025

Good day. I now have a sense of the state of the industry and the top 2 startups (reflection.ai , altera.ai , and some others like reflection.ai (coding AGI)).

Reflection.ai is chasing AGI through AI coding. One approach, but not mine; I am curious to see if they accept me and what they will do about it.

Altera.ai remains the GOAT. You might want to copy part of their system (3 different vector memories used, diff time scales), but later.
 -- I don't want to retrain any time soon, but I think your initial tasks, especially explicable corners, can be done with only one or two memories.
 -- Altera paper, should be always top of mind: https://arxiv.org/pdf/2411.00114

FAIR isn't doing anything fun. Interesting statistical similarities between training LLMs and teaching kids. I do not have the time to understand this well enough to use it.
Cool, useful results on learning to read minds from non-invasive electrical sensors.

Alpha Evolve is a cool, iterative (genetic algorithm) method for algorithm search. LLMs limit dimensionality; only "mutations" taken by LLMs work.
Cool that it helps design TPUs and the like. Not on the path to AGI (at least not immediately).

All the large models (Grok, Gemini, GPT) are multi-modal and on the way to AGI. However, they're still not trained for the rigorous grounding nor the 
internal simulation nor the self-guided learning that's needed for this. The internal consistency and breadth of their training data is scary - USAMO 
was not on my bingo card for solved problems - but they're still a couple years away.

GPT 2 architecture was just 1024 symbol window + 12 heads and 12 layers of transformer.
 Experiment to try: can I tran something this size?
 Skil to get: learn to estimate the number of parameters you have to train. Try to minimize it, and maximize "canvas size"

Abacus.ai is still a waste of time: just another coding LLM. Use others (eg Claude) for vibe coding instead.

~~~~~~~~

I applied to Altera and Reflection. Will forget those for now, unless I hear from them.

~~~~~~~~

Next steps:
Glance at Diffusion Language Models for approach (maybe update 'full task list' based on those approaches)
Phi 4, paper
Scroll through the conferences, one eye open (that's not where innovation happens anymore)
 -- search for knowledge representation, RL, multimodal, grounding
GPT / Gemini / AlphaEvolve demos (videos; don't waste too much time playing with it)

Then, move on to getting local Phi 4 and modifying / finetuning it. See how to turn a huggingface model into something you can train / hack in familiar pytorch.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 26th, 2025

Here's Flamingo

If I have to change architecture again (likely, to add more memory), I can try their general approach (freeze network with translator layer)

https://arxiv.org/abs/2204.14198

~~~~~~~~

Really not worth the time. I could use one of their frozen models, later, but I don't want to investigate this right now. Model is too big, anyway.



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 27th, 2025

Wrapping up research tangent.

Phi 4 is not a practical system for me to use at all: 40B parameters is too large. I need something under 1B.
 -- cool concept, though. Basically, they made artificial documents in a single format that had all the information they wanted
    (from heterogenous sources, sources with many different formats).
 -- not really relevant to me.

Two good papers from NeurIPS which I *might* steal, later:
1) Extending text context by using the visual field (damn genius): https://neurips.cc/virtual/2024/poster/94826
2) Video system I may steal / use as a component later: https://neurips.cc/virtual/2024/poster/96668

Both are tentative and both are big stretches.

Main summary: 
1) The big models look like they're getting close to AGI, but the approach is still wrong and still no one insists on good grounding.
2) The robots / real world agents are one place where AGI might happen
3) Another scary thing is that the self-consistency is now good enough for very long, intricate tasks, like USAMO (sometimes)
4) Best shot is Altera. They are pretty much your direct competitor. I have the advantage of using more things that don't scale, and not caring 
for intermediate results much. Let's hope for that
5) I should really really test AI coding helpers and maybe some video stuff, but I'll only do this later, not now.
   -- AI coding weekend or full week after quitting?

~~~~~~~~

Next steps: 
0) No demos. Not the time; I got the gist of SOTA and there's no need for me to learn the skill of talking to these things.
1) In new repo, find way to download, *modify*, and then retrain / finetune Florence 2 and maybe Flamingo (if it fits)
   -- remember, you're constrained probably below 500M parameters with the current setup.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 30th, 2025

Good paper: https://arxiv.org/pdf/2404.05726

I will base the memory unit on (elements of) this paper.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jun 20th, 2025

 -- look up the video stuff from the video LLM presentation at work. Try to find something you can replicate or steal
 -- look up the "reasoning" stuff; what it means, how it's done. Consider how to use that
 -- I loved the "tag" which meant "go look at this caption". It was very useful. Use that (later, when instructions arise)

Those two are important. I should figure out the current SOTA.

This can be done on trips / planes.

Reread the May 30th paper, again, too.

~~~~~~~

Maybe I'll end up taking a standard video QA network and modify it for my purposes. THis will not happen until after I quit my current job.




