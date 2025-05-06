March 18th 2025

First order: (one, max two steps)
~ Are you to the left or right of the gold?
~ Which way do you need to turn to face the gold?
~ Is the gold in front or behind you?
~ Are you facing the gold? (interpreted as 'if you go forward will you eat it'
 
~ Are you facing the blue line?
~ Do you need to turn to face the blue line? Which way?
 
~ Please step forward
~ Step backward
~ Turn right
~ Turn left
 
~ Turn towards the gold
~ Turn away from the gold

 Are you near the gold? (for some function of near; choose this).

 Do you want to be in the second or first image? Which is preferable? (other versions of this; uses dopamine and language).
 Is the gold closer to you in the first or second image?
 
Requires visualization:
~ Where were you in the last step?
 Where were you 5 steps ago, roughly?
 Imagine this room without you.
 Imagine this room without gold.
 Imagine this room empty.
 Please zoom in on just the agent
 Please zoom in on the agent and the gold
 Please zoom in on the gold

Will require difficult loss funcs:
 Imagine if you were further away from the gold?
~ Imagine if you were closer to the gold?

 Imagine if the gold were closer to the upper left corner of the game.

These will help when I get to corners:
~ Imagine a line from you to the gold.
 Imagine what it would look like if you were facing the gold
 Imagine if you were halfway to the gold.


Second order:

~ Please solve this game: turn towards the gold and eat it.
 Please tell me which of the gold pieces is closer
 Is this reconstruction better, or worse, than this one?
 Is this path better, or worse, than this one?
 Look at this agent move over time. Please draw his path in the next image (memory exam; easily can do more than 3 images this way).
 Look at this agent's path. Is he moving in the right direction?

 Please imagine a piece of gold near the center of the board. Then, visualize a path.

Early plans (no more than 2 or 3 steps; longer waits for stored plans):
 Please take 2 steps forward, then repeat this phrase: "phrase".
 Please take 2 steps forward, then visualize the path from your old position to your new position.

NON-VERBAL:
** Train the dopamine session every time it reaches gold or not, or every time it can make choices.
 Teain recalll from past events (use text, I guess, but it's not really necessary).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mar 19th, 2025

The list above is mostly 'toddler-level': basic orientation and imagination.
This is what I will do in the 'no internal walls' setting.

Once that is done, I will move on to 'internal walls' and 'using imagination deliberately' and maybe 'store plans.'

List of broad categories, and order:
1) Basic navigation, imagination, description, and goal-targeted actions. This is gold-seeking in empty room. All above.
2) Brainstorming deliberately, maybe storing plans explicitly. This is corners / internal walls. Will write later.
   -- this may require human input and RL-ish feedback. Try several outputs, reward / punishment, then generate 'similar' situations with same answer, then store all in a batch with correct reward.
   -- basically: super slow human-in-the-loop RL may be improved by automatically generating the correct training (like an entire batch) and then returning to it again later.
      -- initially; this would be part of the training mechanism alone. Only later, much later, would the agent get access to this deliberate learning.
3) More advanced navigation, map-making, maybe storing images. THis is multi-room experiments. Will write later.
4) Explicit novelty; deliberate learning; using L2 masks between reconstruction and original; maybe setting up train loops. This is 'novel items discovered.' Comes much later.
   -- may have happened earlier, see above

Each point take several months. Each task above should be a day or so of writing and a day or so of training, once I have the tools set up (close).
With delays, I'll probably be done with / finishing up point 2 when I quit in August.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apr 24th, 2025

Added some terms above (the subpoints)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May 5th, 2025

I've started crossing things out from the list above. ~ means 'framework written but not tested or trained'. I will use 'X' to mean that the framework has been trained with, successfully.


