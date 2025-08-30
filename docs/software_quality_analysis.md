August 30th, 2025

Main red flags:

1) 'temporal decomposition' (a little) in the 'forward' code
2) 'forward' code and 'framework' code has a lot of if-conditions and bad boolean combinations that could trigger errors
3) the 'random game' methods repeat each other and are hard to follow
4) several shallow modules (I doubt I can fix this)
5) code for angle manipulation duplicated many times; hard to find / use, often re-implemented

On top of that:
1) THe game 'random generation' code is very slow
2) Function / variable names change a little in between uses (in the model, especially.)

~~~~~~~~

Lot was very good. FOr example, I almost never require the higher-level user (me writing frameworks / calling the agent) to understand
deep implementation details (beyond, perhaps, 'reset' and 'soft_reset', and maybe some game settings reading).
I seem to have good interface-implementation division in general

In addition, as bad as the framework repetition is, I'm glad that I wrote it the way I did, greatly reducing prior repetition.

~~~~~~~~

Lessons:

This repo started clean. It was a clean rewrite of the llava-florence efforts. Then it got complex, but I kept making it simpler 
(eg framework standardization). It eventually got away from me, but I did put up a fight. it's just due for a rewrite, with lessons learned.

I do not think I want to rewrite *all* of this, but when I move to the new repo, I think I can make some improvements.
SOme of these are related to the philosophy book, some just to general tidiness (maybe the 'simplicity' book).

Remember to have the simplicity book near you, open, again

0) When moving, keep only the 'enhanced model' file (and rename the classes)
   -- any wrapper-alone classes in that file? Remove.
1) Move the frameworks into a new folder. Perhaps even keep the 'general framework' files in a special 'utils' folder within that
   -- I don't think I can get around the repetitive structure of those functions, but perhaps I can document it
2) A single folder for various angle-related questions.
   -- spend a day after writing it. Go through and replace all the angle-related invocations with calls to this
3) Stare at the 'forward' function and try to think of a redesign of those boolean values
   -- scribble on paper. 
4) Stare at all the 'text gen' functions and try thinking of a redesign
