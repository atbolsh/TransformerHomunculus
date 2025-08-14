August 14th, 2025

Since I am going to make a new repo by / in September, here are some lessons:
 -- use smaller input sizes for easier training (and / or get much much more compute)
 -- Going for very natural language in inputs is not wise, not without transfer learning. Teach naturality later
    -- try to have more clear trigger words
    -- try to have single, predictable output, for easier learning. You are at the edge of what you can train; don't break the model on the basic inputs
 -- any 'input' image should also be an 'output' image in some way
    -- check the option within 'mem_canvas_use' about blue lines, for instance
 -- the complex loss is a wash. It may be used after other (simpler loss) image manipulation is learned, but it's probably not the most fruitful direction


