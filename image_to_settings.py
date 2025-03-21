# For now, the file will not cleanly go from image to settings because that is hard, especially the 'differentiable' part
# Instead, several important properties will be quickly extracted, especially agent_x and agent_y, and gold_x and gold_y, as well
# as agent_r and gold_r.

# TODO: experiment if the differentiable mask does well, or you need a better mask to make the gradient actually propagate (eg, multiply the mask by the input or something)
# TODO: once using this for losses, experiment with agnet reconstructions and choose better default epsilon. My money is on 0.01 or 0.03
# TODO: probably needs some torch.no_grad placement

from game import *
from visual_transformer import *
import math

# UNDIFFERENTIABLE
def _get_color_masks_undiff(img_batch, c_tuple, epsilon=3.0/(255*255)): # needs larger epsilon later; test!
    c_vector = torch.tensor(c_tuple, device=img_batch.device)
    diff = img_batch - (c_vector / 255).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    mag = (diff*diff).sum(1)
    #print(torch.min(mag))
    return (mag < epsilon)

# DIFFERENTIABLE
def _get_color_masks_diff_old(img_batch, c_tuple, epsilon=3.0/(255*255)): # needs larger epsilon later; test!
    c_vector = torch.tensor(c_tuple, device=img_batch.device)
    diff = img_batch - (c_vector / 255).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    mag = (diff*diff).sum(1)
    return F.sigmoid((epsilon - mag) / (10 * epsilon)) # other activation functions possible. So is an undifferentiable mask mult by the input, and other tricks

# This looks like my favorite.
# A default, pretty high epsilon is also backed in: something like 0.04
# Start testing here; natural, good, gradient everywhere
def _get_color_masks_diff(img_batch, c_tuple, epsilon=0.04): # needs larger epsilon later; test!
    c_vector = torch.tensor(c_tuple, device=img_batch.device)
    diff = img_batch - (c_vector / 255).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    mag = (diff*diff).sum(1) / 3 # divide by 3 to get max val of 1 and min val of 0
    return F.sigmoid((1.0 + epsilon - mag) ** 100) # other activation functions possible. So is an undifferentiable mask mult by the input, and other tricks

def _get_color_masks_diff_alt2(img_batch, c_tuple, epsilon=3.0/(255*255)): # needs larger epsilon later; test!
    with torch.no_grad():
        c_vector = torch.tensor(c_tuple, device=img_batch.device)
        diff = img_batch - (c_vector / 255).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        mag = (diff*diff).sum(1)
        mask = (mag < epsilon)
        norm = torch.sum(c_vector*c_vector)
    masked = mask.unsqueeze(1) * img_batch / norm
    return (masked * masked).sum(1) # exactly like mask, but all the 1s have gradients wrt to the color blob.

def _get_color_masks_diff_alt3(img_batch, c_tuple, epsilon=3.0/(255*255)): # needs larger epsilon later; test!
    c_vector = torch.tensor(c_tuple, device=img_batch.device)
    diff = img_batch - (c_vector / 255).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    with torch.no_grad():
        mag = (diff*diff).sum(1)
        mask = (mag < epsilon)
        norm = torch.sum(c_vector*c_vector)
    masked_ones = (1.0 - diff) * mask.unsqueeze(1)
    return masked_ones * masked_ones # zero gradient for values at the color; small gradient back to the color, for deviations. Prob need large epsilon

def get_color_masks(img_batch, c_tuple, epsilon=3.0/(255*255), differentiable=True): # needs larger epsilon later; test!
    if differentiable:
        return _get_color_masks_diff(img_batch, c_tuple, epsilon)
    else:
        return _get_color_masks_undiff(img_batch, c_tuple, epsilon)

# this is differentiable with respect to the color mask batch
def get_mask_centers(color_mask_batch, return_radii=False):
    scale = torch.arange(224, device=color_mask_batch) / 224
    y_scale = scale.unsqueeze(0)
    x_scale = scale.unsqueeze(1).unsqueeze(0) # remember, the way it's displayed is a little backwards, unfortunately

    elems = torch.sum(color_mask_batch, dim=(1, 2))

    x = torch.sum(color_mask_batch * x_scale, dim=(1, 2)) / elems
    y = torch.sum(color_mask_batch * y_scale, dim=(1, 2)) / elems
    centers = torch.stack((x, y)).T.contiguous()

    if return_radii:
        radii = torch.sqrt(elems / torch.pi)
        return centers, radii
    else:
        return centers

"""
# Copy from game
# finds the angle from xo, yo to xt, yt
# needed copy to use torch, differentiable functions
def direction_angle(self, xo, yo, xt, yt):
    # Finds the angle from (xo, yo) to (xt, yt)
    # Degenerate cases first, just in case
    if xo == xt:
        xt += 1e-3 # the math will still work out, this just makes the derivative the right sign
    candidate = torch.atan((yt - yo) / (xt - xo)) # derivatives come from here
    if (yt >= yo):
        # Quadrant 1
        if (xt > xo):
            return candidate
        # Quadrant 2
        else:
            return math.pi + candidate # candidate is negative in this case
    else:
        # Quadrant 3
        if (xt < xo):
            return math.pi + candidate # candidate is positive in this case.
        # Quadrant 4
        else:
            return (2 * math.pi) + candidate # candidate is negative
"""

# These are all batches
# See above, commented, for the logic here
def dirction_angle(self, xo, yo, xt, yt):
    xt += (xo == yt) * 1e-3 # This is to avoid degenerate cases, see below
    candidate = torch.atan((yt - yo) / (xt - xo)) # derivatives come from here

    candidate += torch.logical_and((yt >= yo), (xt <= xo)) * math.pi # quadrant 2

    lower_half = (yt < yo)
    candidate += lower_half * math.pi #Q3, Q4
    candidate += torch.logical_and(lower_half, (xt > xo)) * math.pi # Q4
    return candidate

#################

# With utils out of the way, let's write the main functions

def get_agent_info(img_batch, epsilon=3.0/(255*255), differentiable=True):
    red_masks = get_color_masks(img_batch, RED, epsilon, differentiable)
    green_masks = get_color_masks(img_batch, GREEN, epsilon, differentiable)

    agent_masks = red_masks + green_masks
    agent_centers, agent_radii = get_mask_centers(agent_masks, True)

    red_centers = get_mask_centers(red_masks, False)
    green_centers = get_mask_centers(green_masks, False) # could use agent_centers, but this is further so more accurate

    directions = direction_angle(green_centers[:, 0], green_centers[:, 1], red_centers[:, 0], red_centers[:, 1])

    return agent_centers, directions, agent_radii

# This one assumes there is only one gold
# THis only works on bare games, and good reconstructions
def get_SINGLE_gold_info(img_batch, epsilon=3.0/(255 * 255), differentiable=True, return_radii=False):
    masks = get_color_masks(img-batch, GOLD, epsilon, differentiable)
    return get_mask_centers(masks, return_radii)

# NOT DIFFERENTIABLE
# Hacky, incomplete way to retrieve all gold from a SINGLE mask
# Can run in a loop through a batch, with a fixed gold_r, if needed
def get_all_gold(gold_mask, gold_r):
    centers = []
    offset = math.ceil(gold_r * 224)
    ys = gold_mask * y_scale
    M = torch.argmax(ys)
    while M > 0:
        sample_x, sample_y = divmod(M.item(), 224)
        # I *could* write a complex function here, but I don't need to by a property of circles.
        center_x = sample_x / 224.0
        center_y = sample_y / 224.0 - gold_r
        centers.append((center_x, center_y))
        ys[sample_x - offset-1:sample_x + offset+1, sample_y - 2*offset-1:sample_y + 1] = 0
        M = torch.argmax(ys)
    return centers

