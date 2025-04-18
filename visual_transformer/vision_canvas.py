import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.distributions import Categorical

import math
import copy

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset

from .custom_transformer import *

class VisionCanvases(nn.Module):
    def __init__(self, num_canvases=3, num_channels=3, width=224, height=224, device='cpu'):
        super(VisionCanvases, self).__init__()

        self.current_index = 0
        self.empty = True
        
        self.num_canvases=num_canvases
        self.num_channels=num_channels
        self.width=width
        self.height=height
        
        canvases = torch.zeros(1, device=device) # placeholder
        self.register_buffer('canvases', canvases)

    def get_device(self):
        return self.canvases.device

    def is_empty(self):
        return self.empty

    def store(self, img_batch):
        if self.is_empty():
            nc = img_batch.unsqueeze(0).repeat(self.num_canvases, 1, 1, 1, 1)
            self.register_buffer('canvases', nc)
        else:
            self.current_index = (self.current_index + 1) % self.num_canvases
            # pytorch complains about rewriting in place, so we're left with this immaturity:
            self.canvases[self.current_index] *= 0
            self.canvases[self.current_index] += img_batch
        self.empty = False

    def __getitem__(self, ind):
        if self.is_empty():
            raise ValueError("can't return canvases from an empty container")
        if (ind >= self.num_canvases) or (ind <= -1 - self.num_canvases):
            raise ValueError("index out of bounds")

        newind = (self.current_index - ind) % self.num_canvases
        return self.canvases[newind]

    # should not be used; use store instead
    def forward(self, img_batch):
        self.store(img_batch)
        return self[0] # the stored value

