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

class VisionCanvas(nn.Module):
    def __init__(self, canvas, num_channels=3, width=224, height=224):
        super(VisionCanvas, self).__init__()

        self.num_channels = num_channels
        self.width = width
        self.height = height

#        canvas = torch.zeros(1, num_channels, width, height)
        self.register_buffer('canvas', canvas)

    # Should not be called besides beginning; will mostly be calling on fresh ones, anyway.
    def store(self, imgs):
        self.canvas = imgs + self.canvas

    # should not be used; included for consistency with nn.Module type
    def forward(self, imgs):
        self.store(self, imgs)
        return self.canvas


class VisionCanvases(nn.Module):
    def __init__(self, num_canvases=3, num_channels=3, width=224, height=224):
        super(VisionCanvases, self).__init__()

        self.num_canvases=num_canvases
        self.num_channels=num_channels
        self.width=width
        self.height=height

        self.empty = True
        self.canvases = nn.Sequential(*[VisionCanvas(torch.zeros(1, num_channels, width, height)) for i in range(num_canvases)])

    def get_device(self):
        return self.canvases[0].canvas.device

    def is_empty(self):
        return self.empty

#    def to(self, device):
#        self.device=device
#        if not (self.canvases is None):
#            for i in range(self.num_canvases):
#                self.canvases[i].to(device)
#        return self
#
#    def cpu(self):
#        return self.to('cpu')
#
#    def cuda(self):
#        return self.to('cuda')
#
    def store(self, img):
        b = img.size()[0]      
        if self.is_empty():
            for i in range(self.num_canvases):
                self.canvases[i].store(img)
        else:
            device = self.get_device()
            self.canvases = self.canvases[1:] # drop oldest
            self.canvases.append(VisionCanvas(torch.zeros(1, self.num_channels, self.width, self.height, device=device)))
            self.canvases[-1].store(img)
        self.empty = False

    # should not be used; included for consistency with nn.Module type
    def forward(self, img):
        self.store(img)
        return self.canvases[0].canvas


