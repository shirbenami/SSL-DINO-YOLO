import copy
import torch
import torchvision
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv

class PoolHead(nn.Module):
  """ Apply average pooling to the outputs. Adapted from Classify head."""
  def __init__(self, f, i, c1):
    super().__init__()
    self.f = f  # receive the outputs from these layers
    self.i = i  # layer number
    self.conv = Conv(c1, 1280, 1, 1, None, 1) #c1 = numbers of the channels in the input, 1280 = number of the channels in the output after conv, 1 = kernel size
    self.avgpool = nn.AdaptiveAvgPool2d(1) #output size of the image after pooling

  def forward(self, x):
    return self.avgpool(self.conv(x))