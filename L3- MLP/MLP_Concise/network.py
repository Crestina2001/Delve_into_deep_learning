import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn

mlp = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))