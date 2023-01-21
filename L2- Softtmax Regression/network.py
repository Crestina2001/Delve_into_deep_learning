import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn


# PyTorch does not implicitly reshape the inputs. Thus we define the flatten
# layer to reshape the inputs before the linear layer in our network
Softmax=nn.Sequential(nn.Flatten(), nn.Linear(784, 10))