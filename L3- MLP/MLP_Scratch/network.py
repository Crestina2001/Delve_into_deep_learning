import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn

# Implement MLP with one hidden layer
num_inputs, num_outputs, num_hiddens = 784, 10, 256

## nn.Parameter can be removed
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# Input and Output have the same dimensions
def relu(X):
    a=torch.zeros_like(X)
    return torch.max(X,a)

def mlp(X):
    # flatten
    X=X.reshape(-1,num_inputs)
    H=relu(X @ W1 + b1)
    # No need to add a relu layer to the output
    return (H @ W2 + b2)