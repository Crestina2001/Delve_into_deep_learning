## Background

Model: $y=w^Tx+b$

Number of Samples: $N$

Number of Attributes: $M$ ($w\in R^M$)

Features: $$X=\begin{pmatrix}
x_{1}^T\\
x_{2}^T\\
...\\
x_{N}^T\end{pmatrix}_{N*M}$$

Labels: $$y=\begin{pmatrix}
y_{1}\\
y_{2}\\
...\\
y_{N}\end{pmatrix}_{N*1}$$

Predicting: $\hat{y}=Xw+b$

## Step 1: Data Loading

```python
from torch.utils import data
'''
TensorDataset is similar to Python zip
Function of TensorDataset:
Each sample will be retrieved by indexing tensors along the first dimension.
Parameters:
tensors (Tensor) – tensors that have the same size of the first dimension.
examples shown below

Features and Labels are all torch.tensor, with dimensions defined above
'''
dataset = data.TensorDataset(Features,Labels)
'''
The most important argument of DataLoader constructor is dataset, which indicates a dataset object to load data from. PyTorch supports two different types of datasets:
(1)map-style datasets
(2)iterable-style datasets.
Here, dataset is a map-style dataset
'''
data_iter = data.DataLoader(dataset = dataset,batch_size = batch_size,shuffle=True)
```

```python
# Example of torch.utils.data.TensorDataset
import torch
from torch.utils.data import TensorDataset
a = torch.tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
b = torch.tensor([1,2,3,4])
train_data = TensorDataset(a,b)
print(train_data[0:4])
'''
Results:
(tensor([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]]), tensor([1, 2, 3, 4]))

'''
```

Now, Let's rewrite DataLoader with python code

```python
import random
import torch
'''
Features and Labels are all torch.tensor
'''
def DataLoader(Features, Labels, batch_size):
    # num_examples equals N
    num_examples=len(Features)
    # Get a list of [0,1,...,N-1]
    indices=list(range(num_examples))
    # Shuffle the list
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        # Get a slice of indices from i to i+batch_size(if not reaching the end)
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        # Slice Features and Labels using batch_indices
        yield Features[batch_indices],Labels[batch_indices]
```

## Step 2: Creating the NN

```python
from torch import nn
'''
Creating a Network

nn.Linear
Def: torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
Parameters:
in_features (int) – size of each input sample
out_features (int) – size of each output sample
bias (bool) – If set to False, the layer will not learn an additive bias. Default: True
Variables:
weight (torch.Tensor) – the learnable weights of the module of shape (out_features,in_features). 
bias – the learnable bias of the module of shape (out_features)
Example of its Usage is shown below

nn.Sequential
Def: torch.nn.Sequential(*args: Module)
'''
net = nn.Sequential(nn.Linear(2,1))
# Para initialization
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
```

```python
# Example of usage of torch.nn.Linear
from torch import nn
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
# Result: torch.Size([128, 30])
```

Now, Let's rewrite torch.nn.Linear with python code

```python
import torch
def Linear(X,w,b):
    # Matrix Multiplication of X and w
    return torch.matmul(X,w)+b
```

## Step 3: Define a Loss Function

```python
from torch import nn
'''
Def: torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input x and target y.
Parameters:
reduction='mean' or 'sum'. If reduction='mean', then return the mean squared error, otherwise return the summation of squared error
Shape:
Input: (*), where * means any number of dimensions.
Target: (*), same shape as the input.
'''
loss = nn.MSELoss()
```

Now, Let's rewrite torch.nn.MSELoss with python code

```python
def MSELoss(input, target, reduction='mean'):
    return ((input - target)**2).mean()
```

## Step 4: Define an Optimizer

```python
import torch
'''
Def: torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False)
'''
trainer = torch.optim.SGD(net.parameters(),lr=0.03)
```

Now, Let's rewrite torch.optim.SGD with python code

```python
import torch
# params means parameters, namely [w,b] in Linear Regression, while lr stands for learning rate
def sgd(params,lr):
    # we don't need to compute gradients here, so we call torch.no_grad()
    with torch.no_grad():
        for param in params:
            # param.grad(gradients of param) is not computed here, 
            # but when we call loss.backward()(shown later)
            param-=lr*param.grad
            # Gradients of param will accumulate, so we need to reset it to zero everytime
            param.grad.zero_()
```

## Step 5: Training

```python
# Training epochs
num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        # Actually, pytorch doesn't include param.grad.zero_() in their sgd implementation, 
        # so you need to call it explicitly
        trainer.zero_grad()
        # This is the backward mentioned above. 
        # In this step, pytorch will compute derivative of l(loss) w.r.t parameters(w,b)
        l.backward()
        # This step mainly conducts 'param-=lr*param.grad'
        trainer.step()
    l=loss(net(features),labels)
    print(f'epoch: {epoch+1}, loss: {l:f}')
```

## Full Code(Runnable)

```python
import torch
from torch.utils import data
from torch import nn

# Synthesize data
def synthetic_data(w,b,num_examples):
    X=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(X,w)+b
    # Add noise
    y+=torch.normal(0,0.1,y.shape)
    return X,y.reshape(-1,1)

# Ground Truth of w and b
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# Prepare Synthetic Dataset
features, labels = synthetic_data(true_w, true_b, 1000)

# Load the Dataset into data_iter
batch_size=10
dataset=data.TensorDataset(features,labels)
data_iter=data.DataLoader(dataset,batch_size,shuffle=True)

# Create the NN
net=nn.Sequential(nn.Linear(2,1))

# Parameter Initialization
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

# Define the Loss
loss=nn.MSELoss()

# Optimizer
trainer=torch.optim.SGD(net.parameters(),lr=0.03)

# Training
num_epochs=20
for epoch in range(num_epochs):
    # X and y are the samples and labels from one batch
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    print(f'epoch: {epoch+1}, loss: {l:f}')

# Now you can check that the parameters are quite close to the ground truth
print(net[0].weight)
print(net[0].bias)
```