from utils import load_data_fashion_mnist
from network import mlp, torch, nn, params

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
net=mlp

# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(params, lr=0.1)

num_epochs = 10