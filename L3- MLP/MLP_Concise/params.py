from utils import load_data_fashion_mnist, init_weights
from network import mlp, torch, nn

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
net=mlp
net.apply(init_weights)

# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
loss = nn.CrossEntropyLoss(reduction='none')

# net.parameters() is equivalent to params in scratch version
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10