from utils import load_data_fashion_mnist, init_weights
from network import Softmax, torch, nn

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size)
net=Softmax
net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10