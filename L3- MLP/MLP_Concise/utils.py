from network import torch, transforms, torchvision, nn, data

def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    # Convert the image format from PIL to tensor
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def accuracy(y_hat, y):
    """Compute the number of correct predictions of one batch."""
    # in case that y_hat doesn't have axis=1
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # select the max element in each row(standing for each sample)
        y_hat = y_hat.argmax(axis=1)
    # Firstly convert the type of y_hat to the type of y
    # Then generate a bool vector, denoting whether the prediction equals the ground truth
    cmp = y_hat.type(y.dtype) == y
    # Add up all the correct predictions
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        # Set the model to evaluation mode,
        # turning off layers like Dropouts Layers and BatchNorm Layers while evaluating
        net.eval()
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            # y.numel() means number of elements
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

