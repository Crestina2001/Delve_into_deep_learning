from network import torch
from utils import Accumulator, accuracy, evaluate_accuracy

def train_epoch(net, train_iter, loss, updater):
    """The training loop defined in Chapter 3."""
    # Set the model to the training mode, turning on some layers like dropout and computing gradients
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        # shape of y_hat: batch_size * num_outputs
        # shape of y: batch_size * 1
        # shape of output: batch_size * 1
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            # Actually, this is equivalent to using reduction = 'mean'(by default) in nn.CrossEntropyLoss
            # that is: loss = nn.CrossEntropyLoss()
            # If using l.backward() here, you will get an error:
            # RuntimeError: grad can be implicitly created only for scalar outputs
            l.mean().backward()
            updater.step()
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model (defined in Chapter 3)."""
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch: {epoch+1}, loss: {train_metrics[0]:2f}, train acc: {train_metrics[1]:2f}, test acc: {test_acc:2f}')