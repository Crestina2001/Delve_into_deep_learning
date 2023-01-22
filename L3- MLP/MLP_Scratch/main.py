from runner import train
from params import net, train_iter, test_iter, loss, num_epochs, trainer

if __name__ == '__main__':
    train(net, train_iter, test_iter, loss, num_epochs, trainer)