import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct/len(y_pred)
    return acc

def accuracy_fn_variance(y_true, y_pred, variance_allowed=0.025):
    variance = (torch.max(y_true) - torch.min(y_true)).item() * variance_allowed
    deltas = torch.abs(y_pred - y_true)
    correct = torch.le(deltas, variance).sum().item()
    acc = correct/len(y_pred)
    return acc

def accuracy_fn_regression(y_true, y_pred):
    # deltas = torch.abs(y_pred - y_true)
    # deltas = deltas ** 2
    # sum_acc = (deltas / y_true).sum().item()
    # acc = 1-sum_acc/len(y_true)
    try:
        acc = 1-mean_absolute_percentage_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    except ValueError:
        acc = -641967480526698.00
    
    return acc

def plot_loss_curves(results):
    """from: Daniel Bourke (https://youtu.be/Z_ikDlimN6A)
    Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


def plot_regression_pred_full(train_data, train_labels, test_data, test_labels, predictions=None, predictions2=None):
    train_data, train_labels = train_data.type(torch.float32), train_labels.type(torch.float32)

    plt.figure(figsize=(train_data.max() * 4, train_labels.max() * 4))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions != None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    if predictions2 != None:
        plt.scatter(train_data, predictions2, c="r", s=4, label="Predictions 2")

    plt.show()