import torch
import matplotlib.pyplot as plt

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
    print(y_true[:10])
    print(y_pred[:10])
    deltas = torch.abs(y_pred - y_true)
    print(deltas[:10])
    sum_acc = (deltas / y_true).sum().item()
    print(sum_acc)
    acc = sum_acc/len(y_true)
    print(acc)
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