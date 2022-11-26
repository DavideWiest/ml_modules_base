import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from .modelmanager import ModelManager
import json


def train_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device):
    model.train()

    train_loss, train_acc = 0,0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        y_pred = model(x)# .squeeze()

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc



def test_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, accuracy_fn, device):

    model.eval()

    with torch.inference_mode():
        test_loss, test_acc = 0, 0

        for x_test, y_test in dataloader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            test_pred = model(x_test).squeeze().to(device)
            # test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc



def train_full_fn(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, accuracy_fn, epochs: int, device, save_each=2, save_results_location="results.json", compare_saved_metric="loss", early_stop_epoch=None):
    """
    wrapper function to train and test model
    """
    
    mm = ModelManager()
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, optimizer, accuracy_fn, device)
        results["train_loss"].append(float(train_loss))
        results["train_acc"].append(float(train_acc))
        results["test_loss"].append(float(test_loss))
        results["test_acc"].append(float(test_acc))

        print(f"\n\nEpoch {epoch} | train loss: {train_loss:.3f} | train acc: {(train_acc*100):.2f}% | test loss: {test_loss:.3f} | test acc: {(test_acc*100):.2f}% \n")

        with open(save_results_location, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        if save_each != None:
            if epochs % save_each == 0:
                mm.save(model, loss=test_loss, acc=test_acc, compare_saved_metric=compare_saved_metric)
        
        if early_stop_epoch != None:
            recent_test_loss = min(results["test_loss"][-5:])
            recent_test_acc = max(results["test_acc"][-5:])
            

    
    return results








