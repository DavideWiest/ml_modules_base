import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from .modelmanager import ModelManager
import json


def train_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device, pred_argmax=None, logging=None, print_debug=False):
    model.train()

    train_loss, train_acc = 0,0

    if isinstance(dataloader, torch.utils.data.DataLoader):
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            y_pred = model(x).to(device)# .squeeze()
            if pred_argmax != None:
                y_pred = y_pred.argmax(dim=pred_argmax)

            if batch == 0 and print_debug:
                debugmsg = "\n------\n".join([
                    f"First 20 x      elems of batch 0 (train): " + str(x[:20]),
                    f"First 20 y      elems of batch 0 (train): " + str(y[:20]),
                    f"First 20 y pred elems of batch 0 (train): " + str(y_pred[:20])
                ])

                if logging != None:
                    logging.debug(debugmsg)
                else:
                    print(debugmsg)

            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += accuracy_fn(y, y_pred)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
    elif isinstance(dataloader, tuple):
        x, y = dataloader
        x, y = x.to(device), y.to(device)

        y_pred = model(x).to(device)# .squeeze()
        if pred_argmax != None:
            y_pred = y_pred.argmax(dim=pred_argmax)

        if print_debug:
            debugmsg = "\n------\n".join([
                f"First 20 x      elems of batch 0 (train): " + str(x[:20]),
                f"First 20 y      elems of batch 0 (train): " + str(y[:20]),
                f"First 20 y pred elems of batch 0 (train): " + str(y_pred[:20])
            ])

            if logging != None:
                logging.debug(debugmsg)
            else:
                print(debugmsg)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y, y_pred)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc


def test_step(model: nn.Module, dataloader, loss_fn: nn.Module, accuracy_fn, device, pred_argmax=None, logging=None, print_debug=False):

    model.eval()

    with torch.inference_mode():
        test_loss, test_acc = 0, 0

        if isinstance(dataloader, torch.utils.data.DataLoader):
            for batch, (x_test, y_test) in enumerate(dataloader):
                x_test, y_test = x_test.to(device), y_test.to(device)
                
                test_pred = model(x_test).to(device)
                if pred_argmax != None:
                    test_pred = test_pred.argmax(dim=pred_argmax)
                # test_pred = torch.round(torch.sigmoid(test_logits))

                if batch == 0 and print_debug:
                    debugmsg = "\n------\n".join([
                        f"First 20 x      elems of batch 0 (test ): " + str(x_test[:20]),
                        f"First 20 y      elems of batch 0 (test ): " + str(y_test[:20]),
                        f"First 20 y pred elems of batch 0 (test ): " + str(test_pred[:20])
                    ])

                    if logging != None:
                        logging.debug(debugmsg)
                    else:
                        print(debugmsg)

                test_loss += loss_fn(test_pred, y_test)
                test_acc += accuracy_fn(y_test, test_pred)
                
        elif isinstance(dataloader, tuple):
            x_test, y_test = dataloader
            x_test, y_test = x_test.to(device), y_test.to(device)
                
            test_pred = model(x_test).to(device)
            if pred_argmax != None:
                test_pred = test_pred.argmax(dim=pred_argmax)
            # test_pred = torch.round(torch.sigmoid(test_logits))

            if batch == 0 and print_debug:
                debugmsg = "\n------\n".join([
                    f"First 20 x      elems of batch 0 (test ): " + str(x_test[:20]),
                    f"First 20 y      elems of batch 0 (test ): " + str(y_test[:20]),
                    f"First 20 y pred elems of batch 0 (test ): " + str(test_pred[:20])
                ])

                if logging != None:
                    logging.debug(debugmsg)
                else:
                    print(debugmsg)

            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_test, test_pred)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc



def train_full_fn(model: nn.Module, train_dataloader, test_dataloader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, accuracy_fn, epochs: int, device, save_each=2, save_results_location="results.json", compare_saved_metric="loss", early_stop_epoch=None, logging=None, models_dir="models", models_subdir=None, pred_argmax=None, print_debug_each=False):
    """
    wrapper function to train and test model
    """

    modelname = model.__class__.__name__
    mm = ModelManager(logging)

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in tqdm(range(epochs)):

        epoch = epoch+1

        print_debug = epoch % print_debug_each == 0 if print_debug_each != False else False
        
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device, pred_argmax, logging=logging, print_debug=print_debug)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, accuracy_fn, device, pred_argmax, logging=logging, print_debug=print_debug)
        results["train_loss"].append(float(train_loss))
        results["train_acc"].append(float(train_acc))
        results["test_loss"].append(float(test_loss))
        results["test_acc"].append(float(test_acc))

        if logging != None:
            logging.info(f"\n\nEPOCH {epoch} | Train loss: {train_loss:.3f} | Train acc: {(train_acc*100):.2f}% | Test loss: {test_loss:.3f} | Test acc: {(test_acc*100):.2f}% \n")
        else:
            print(f"\n\nEPOCH {epoch} | Train loss: {train_loss:.3f} | Train acc: {(train_acc*100):.2f}% | Test loss: {test_loss:.3f} | Test acc: {(test_acc*100):.2f}% \n")


        if save_each != None:
            if epoch % save_each == 0:
                with open(save_results_location, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)

                mm.save(model, modelname=modelname, dir=models_dir, subdir=models_subdir, loss=test_loss, acc=test_acc, compare_saved_metric=compare_saved_metric)
        else:
            with open(save_results_location, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

                
        if early_stop_epoch != None and epoch > early_stop_epoch:
            recent_test_loss = min(results["test_loss"][-5:])
            recent_test_acc = max(results["test_acc"][-5:])

            if recent_test_loss > min(results["test_loss"][:-5]) and recent_test_acc < max(results["test_acc"][:-5]):
                logging.info(f"\nStopping early with EPOCH {epoch} as no improvements in loss and accuracy have been made within the {early_stop_epoch} last epochs.\n")
                break
    
    return results, model









