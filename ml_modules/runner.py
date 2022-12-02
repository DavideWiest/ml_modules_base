import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from .modelmanager import ModelManager
import json


def train_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device, pred_argmax=None, logging=None, print_debug=False, scheduler=None, clip_grad_norm_params=None, clip_grad_value_params=None):
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
                    f"First 20 x      elems of batch 0 (train): \n" + str(x[:20]),
                    f"First 20 y      elems of batch 0 (train): \n" + str(y[:20]),
                    f"First 20 y pred elems of batch 0 (train): \n" + str(y_pred[:20])
                ]) + "\n------\n"

                if logging != None:
                    logging.info(debugmsg)
                else:
                    print(debugmsg)

            if isinstance(y_pred, tuple) and isinstance(y, tuple):
                loss = sum([loss_fn(y_pred[i], y[i]) for i in range(len(y_pred))])
            elif isinstance(y_pred, tuple):
                loss = sum([loss_fn(y_pred_pt, y) for y_pred_pt in y_pred])
            else:
                loss = loss_fn(y_pred, y)

            train_loss += loss
            train_acc += accuracy_fn(y, y_pred)

            optimizer.zero_grad()

            loss.backward()

            if clip_grad_norm_params != None and isinstance(clip_grad_norm_params, dict):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm_params.get("max_norm", 2.0), norm_type=clip_grad_norm_params.get("norm_type", 2))
            if clip_grad_value_params != None and isinstance(clip_grad_value_params, dict):
                nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad_value_params.get("clip_value", 1.0))
        

            optimizer.step()

            if scheduler != None:
                scheduler.step()

    elif isinstance(dataloader, tuple):
        x, y = dataloader
        x, y = x.to(device), y.to(device)

        y_pred = model(x).to(device)# .squeeze()
        
        if pred_argmax != None:
            y_pred = y_pred.argmax(dim=pred_argmax)

        if print_debug:
            debugmsg = "\n------\n".join([
                f"First 20 x      elems of batch 0 (train): \n" + str(x[:20]),
                f"First 20 y      elems of batch 0 (train): \n" + str(y[:20]),
                f"First 20 y pred elems of batch 0 (train): \n" + str(y_pred[:20])
            ]) + "\n------\n"

            if logging != None:
                logging.info(debugmsg)
            else:
                print(debugmsg)

        if isinstance(y_pred, tuple) and isinstance(y, tuple):
            loss = sum([loss_fn(y_pred[i], y[i]) for i in range(len(y_pred))])
        elif isinstance(y_pred, tuple):
            loss = sum([loss_fn(y_pred_pt, y) for y_pred_pt in y_pred])
        else:
            loss = loss_fn(y_pred, y)
        
        train_loss += loss
        train_acc += accuracy_fn(y, y_pred)

        optimizer.zero_grad()

        loss.backward()

        if clip_grad_norm_params != None and isinstance(clip_grad_norm_params, dict):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm_params.get("max_norm", 2.0), norm_type=clip_grad_norm_params.get("norm_type", 2))
        if clip_grad_value_params != None and isinstance(clip_grad_value_params, dict):
            nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad_value_params.get("clip_value", 1.0))
        

        optimizer.step()

        if scheduler != None:
            scheduler.step()


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
                
                y_pred = model(x_test).to(device)
                if pred_argmax != None:
                    y_pred = y_pred.argmax(dim=pred_argmax)
                # y_pred = torch.round(torch.sigmoid(test_logits))

                if batch == 0 and print_debug:
                    debugmsg = "\n------\n".join([
                        f"First 20 x      elems of batch 0 (test ): \n" + str(x_test[:20]),
                        f"First 20 y      elems of batch 0 (test ): \n" + str(y_test[:20]),
                        f"First 20 y pred elems of batch 0 (test ): \n" + str(y_pred[:20])
                    ]) + "\n------\n"

                    if logging != None:
                        logging.info(debugmsg)
                    else:
                        print(debugmsg)

                if isinstance(y_pred, tuple) and isinstance(y_test, tuple):
                    loss = sum([loss_fn(y_pred[i], y_test[i]) for i in range(len(y_pred))])
                elif isinstance(y_pred, tuple):
                    loss = sum([loss_fn(y_pred_pt, y_test) for y_pred_pt in y_pred])
                else:
                    loss = loss_fn(y_pred, y_test)

                test_loss += loss
                test_acc += accuracy_fn(y_test, y_pred)
                
        elif isinstance(dataloader, tuple):
            x_test, y_test = dataloader
            x_test, y_test = x_test.to(device), y_test.to(device)
                
            y_pred = model(x_test).to(device)
            if pred_argmax != None:
                y_pred = y_pred.argmax(dim=pred_argmax)
            # y_pred = torch.round(torch.sigmoid(test_logits))

            if print_debug:
                debugmsg = "\n------\n".join([
                    f"First 20 x      elems of batch 0 (test ): \n" + str(x_test[:20]),
                    f"First 20 y      elems of batch 0 (test ): \n" + str(y_test[:20]),
                    f"First 20 y pred elems of batch 0 (test ): \n" + str(y_pred[:20])
                ]) + "\n------\n"

                if logging != None:
                    logging.info(debugmsg)
                else:
                    print(debugmsg)

            if isinstance(y_pred, tuple) and isinstance(y_test, tuple):
                loss = sum([loss_fn(y_pred[i], y_test[i]) for i in range(len(y_pred))])
            elif isinstance(y_pred, tuple):
                loss = sum([loss_fn(y_pred_pt, y_test) for y_pred_pt in y_pred])
            else:
                loss = loss_fn(y_pred, y_test)

            test_loss += loss
            test_acc += accuracy_fn(y_test, y_pred)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc



def train_full_fn(model: nn.Module, train_dataloader, test_dataloader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, accuracy_fn, epochs: int, device, scheduler=None, save_each=2, save_results_location="results.json", compare_saved_metric="loss", early_stop_epoch=None, logging=None, models_dir="models", models_subdir=None, pred_argmax=None, print_debug_each=False, clip_grad_norm_params=None, clip_grad_value_params=None):
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
        
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device, pred_argmax, logging=logging, print_debug=print_debug, scheduler=scheduler, clip_grad_norm_params=clip_grad_norm_params, clip_grad_value_params=clip_grad_value_params)
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









