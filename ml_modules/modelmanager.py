"""

"""

from torch import nn
from pathlib import Path
import os
import torch
import logging

class ModelManager():
    """
    Loads and saves models, compares them to already existing ones
    """
    def __init__(self, logging1):
        if logging1 != None:
            self.logging = logging1
        else:
            self.logging = logging

    def _mk_model_path(self, dir, subdir, name=None):
        """
        create path to model or to directory for model
        """
        model_path = Path(dir)
        if subdir != None:
            model_path = model_path / subdir

        if name != None:
            model_path = model_path / name

        return model_path

    def save(self, model, name="{modelname}", dir="models", subdir=None, loss=None, acc=None, compare_saved_metric="loss"):
        """
        Saves a given models state_dict
        Returns path
        loss must be specified in default name
        compare_saved_metric: None (default), "loss" or "acc"
        """
        if "{loss}" in name:
            assert loss != None, "Name got {loss} substring to fill, loss therefore needs to be specified"
        if "{acc}" in name:
            assert acc != None, "Name got {acc} substring to fill, acc therefore needs to be specified"

        assert compare_saved_metric in (False, "loss", "acc"), f"Unsupported argument for compare_saved_metric: {compare_saved_metric}"

        if name == "{modelname}":
            name = model.__class__.__name__
            
        save = True
        if compare_saved_metric != False:
            # compare with other saved models
            compare_files = []
            dirpath = self._mk_model_path(dir, subdir)
            for file in os.listdir(dirpath):
                filename = os.fsdecode(file)
                if filename.split(".")[-1] in ("pth", "pt") and filename.startswith(name) and compare_saved_metric in filename:
                    filename_metric = filename.split(compare_saved_metric + "=")[1]
                    filename_metric = filename_metric.split("_")[0]
                    if filename_metric.isnumeric():
                        compare_files.append(int(filename_metric))

            if compare_saved_metric == "loss":
                if not any([metric < loss for metric in compare_files]):
                    save = False
            else:
                if not any([metric > acc for metric in compare_files]):
                    save = False

        if save:
            name = name + "" if loss == None else f"_loss={loss:.3f}" + "" if acc == None else f"_acc={acc:.2f}" + ".pth"
            path = self._mk_model_path(dir, subdir, name)
            torch.save(model.state_dict(), path)
            self.logging.info(f"Saving model at path {path}")
            return path
        else:
            self.logging.info(f"Not saving model as better ones have been found")
            return None


    def load(self, name=None, dir="models", subdir=None, load_best_metric=False):
        """
        Loads a models state_dict
        Returns the state_dict, which needs to be put into an instantiated model with model.load_state_dict()
        load_best: None (default), "loss" or "acc"
        """
        assert name != None or load_best_metric != False, "Specify a name or load the best model"
        
        assert load_best_metric in (False, "loss", "acc"), f"Unsupported argument for load_best_metric: {load_best_metric}"

        if load_best_metric:
            dirpath = self._mk_model_path(dir, subdir)
            
            best_model = []
            for file in os.listdir(dirpath):
                filename = os.fsdecode(file)
                if filename.split(".")[-1] in ("pth", "pt") and filename.startswith(name) and load_best_metric in filename:
                    filename_metric = filename.split(load_best_metric + "=")[1]
                    filename_metric = filename_metric.split("_")[0]
                    if filename_metric.isnumeric():
                        if best_model == []:
                            best_model = [int(filename_metric), filename]
                            continue
                        if load_best_metric == "loss":
                            if int(filename_metric) < best_model[0]:
                                best_model = [int(filename_metric), filename]
                        else:
                            if int(filename_metric) > best_model[0]:
                                best_model = [int(filename_metric), filename]

            name = best_model[1]
        
        path = self._mk_model_path(dir, subdir, name)
        self.logging.info(f"Loading model from path {path}")
        return torch.load(path)
            
    def test(self):
        """
        Test function. If successful, prints "test successful"
        """
        print("test successful")

