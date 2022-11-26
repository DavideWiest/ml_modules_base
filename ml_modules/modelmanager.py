"""

"""

from torch import nn
from pathlib import Path
import os
import torch

class ModelManager():
    """
    
    """
    def __init__(self):
        pass

    def _ml_model_path(self, dir, subdir, name):
        """
        
        """
        model_path = Path(dir)
        if subdir != None:
            model_path = model_path / subdir
        model_path = model_path / name
        return model_path

    def save(self, model, name="{modelname}", dir="models", subdir=None, loss=None, acc=None, epoch=None, compare_saved_metric="loss"):
        """
        loss must be specified in default name
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
                if not any([metric > loss for metric in compare_files]):
                    save = False
            else:
                if not any([metric > acc for metric in compare_files]):
                    save = False

        if save:
            path = self._mk_model_path(dir, subdir, name)
            torch.save(model.state_dict(), path)


    def load(self, name=None, dir="models", subdir=None, load_best_metric=False):
        """
        load_best: None (default), "loss" or "acc"
        """
        assert name != None or load_best_metric != False, "Specify a name or load the best model"
        
        assert load_best_metric in (False, "loss", "acc"), f"Unsupported argument for load_best_metric: {load_best_metric}"

        if load_best_metric:
            dirpath = self._mk_model_path(dir, subdir)
            # find best model
        else:
            path = self._mk_model_path(dir, subdir, name)

    def test(self):
        print("test successful")

