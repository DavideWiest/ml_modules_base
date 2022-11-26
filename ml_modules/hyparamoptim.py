"""

"""

from torch import nn
from .runner import train_full_fn

class HyParamOptim():
    """
    Optimize hyper-parameters
    """

    def __init__(self, modelclass: nn.Module, modelclass_kwargs, optimizerclass, variance: int, perf_importance: float, learning_rate: float, layers, hidden_units, train_full_fn_kwargs, epochs=5, stop_early_iters=5):
        train_full_fn_kwargs["epochs"] = epochs
        
        self.modelclass = modelclass
        self.variance = variance
        self.perf_importance = perf_importance
        self.epochs = epochs
        self.train_full_fn_kwargs = train_full_fn_kwargs
        self.stop_early_iters = stop_early_iters # stop after x iterations with no improvement
        self.modelclass_kwargs = modelclass_kwargs

        self.values = {
            "lr": self.lr,
            "layers": self.layers,
            "hidden_units": hidden_units
        }

    def test_values(self):
        modelclass_kwargs = self.modelclass_kwargs
        modelclass_kwargs["hidden_units"] = self.values["hidden_units"]

        model = self.modelclass()
        optim = self.optimizer()

        train_full_fn_kwargs = self.train_full_fn_kwargs
        train_full_fn_kwargs["optimizer"] = optim
        train_full_fn_kwargs["model"] = model
        train_full_fn()