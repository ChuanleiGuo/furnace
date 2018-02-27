from .core import chain_params

def optimizer_params(param, lr, wd):
    return {
        "params": chain_params(param),
        "lr": lr,
        "weight_decay": wd
    }
