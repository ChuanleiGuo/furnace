from .core import *

def cut_model(model, cut):
    return list(model.children())[:cut] else [model]
