import os
import pickle
from itertools import chain

import bcolz
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

conv_dict = {
    np.dtype("int8"): torch.LongTensor,
    np.dtype("int16"): torch.LongTensor,
    np.dtype("int32"): torch.LongTensor,
    np.dtype("int64"): torch.LongTensor,
    np.dtype("float32"): torch.FloatTensor,
    np.dtype("float64"): torch.FloatTensor
}

def to_tensor(x):
    if torch.is_tensor(x):
        res = x
    else:
        x = np.array(np.ascontiguousarray(x))
        if x.dtype in (np.int8, np.int16, np.int32, np.int64):
            res = torch.LongTensor(x.astype(np.int64))
        elif x.dtype in (np.float32, np.float64):
            res = torch.FloatTensor(x.astype(np.float32))
        else:
            raise NotImplementedError(x.dtype)
    return to_gpu(res, async=True)

def create_variable(x, volatile, requires_grad=False):
    if not isinstance(x, Variable):
        x = Variable(x, volatile=volatile, requires_grad=requires_grad)
    return x

def to_variable_(x, requires_grad=False):
    return create_variable(x, False, requires_grad=requires_grad)

def to_variable(x, requires_grad=False):
    return [to_variable_(v, requires_grad) for v in x] if isinstance(x, list) \
        else to_variable_(x, requires_grad)

def to_volatile_variable_(x):
    return create_variable(x, True)

def to_volatile_variable(x):
    return [to_volatile_variable_(o) for o in x] if isinstance(x, list) \
        else to_volatile_variable_(x)

USE_GPU = True
def to_gpu(x, *args, **kwargs):
    return x.cuda(*args, **kwargs) if torch.cuda.is_available() and USE_GPU else x

def to_numpy(x):
    if isinstance(x, (list, tuple)):
        return [to_numpy(o) for o in x]
    if isinstance(x, Variable):
        x = x.data
    return x.cpu().numpy()

def noop(*args, **kwargs):
    return

def one_hot(x, c):
    return np.eye(c)[x]

def partition(x, size):
    return [x[i: i + size] for i in range(0, len(x), size)]

def partition_by_cores(x):
    return partition(x, len(x) // num_cpus() + 1)

def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()

def trainable_params_(model):
    return [p for p in model.parameters() if p.requires_grad]

def chain_params(params):
    if isinstance(params, (list, tuple)):
        return list(chain(*[trainable_params_(o) for o in params]))
    return trainable_params_(params)

def children(model):
    return model if isinstance(model, (list, tuple)) else list(model.children())

def apply_leaf(model, function):
    c = children(model)
    if isinstance(model, nn.Module):
        function(model)
    if len(c) > 0:
        for layer in c:
            apply_leaf(layer, function)

def set_trainable_attr(model, trainable):
    model.trainable = trainable
    for param in model.parameters():
        param.requires_grad = trainable

def set_trainable(layer, trainable):
    apply_leaf(layer, lambda model: set_trainable_attr(model, trainable))

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

class BasicModel():
    def __init__(self, model, name="unnamed"):
        self.model, self.name = model, name

    def get_layer_groups(self, do_fc=False):
        return children(self.model)

class SingleModel(BasicModel):
    def get_layer_groups(self):
        return [self.model]

class SimpleNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return F.log_softmax(l_x, dim=-1)

def save(array, path):
    pickle.dump(array, open(path, 'wb'))

def load(path, encoding="ASCII"):
    return pickle.load(open(path, "rb"), encoding=encoding)

def load_array(path):
    return bcolz.open(path)[:]
