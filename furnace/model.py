import numpy as np
import torch.nn as nn

from tqdm import tqdm, trange
from .core import apply_leaf, trainable_params_, to_variable

def cut_model(model, cut):
    return list(model.children())[:cut] if cut else [model]

def set_train_mode(model):
    if (hasattr(model, "running_mean") and (getattr(model, "bn_freeze", False)
            or not getattr(model, "trainable", False))):
        model.eval()
    elif (getattr(m,'drop_freeze',False) and hasattr(m, 'p')
            and ('drop' in type(m).__name__.lower())):
        model.eval()
    else:
        model.train()

def Stepper():
    def __init__(self, model, optimizer, criterion, clip=0, register_func=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.clip = criterion
        self.register_func = register_func
        self.reset(True)

    def reset(self, train=True):
        if train:
            apply_leaf(self.model, set_train_mode)
        else:
            self.model.eval()
        if hasattr(self.model, "reset"):
            self.model.reset()

    def step(self, xs, y):
        extra = []
        output = self.model(*xs)
        if isinstance(output, (list, tuple)):
            output, *extra = output
        self.optimizer.zero_grad()
        loss = raw_loss = self.criterion(output, y)
        if self.register_func:
            loss = self.register_func(output, extra, raw_loss)
        loss.backward()
        if self.clip:
            nn.utils.clip_grad_norm(trainable_params_(self.model), self.clip)
        self.optimizer.step()
        return raw_loss.data[0]

    def evaluate(self, xs, y):
        preds = self.model(*xs)
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        return preds, self.criterion(preds, y)

def fit(model, data, epochs, optimizer, criterion, metrics=None, callbacks=None, stepper=Stepper, **kwargs):
    """Fit a model

    ## Parameters
    model: nn.Module
    data: ModelData
    optimizer: torch.optim
    epochs: int
        number of epochs
    criterion: nn.functional
        loss function
    """

    stepper = stepper(model, optimizer, criterion, **kwargs)
    metrics = metrics or []
    callbacks = callbacks or []
    avg_momentum = 0.98
    batch_num, avg_loss = 0, 0.0
    for callback in callbacks:
        callback.on_train_begin()
    names = ["epoch", "train_loss", "val_loss"] + [f.__name__ for f in metrics]
    output_layout = "{!s:10} " * len(names)

    num_batch = len(data.train_dataloader)
    if epochs < 1:
        num_batch = int(num_batch * epochs)
        epochs = 1

    for epoch in trange(epochs, desc="Epoch"):
        stepper.reset()
        batch_idx = 0
        t = tqdm(iter(data.train_dataloader), leave=False, total=num_batch)
        for (*x, y) in t:
            batch_num += 1
            for callback in callbacks:
                callback.on_batch_begin()
            loss = stepper.step(to_variable(x), to_variable(y))
            avg_loss = avg_loss * avg_momentum + loss * (1 - avg_momentum)
            debias_loss = avg_loss / (1 - avg_momentum ** batch_num)
            t.set_postfix(loss=debias_loss)
            stop = False
            for callback in callbacks:
                stop = stop or callback.on_batch_end(debias_loss)
            if stop:
                return
            if batch_idx > num_batch:
                break
            batch_idx += 1

        vals = validate(stepper, data.val_dataloader, metrics)
        if epoch == 0:
            print(output_layout.format(*names))
        print_stats(epoch, [debias_loss] + vals)
        stop = False
        for callback in callbacks:
            stop = stop or callback.on_epoch_end(vals)
        if stop:
            break

    for callback in callbacks:
        callback.on_train_end()
    return vals

def print_stats(epoch, values, decimals=6):
    layout = "{!s:^10}" + " {!s:10}" * len(values)
    values = [epoch] + list(np.round(values, decimals))
    print(layout.format(*values))

def validate(stepper, dataloader, metrics):
    loss,res = [],[]
    stepper.reset(False)
    for (*x, y) in iter(dataloader):
        preds,l = stepper.evaluate(VV(x), VV(y))
        loss.append(to_np(l))
        res.append([f(preds.data,y) for f in metrics])
    return [np.mean(loss)] + list(np.mean(np.stack(res),0))
