import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm, trange
from .core import apply_leaf, trainable_params_, to_variable, \
    to_volatile_variable, to_numpy

def cut_model(model, cut):
    return list(model.children())[:cut] if cut else [model]

def set_train_mode(model):
    if (hasattr(model, "running_mean") and (getattr(model, "bn_freeze", False)
            or not getattr(model, "trainable", False))):
        model.eval()
    elif (getattr(model,'drop_freeze',False) and hasattr(model, 'p')
            and ('drop' in type(model).__name__.lower())):
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
        preds,l = stepper.evaluate(to_volatile_variable(x), to_volatile_variable(y))
        loss.append(to_numpy(l))
        res.append([f(preds.data,y) for f in metrics])
    return [np.mean(loss)] + list(np.mean(np.stack(res),0))

def get_prediction(x):
    if isinstance(x, (tuple, list)):
        x = x[0]
    return x.data

def predict(model, dataloader):
    return predict_with_targs(model, dataloader)[0]

def predict_with_targs(model, dataloader):
    model.eval()
    if hasattr(model, "reset"):
        model.reset()
    res = []
    for *x, y in iter(dataloader):
        res.append([get_prediction(model(*to_volatile_variable(x))), y])
    pred, target = zip(*res)
    return to_numpy(torch.cat(pred)), to_numpy(torch.cat(target))

def model_summary(m, input_size):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and module.bias is not None:
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
           not isinstance(module, nn.ModuleList) and
           not (module == m)):
            hooks.append(module.register_forward_hook(hook))

    summary = OrderedDict()
    hooks = []
    m.apply(register_hook)

    if isinstance(input_size[0], (list, tuple)):
        x = [to_gpu(Variable(torch.rand(1,*in_size))) for in_size in input_size]
    else: x = [to_gpu(Variable(torch.rand(1,*input_size)))]
    m(*x)

    for h in hooks: h.remove()
    return summary
