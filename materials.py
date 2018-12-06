import numpy as np
import chainer
from chainer import backend
import chainer.functions as F


def snake(x, alpha=1.):
    return x - alpha * F.sin(2 * np.pi * x) / (2 * np.pi)


def round_data(x):
    xp = backend.get_array_module(x.data)
    h = x
    h.data = xp.asarray(xp.rint(h.data), dtype=np.float32)
    return h


def quantizer(x, q_num):
    h = x
    h = h / 4
    h = F.clip(h, 0., 1.)
    h = h * (q_num - 1)
    h = snake(h, alpha=0.5)
    qh = round_data(h)
    return qh


def final_quantizer(x):
    h = F.sigmoid(x)
    h = h * 255
    h = snake(h, alpha=0.5)
    h = round_data(h)
    h = h / 255
    return h


def unpooling_2d(x, ksize):
    return F.unpooling_2d(x, ksize=(ksize, ksize), outsize=(x.shape[2] * ksize, x.shape[3] * ksize))


def preprocessing(x):
    x = x / 255
    x = x.astype(np.float32)
    x = x.transpose((0, 3, 1, 2))
    if chainer.config.train:
        x = x[:, :, :, ::-1] if np.random.rand() > 0.5 else x  # LR flip
    return x
