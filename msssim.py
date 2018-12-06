import numpy as np
import chainer.functions as F
from chainer import backend


def _gaussian_filter(size, sigma):
    x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    gauss = np.exp(-((x * x) / (2.0 * sigma * sigma)))
    gauss_2d = np.tile(gauss, (size, 1)) * np.tile(gauss[:, None], (1, size))
    return gauss_2d / np.sum(gauss_2d)


class SSIM:
    def __init__(self, window_size=11, sigma=1.5, max_val=1, k1=0.01, k2=0.03):
        self.window_size = window_size
        self.sigma = sigma
        self.c1 = (k1 * max_val) ** 2
        self.c2 = (k2 * max_val) ** 2
        g_filter = _gaussian_filter(self.window_size, self.sigma)
        self.window = np.asarray(g_filter, dtype=np.float32)[None, None, :, :]

    def __call__(self, x0, x1, cs_map=False):
        xp = backend.get_array_module(x0.data)
        assert x0.shape[1] == 1, 'x0.shape[1] must be 1'
        self.window = xp.asarray(self.window)

        mu0 = F.convolution_2d(x0, self.window)
        mu1 = F.convolution_2d(x1, self.window)
        sigma00 = F.convolution_2d(x0 * x0, self.window)
        sigma11 = F.convolution_2d(x1 * x1, self.window)
        sigma01 = F.convolution_2d(x0 * x1, self.window)

        mu00 = mu0 * mu0
        mu11 = mu1 * mu1
        mu01 = mu0 * mu1
        sigma00 = sigma00 - mu00
        sigma11 = sigma11 - mu11
        sigma01 = sigma01 - mu01

        v1 = 2 * sigma01 + self.c2
        v2 = sigma00 + sigma11 + self.c2

        if cs_map:
            cs = v1 / v2
            cs = F.mean(cs, axis=(1, 2, 3))
            return cs

        w1 = 2 * mu01 + self.c1
        w2 = mu00 + mu11 + self.c1

        ssim = (w1 * v1) / (w2 * v2)
        ssim = F.mean(ssim, axis=(1, 2, 3))

        return ssim


class MSSSIM:  # need 256x256 at least
    def __init__(self, max_val=1, transpose=True):
        self.ssim_func = SSIM(max_val=max_val)
        self.weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.level = len(self.weight)
        self.transpose = transpose

    def __call__(self, x0, x1):  # (B,C,H,W)
        assert x0.shape[1] == x1.shape[1], 'x0.shape[1] != x1.shape[1]'
        sb, sc, sh, sw = x0.shape
        h0 = x0.reshape(sb * sc, 1, sh, sw)
        h1 = x1.reshape(sb * sc, 1, sh, sw)
        msssim = self.calc_msssim(h0, h1)
        if self.transpose:
            return 1 - msssim
        return msssim

    def calc_msssim(self, x0, x1):
        msssim = 1
        for i in range(self.level - 1):
            cs = self.ssim_func(x0, x1, cs_map=True)
            cs = F.clip(cs, 0., np.inf)
            msssim *= cs ** self.weight[i]
            x0 = F.average_pooling_2d(x0, 2)
            x1 = F.average_pooling_2d(x1, 2)

        ssim = self.ssim_func(x0, x1)
        ssim = F.clip(ssim, 0., np.inf)
        msssim *= ssim ** self.weight[-1]
        return msssim
