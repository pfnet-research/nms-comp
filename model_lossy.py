import numpy as np
import chainer
from chainer import backend
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from msssim import MSSSIM
import materials as M


class LossyModel(chainer.Chain):
    def __init__(self, c_list, q_num):
        super(LossyModel, self).__init__()
        self.q_num = q_num
        self.lossfunc = MSSSIM()

        with self.init_scope():
            self.analyzer = Analyzer(c_list=c_list)
            self.synthesizer = Synthesizer(c_list=c_list)

    def __call__(self, x):
        x = M.preprocessing(x)
        h = self.encode(x)
        y = self.decode(h)
        loss = F.mean(self.lossfunc(y, x))
        reporter.report({'MSSSIM': loss}, self)
        return loss

    def encode(self, x):
        h = self.analyzer(x)
        h = M.quantizer(h, self.q_num)
        return h

    def decode(self, h):
        h = self.synthesizer(h)
        y = M.final_quantizer(h)
        return y


class Analyzer(chainer.Chain):
    def __init__(self, c_list=(0, 0, 4, 32)):
        super(Analyzer, self).__init__()

        c1, c2, c3, c4 = c_list
        self.c1, self.c2, self.c3 = c1, c2, c3

        flag2 = any([c2, c3, c4])
        flag3 = any([c3, c4])
        flag4 = any([c4])

        with self.init_scope():
            self.conv = L.Convolution2D(12, 128, ksize=3, pad=1)
            self.b0 = ABlock(128, 64)
            self.b1 = ABlock(256, 128 * flag2 + c1)
            self.b2 = ABlock(512, 256 * flag3 + c2) if flag2 else empty
            self.b3 = ABlock(1024, 512 * flag4 + c3) if flag3 else empty
            self.b4 = ABlock(2048, c4) if flag4 else empty
            self.bn1 = L.BatchNormalization(c1, use_gamma=False, use_beta=False) if c1 != 0 else empty
            self.bn2 = L.BatchNormalization(c2, use_gamma=False, use_beta=False) if c2 != 0 else empty
            self.bn3 = L.BatchNormalization(c3, use_gamma=False, use_beta=False) if c3 != 0 else empty
            self.bn4 = L.BatchNormalization(c4, use_gamma=False, use_beta=False) if c4 != 0 else empty

    def __call__(self, x):
        h = x

        h = F.space2depth(h, 2)
        h = self.conv(h)
        h = self.b0(h)
        h = F.space2depth(h, 2)

        h = self.b1(h)
        z1, h = F.split_axis(h, axis=1, indices_or_sections=[self.c1])
        h = F.space2depth(h, 2)

        h = self.b2(h)
        z2, h = F.split_axis(h, axis=1, indices_or_sections=[self.c2])
        h = F.space2depth(h, 2)

        h = self.b3(h)
        z3, h = F.split_axis(h, axis=1, indices_or_sections=[self.c3])
        h = F.space2depth(h, 2)

        z4 = self.b4(h)

        z1 = self.bn1(z1)
        z1 = F.space2depth(z1, 8)
        z2 = self.bn2(z2)
        z2 = F.space2depth(z2, 4)
        z3 = self.bn3(z3)
        z3 = F.space2depth(z3, 2)
        z4 = self.bn4(z4)
        z = F.concat([z1, z2, z3, z4], axis=1)
        return z


class Synthesizer(chainer.Chain):
    def __init__(self, c_list=(0, 0, 4, 32)):
        super(Synthesizer, self).__init__()

        c1, c2, c3, c4 = c_list

        flag2 = any([c2, c3, c4])
        flag3 = any([c3, c4])
        flag4 = any([c4])

        self.idx = [
            c1 * 64,
            c1 * 64 + c2 * 16,
            c1 * 64 + c2 * 16 + c3 * 4
        ]

        with self.init_scope():
            self.b5 = SBlock(c4, 2048) if flag4 else empty
            self.b4 = SBlock(512 * flag4 + c3, 1024) if flag3 else empty
            self.b3 = SBlock(256 * flag3 + c2, 512) if flag2 else empty
            self.b2 = SBlock(128 * flag2 + c1, 256)
            self.b1 = SBlock(64, 128)
            self.conv = L.Convolution2D(128, 12, ksize=3, pad=1)

    def __call__(self, x):
        z1, z2, z3, z4 = F.split_axis(x, indices_or_sections=self.idx, axis=1)
        z3 = F.depth2space(z3, 2)
        z2 = F.depth2space(z2, 4)
        z1 = F.depth2space(z1, 8)

        h = self.b5(z4)

        h = F.depth2space(h, 2)
        h = F.concat([z3, h], axis=1)
        h = self.b4(h)

        h = F.depth2space(h, 2)
        h = F.concat([z2, h], axis=1)
        h = self.b3(h)

        h = F.depth2space(h, 2)
        h = F.concat([z1, h], axis=1)
        h = self.b2(h)

        h = F.depth2space(h, 2)
        h = self.b1(h)

        h = self.conv(h)
        h = F.depth2space(h, 2)
        return h


class ABlock(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(ABlock, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_channels)
            self.conv = L.Convolution2D(in_channels, out_channels, ksize=3, pad=1)

    def __call__(self, x):
        h = x
        h = self.bn(h)
        h = F.relu(h)
        h = self.conv(h)
        return h


class SBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(SBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, ksize=3, pad=1)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = x
        h = self.conv(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


def empty(x):
    xp = backend.get_array_module(x)
    s0, _, s2, s3 = x.shape
    return F.identity(xp.empty((s0, 0, s2, s3), dtype=np.float32))
