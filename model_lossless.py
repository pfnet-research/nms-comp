import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
from chainer import reporter
import cpp.range_coder_cpp as RC
import materials as M
from model_lossy import LossyModel


class LosslessModel(chainer.Chain):
    def __init__(self, c_list, q_num, out):
        super(LosslessModel, self).__init__()

        with self.init_scope():
            self.model_lossy = LossyModel(c_list=c_list, q_num=q_num)
            self.model_lossless = LossLess(c_list=c_list, q_num=q_num)

        chainer.serializers.load_npz('{}/model_lossy'.format(out), self.model_lossy)

    def __call__(self, x):
        x = M.preprocessing(x)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            qh = self.model_lossy.encode(x)
        mean_entropy = self.model_lossless(qh)
        reporter.report({'mean_entropy': mean_entropy}, self)
        return mean_entropy

    def fixed_encode(self, x, loop, filename, check=False):
        x = M.preprocessing(x)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            _x = self.xp.pad(x, ((0, 0),
                                 (0, 0),
                                 (0, -(x.shape[2]) % (2 ** 5)),
                                 (0, -(x.shape[3]) % (2 ** 5))), mode='constant')
            qh = self.model_lossy.encode(_x)
            self.model_lossless.fixed_encode(qh, im_shape=x.shape, loop=loop, filename=filename)
        if check:
            return qh

    def fixed_decode(self, loop, filename, check=False):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            qh = self.model_lossless.fixed_decode(loop=loop, filename=filename)
            x = self.model_lossy.decode(qh)
            x = x[:, :, :self.model_lossless.im_h, :self.model_lossless.im_w]
        if check:
            return x.data, qh
        return x.data


class LossLess(chainer.Chain):
    def __init__(self, c_list, q_num):
        super(LossLess, self).__init__()
        xp = self.xp
        self.q_num = q_num
        mid_ch = 512

        c1, c2, c3, c4 = c_list

        self.ct1 = c1
        self.ct2 = c1 + c2
        self.ct3 = c1 + c2 + c3
        self.ct4 = c1 + c2 + c3 + c4

        self.idx = [c1 * 64,
                    c1 * 64 + c2 * 16,
                    c1 * 64 + c2 * 16 + c3 * 4]

        self.lossy_shrink = 5

        with self.init_scope():
            if self.ct1 != 0:
                self.b4_0 = Block(self.ct4, mid_ch, self.ct1, self.q_num)
                self.b4_1 = Block(self.ct4 + self.ct1, mid_ch, self.ct1 * 2, self.q_num)
            if self.ct2 != 0:
                self.b8_0 = Block(self.ct4, mid_ch, self.ct2, self.q_num)
                self.b8_1 = Block(self.ct4 + self.ct2, mid_ch, self.ct2 * 2, self.q_num)
            if self.ct3 != 0:
                self.b16_0 = Block(self.ct4, mid_ch, self.ct3, self.q_num)
                self.b16_1 = Block(self.ct4 + self.ct3, mid_ch, self.ct3 * 2, self.q_num)

            self.b32a_0 = Block(self.ct4, mid_ch, self.ct4, self.q_num)
            self.b32a_1 = Block(self.ct4 * 2, mid_ch, self.ct4 * 2, self.q_num)
            self.b32b_0 = Block(self.ct4, mid_ch, self.ct4, self.q_num)
            self.b32b_1 = Block(self.ct4 * 2, mid_ch, self.ct4 * 2, self.q_num)
            self.b32c_0 = Block(self.ct4, mid_ch, self.ct4, self.q_num)
            self.b32c_1 = Block(self.ct4 * 2, mid_ch, self.ct4 * 2, self.q_num)

            self.b32_0 = [self.b32a_0, self.b32b_0, self.b32c_0]
            self.b32_1 = [self.b32a_1, self.b32b_1, self.b32c_1]

            self.d_pred = chainer.Parameter(initializer=xp.ones(shape=(self.ct4, self.q_num), dtype=np.float32))

    def __call__(self, x):
        xp = self.xp
        info_total = 0

        z1, z2, z3, z4 = xp.split(x.data, indices_or_sections=self.idx, axis=1)
        z3 = F.depth2space(z3, 2).data
        z2 = F.depth2space(z2, 4).data
        z1 = F.depth2space(z1, 8).data

        if self.ct1 != 0:
            h = z1
            h_small = xp.concatenate((z2, M.unpooling_2d(z3, 2).data, M.unpooling_2d(z4, 4).data), axis=1)
            info = self.single_scale([self.b4_0, self.b4_1], '4', h, x_small=h_small)
            info_total += info

        if self.ct2 != 0:
            h = xp.concatenate((z1[:, :, ::2, ::2], z2), axis=1)
            h_small = xp.concatenate((z3, M.unpooling_2d(z4, 2).data), axis=1)
            info = self.single_scale([self.b8_0, self.b8_1], '8', h, x_small=h_small)
            info_total += info

        if self.ct3 != 0:
            h = xp.concatenate((z1[:, :, ::4, ::4], z2[:, :, ::2, ::2], z3), axis=1)
            h_small = z4
            info = self.single_scale([self.b16_0, self.b16_1], '16', h, x_small=h_small)
            info_total += info

        h = xp.concatenate((z1[:, :, ::8, ::8], z2[:, :, ::4, ::4], z3[:, :, ::2, ::2], z4), axis=1)

        for i, name in zip(range(3), 'abc'):
            info = self.single_scale([self.b32_0[i], self.b32_1[i]], '32{}'.format(name), h)
            info_total += info
            h = h[:, :, ::2, ::2]

        info_total += self.final(h)

        mean_entropy = info_total / x[0].data.size
        return mean_entropy

    def half_scale(self, func, _h, _t):
        h = func(_h).transpose(0, 2, 3, 1).reshape((_t.size, -1))
        t = _t.astype(np.int32).transpose(0, 2, 3, 1).flatten()
        entropy = F.softmax_cross_entropy(h, t) * math.log2(math.e)
        info = entropy * _t[0].size
        return info, entropy

    def single_scale(self, func_list, name, x, x_small=None):
        xp = self.xp
        h = x[:, :, ::2, ::2]
        if x_small is not None:
            h = xp.concatenate((h, x_small), axis=1)
        t0 = x[:, :, 1::2, 1::2]
        t1 = xp.concatenate((x[:, :, ::2, 1::2], x[:, :, 1::2, ::2]), axis=1)
        info_0, entropy = self.half_scale(func_list[0], h, t0)
        info_1, entropy = self.half_scale(func_list[1], xp.concatenate((h, t0), axis=1), t1)
        return info_0 + info_1

    def final(self, x):
        xp = self.xp
        x = x.astype(np.int32)

        p = F.softmax(self.d_pred, axis=1)

        h = xp.eye(self.q_num)[x]
        d = xp.sum(h, axis=(0,2,3))

        info = - F.sum(d * F.log2(p)) / x.shape[0]
        return info

    def single_scale_encode(self, func_list, pt_list, x, x_small=None):
        xp = self.xp
        x00 = x[:, :, ::2, ::2]
        x01 = x[:, :, ::2, 1::2]
        x10 = x[:, :, 1::2, ::2]
        x11 = x[:, :, 1::2, 1::2]

        x00s = x00 if (x_small is None) else xp.concatenate((x00, x_small), axis=1)
        x11p = xp.pad(x11, [(0, i - j) for i, j in zip(x00.shape, x11.shape)], mode='constant')

        h = x00s
        p11 = func_list[0](h).data
        p11 = p11[:, :, :x11.shape[2], :x11.shape[3]]
        p11 = p11.transpose((0, 2, 3, 1)).reshape((x11.size, -1))
        p11 = F.softmax(p11, axis=1).data
        t11 = x11.astype(np.int32).transpose((0, 2, 3, 1)).flatten()

        h = xp.concatenate((x00s, x11p), axis=1)
        p01, p10 = xp.split(func_list[1](h).data, axis=1, indices_or_sections=2)
        p01 = p01[:, :, :x01.shape[2], :x01.shape[3]]
        p01 = p01.transpose((0, 2, 3, 1)).reshape((x01.size, -1))
        p01 = F.softmax(p01, axis=1).data
        p10 = p10[:, :, :x10.shape[2], :x10.shape[3]]
        p10 = p10.transpose((0, 2, 3, 1)).reshape((x10.size, -1))
        p10 = F.softmax(p10, axis=1).data
        t01 = x01.astype(np.int32).transpose((0, 2, 3, 1)).flatten()
        t10 = x10.astype(np.int32).transpose((0, 2, 3, 1)).flatten()

        pt_list.append((cuda.to_cpu(p10), cuda.to_cpu(t10)))
        pt_list.append((cuda.to_cpu(p01), cuda.to_cpu(t01)))
        pt_list.append((cuda.to_cpu(p11), cuda.to_cpu(t11)))
        return x00s

    def single_scale_decode(self, func_list, decoder, x, out_shape, x_small=None):
        xp = self.xp
        shape00 = (1, out_shape[1], math.ceil(out_shape[2] / 2), math.ceil(out_shape[3] / 2))
        shape01 = (1, out_shape[1], math.ceil(out_shape[2] / 2), math.floor(out_shape[3] / 2))
        shape10 = (1, out_shape[1], math.floor(out_shape[2] / 2), math.ceil(out_shape[3] / 2))
        shape11 = (1, out_shape[1], math.floor(out_shape[2] / 2), math.floor(out_shape[3] / 2))

        h = x
        p11 = func_list[0](h).data
        p11 = p11[:, :, :shape11[2], :shape11[3]]
        p11 = p11.transpose((0, 2, 3, 1)).reshape((np.prod(shape11), -1))
        p11 = F.softmax(p11, axis=1).data
        t11 = xp.array(decoder.call(cuda.to_cpu(p11)))
        t11 = t11.reshape((1, shape11[2], shape11[3], shape11[1])).transpose((0, 3, 1, 2)).astype(np.float32)
        t11p = xp.pad(t11, [(0, i - j) for i, j in zip(shape00, shape11)], mode='constant')

        h = xp.concatenate((x, t11p), axis=1)
        p01, p10 = xp.split(func_list[1](h).data, axis=1, indices_or_sections=2)
        p01 = p01[:, :, :shape01[2], :shape01[3]]
        p01 = p01.transpose((0, 2, 3, 1)).reshape((np.prod(shape01), -1))
        p01 = F.softmax(p01, axis=1).data
        t01 = xp.array(decoder.call(cuda.to_cpu(p01)))
        t01 = t01.reshape((1, shape01[2], shape01[3], shape01[1])).transpose((0, 3, 1, 2)).astype(np.float32)
        t01p = xp.pad(t01, [(0, i - j) for i, j in zip(shape00, shape01)], mode='constant')
        p10 = p10[:, :, :shape10[2], :shape10[3]]
        p10 = p10.transpose((0, 2, 3, 1)).reshape((np.prod(shape10), -1))
        p10 = F.softmax(p10, axis=1).data
        t10 = xp.array(decoder.call(cuda.to_cpu(p10)))
        t10 = t10.reshape((1, shape10[2], shape10[3], shape10[1])).transpose((0, 3, 1, 2)).astype(np.float32)
        t10p = xp.pad(t10, [(0, i - j) for i, j in zip(shape00, shape10)], mode='constant')

        t = xp.concatenate((x[:, :out_shape[1]], t01p, t10p, t11p), axis=1)
        t = F.depth2space(t, 2).data
        t = t[:, :out_shape[1], :out_shape[2], :out_shape[3]]
        return t

    def fixed_encode(self, x, im_shape, loop=3, filename='test.bin'):
        if x.shape[0] != 1:
            print('batchsize have to be 1')
            exit(-1)
        xp = self.xp
        pt_list = []
        # print('init:\n{}'.format(x.data[0, 0, :5, :5]))

        z1, z2, z3, z4 = xp.split(x.data, indices_or_sections=self.idx, axis=1)
        z3 = F.depth2space(z3, 2).data
        z2 = F.depth2space(z2, 4).data
        z1 = F.depth2space(z1, 8).data

        z_concat = z1
        if self.ct1 != 0:
            h = _crop(z_concat, im_shape=im_shape, shrink=3)
            # print('d4:\n{}'.format(h[0, 0, :5, :5]))
            h_small = xp.concatenate((z2, M.unpooling_2d(z3, 2).data, M.unpooling_2d(z4, 4).data), axis=1)
            self.single_scale_encode([self.b4_0, self.b4_1], pt_list, h, x_small=h_small)

        z_concat = xp.concatenate((z_concat[:, :, ::2, ::2], z2), axis=1)
        if self.ct2 != 0:
            h = _crop(z_concat, im_shape=im_shape, shrink=4)
            # print('d8:\n{}'.format(h[0, 0, :5, :5]))
            h_small = xp.concatenate((z3, M.unpooling_2d(z4, 2).data), axis=1)
            self.single_scale_encode([self.b8_0, self.b8_1], pt_list, h, x_small=h_small)

        z_concat = xp.concatenate((z_concat[:, :, ::2, ::2], z3), axis=1)
        if self.ct3 != 0:
            h = _crop(z_concat, im_shape=im_shape, shrink=5)
            # print('d16:\n{}'.format(h[0, 0, :5, :5]))
            h_small = z4
            self.single_scale_encode([self.b16_0, self.b16_1], pt_list, h, x_small=h_small)

        h = xp.concatenate((z_concat[:, :, ::2, ::2], z4), axis=1)
        for i in range(loop):
            # print('d32_{}:\n{}'.format(i, h[0, 0, :5, :5]))
            h = self.single_scale_encode([self.b32_0[i], self.b32_1[i]], pt_list, h)

        # print('df:\n{}'.format(h[0, 0, :5, :5]))
        t_init = h.astype(np.int32).transpose(0, 2, 3, 1).flatten()
        p_init = xp.tile(F.softmax(self.d_pred, axis=1).data, (h.shape[2] * h.shape[3], 1))
        pt_list.append((cuda.to_cpu(p_init), cuda.to_cpu(t_init)))

        encoder = RC.Encoder(np.uint16(im_shape[2]), np.uint16(im_shape[3]), filename)
        for p, t in reversed(pt_list):
            encoder.call(p, t)
        encoder.finish()
        del encoder

    def fixed_decode(self, loop=3, filename='test.bin'):
        xp = self.xp
        decoder = RC.Decoder(filename)
        im_h = decoder.height
        im_w = decoder.width
        self.im_h = im_h
        self.im_w = im_w

        h_h = math.ceil(im_h / (2 ** (loop + self.lossy_shrink)))
        h_w = math.ceil(im_w / (2 ** (loop + self.lossy_shrink)))

        decoder = RC.Decoder(filename)
        p_init = xp.tile(F.softmax(self.d_pred, axis=1).data, (h_h * h_w, 1))
        t_init = xp.array(decoder.call(cuda.to_cpu(p_init)), dtype=np.float32)
        h = t_init.reshape((1, h_h, h_w, self.ct4)).transpose((0, 3, 1, 2))
        # print('df:\n{}'.format(h[0, 0, :5, :5]))

        for i in range(loop):
            h_h = math.ceil(im_h / (2 ** (loop + self.lossy_shrink - i - 1)))
            h_w = math.ceil(im_w / (2 ** (loop + self.lossy_shrink - i - 1)))
            h = self.single_scale_decode([self.b32_0[loop-i-1], self.b32_1[loop-i-1]], decoder, h, out_shape=(1, self.ct4, h_h, h_w))
            # print('d32_{}:\n{}'.format(loop-i-1, h[0, 0, :5, :5]))

        if self.ct3 != 0:
            h_h = math.ceil(im_h / (2 ** (self.lossy_shrink - 1)))
            h_w = math.ceil(im_w / (2 ** (self.lossy_shrink - 1)))
            t = self.single_scale_decode([self.b16_0, self.b16_1], decoder, h, out_shape=(1, self.ct3, h_h, h_w))
            ts = xp.pad(t, ((0, 0), (0, 0), (0, t.shape[2] % 2), (0, t.shape[3] % 2)), mode='constant')
            # print('d16:\n{}'.format(ts[0, 0, :5, :5]))
            if self.ct2 == 0:
                h = xp.concatenate((F.space2depth(ts, 2).data, h[:, self.ct3:]), axis=1)
            else:
                h = xp.concatenate((ts, M.unpooling_2d(h[:, self.ct3:], 2).data), axis=1)

        if self.ct2 != 0:
            h_h = math.ceil(im_h / (2 ** (self.lossy_shrink - 2)))
            h_w = math.ceil(im_w / (2 ** (self.lossy_shrink - 2)))
            t = self.single_scale_decode([self.b8_0, self.b8_1], decoder, h, out_shape=(1, self.ct2, h_h, h_w))
            ts = xp.pad(t, ((0, 0), (0, 0), (0, t.shape[2] % 2), (0, t.shape[3] % 2)), mode='constant')
            # print('d8:\n{}'.format(ts[0, 0, :5, :5]))
            if self.ct1 == 0:
                h = xp.concatenate((F.space2depth(ts, 4).data,
                                    F.space2depth(h[:, self.ct2:self.ct3], 2).data,
                                    h[:, self.ct3:, ::2, ::2]), axis=1)
            else:
                h = xp.concatenate((ts, M.unpooling_2d(h[:, self.ct2:], 2).data), axis=1)

        if self.ct1 != 0:
            h_h = math.ceil(im_h / (2 ** (self.lossy_shrink - 3)))
            h_w = math.ceil(im_w / (2 ** (self.lossy_shrink - 3)))
            t = self.single_scale_decode([self.b4_0, self.b4_1], decoder, h, out_shape=(1, self.ct1, h_h, h_w))
            ts = xp.pad(t, ((0, 0), (0, 0), (0, t.shape[2] % 2), (0, t.shape[3] % 2)), mode='constant')
            # print('d4:\n{}'.format(ts[0, 0, :5, :5]))
            h = xp.concatenate((F.space2depth(ts, 8).data,
                                F.space2depth(h[:, self.ct1:self.ct2], 4).data,
                                F.space2depth(h[:, self.ct2:self.ct3, ::2, ::2], 2).data,
                                h[:, self.ct3:, ::4, ::4]), axis=1)

        # print('init:\n{}'.format(h[0, 0, :5, :5]))
        decoder.finish()
        del decoder
        return h


class Block(chainer.Chain):
    def __init__(self, in_channels, mid_channels, out_channels, n_class):
        super(Block, self).__init__()
        self.n_class = n_class
        with self.init_scope():
            self.conv0 = L.Convolution2D(in_channels, mid_channels, ksize=3, pad=1)
            self.conv1 = L.Convolution2D(mid_channels, mid_channels, ksize=3, pad=1)
            self.conv2 = L.Convolution2D(mid_channels, mid_channels, ksize=3, pad=1)
            self.conv3 = L.Convolution2D(mid_channels, out_channels * self.n_class, ksize=3, pad=1)
            self.bn0 = L.BatchNormalization(mid_channels)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.bn2 = L.BatchNormalization(mid_channels)

    def __call__(self, x):
        h = x / (self.n_class - 1)
        h = self.conv0(h)
        h = self.bn0(h)
        h = F.relu(h)
        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)
        y = self.conv3(h)
        return y


def _crop(x, im_shape, shrink):
    if -(im_shape[2]) % (2 ** (shrink)) >= (2 ** (shrink - 1)):
        x = x[:, :, :-1]
    if -(im_shape[3]) % (2 ** (shrink)) >= (2 ** (shrink - 1)):
        x = x[:, :, :, :-1]
    return x
