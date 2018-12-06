import argparse
import chainer
from chainer.backends import cuda
from chainer import training
from chainer.training import extensions
import dataset as D
from model_lossy import LossyModel


def run_training(args):

    model = LossyModel(c_list=args.c_list, q_num=args.q_num)
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    log_interval = (100, 'iteration')
    eval_interval = (5000, 'iteration')

    train = D.ImageDataset(args.root, args.train_paths)
    test = D.ImageDataset(args.root, args.test_paths)

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    test_iter = chainer.iterators.MultiprocessIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)

    evaluator = extensions.Evaluator(test_iter, model, device=args.gpu)
    trainer.extend(evaluator, trigger=eval_interval, name='val')

    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.LogReport(trigger=log_interval, log_name='log_lossy'))
    trainer.extend(extensions.snapshot_object(model, 'model_lossy'), trigger=(args.iteration, 'iteration'))

    trainer.extend(extensions.PrintReport([
        'iteration',
        'main/MSSSIM',
        'val/main/MSSSIM',
        'elapsed_time',
    ]))

    trainer.extend(extensions.LinearShift('alpha', (0.001, 0), (int(args.iteration * 0.75), args.iteration)))

    print('==========================================')
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--out', '-o', type=str, default='results')
    parser.add_argument('--root', type=str)
    parser.add_argument('--train_paths', type=str)
    parser.add_argument('--test_paths', type=str)
    parser.add_argument('--c_list', type=int, nargs=4, default=(0, 0, 4, 32))
    parser.add_argument('--q_num', type=int, default=7)
    parser.add_argument('--batchsize', '-b', type=int, default=24)
    parser.add_argument('--iteration', '-i', type=int, default=100000)
    args = parser.parse_args()

    chainer.using_config('autotune', True)
    run_training(args)
