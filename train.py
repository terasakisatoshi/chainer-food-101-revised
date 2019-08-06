import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

import random
import os

import cv2

# https://docs.chainer.org/en/stable/reference/generated/chainer.iterators.MultiprocessIterator.html#chainer-iterators-multiprocessiterator
cv2.setNumThreads(0)

import matplotlib

matplotlib.use('Agg')  # to prevent software hang

import chainer
import chainer.links as L
from chainer.datasets import split_dataset_random
from chainer.iterators import MultiprocessIterator, MultithreadIterator, SerialIterator
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import MinValueTrigger
import numpy as np

from food101_dataset import get_food101_dataset
from utils import save_args, parse_device_list


def select_model(model_name):
    if model_name == "resnet":
        from network_resnet import ResNet50
        model = ResNet50(num_classes=101)
        model.disable_target_layers()
    elif model_name == "mv2":
        from network_mobilenet import MobileNetV2
        model = MobileNetV2(101)
    else:
        NotImplementedError("This {} is not implemented".format(model_name))
    return L.Classifier(model)


def set_random_seed(args):
    logger.info("> set random seed")
    random.seed(args.seed)
    np.random.seed(args.seed)
    main_device = args.device[0]
    if chainer.backends.cuda.available and main_device >= 0:
        chainer.cuda.get_device_from_id(main_device).use()
        chainer.cuda.cupy.random.seed(args.seed)


def main(args):
    logger.info("> begin setup")
    chainer.config.cv_resize_backend = "cv2"
    chainer.global_config.autotune = True
    chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)
    chainer.config.cudnn_fast_batch_normalization = True

    logger.info("> show args info")
    save_args(args)
    dataset_dir = args.dataset
    imsize = (args.height, args.width)
    logger.info("> imsize {}".format(imsize))
    logger.info("> load dataset from {}".format(dataset_dir))
    train_set = get_food101_dataset(dataset_dir, mode="train", imsize=imsize)
    val_set = get_food101_dataset(dataset_dir, mode="val", imsize=imsize)

    logger.info("> training size {}".format(len(train_set)))
    logger.info("> validation size {}".format(len(val_set)))

    logger.info("> make iterator")
    train_iter = MultiprocessIterator(train_set, args.batch_size)
    val_iter = MultiprocessIterator(
        val_set, args.batch_size,
        repeat=False, shuffle=False
    )

    model_name = args.model
    logger.info("> setup mdoel {}".format(model_name))
    model = select_model(model_name)

    logger.info("> setup optimzier")
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)

    devices = parse_device_list(args.device)
    logger.info("> device list: {}".format(devices))

    updater = training.updaters.ParallelUpdater(
        train_iter, optimizer, devices=devices)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.destination)

    snapshot_interval = (1, 'epoch')

    logger.info("> setup trainer")
    trainer.extend(
        extensions.Evaluator(val_iter, model, device=devices["main"]),
        trigger=snapshot_interval
    )

    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.LogReport(
        trigger=snapshot_interval,
        log_name='log.json')
    )
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval
    )
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval
    )

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'validation/main/loss'],
                'epoch',
                file_name='loss.png',
            ),
            trigger=snapshot_interval,
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'
            ),
            trigger=snapshot_interval
        )

    if args.resume:
        logger.info("resume trainer object from {}".format(args.resume))
        chainer.serializers.load_npz(args.resume, trainer)
    logger.info("> start to train")
    trainer.run()
    logger.info("> end")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "mv2"], type=str, default="resnet")
    parser.add_argument("--seed", type=int, default=12345, help='seed for numpy cupy random module %(default)s')
    parser.add_argument("--dataset", type=str, default=os.path.expanduser("~/dataset/food-101"),
                        help='path/to/food-101 default = %(default)s')
    parser.add_argument("--destination", default="trained", help="path/to/save/directory %(default)s")
    parser.add_argument("--device", nargs='+', type=int, default=[0],
                        help="specify gpu id on training %(default)s -1 means use cpu")
    parser.add_argument("--height", type=int, default=224, help="input image height %(default)s")
    parser.add_argument("--width", type=int, default=224, help="input image width %(default)s")

    parser.add_argument("--batch_size", "-b", type=int, default=64, help="batch size per device %(default)s")
    parser.add_argument("--epoch", "-e", type=int, default=100, help="batch size per device %(default)s")
    parser.add_argument("--resume", type=str, default="", help="path/to/snapshot/of/trainer")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args)
    print(args.device)
    main(args)
