import argparse
from glob import glob
import json
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

import os
import random

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

from tqdm import tqdm
from network_resnet import ResNet50
from food101_dataset import get_food101_dataset


def find_latest(model_dir):
    files = glob(os.path.join(model_dir, "model_epoch_*.npz"))
    numbers = []
    for f in files:
        base = os.path.basename(f)
        base = base[len("model_epoch_"):]
        base = base[:-len(".npz")]
        e = int(base)
        numbers.append(e)
    e = max(numbers)
    latest = os.path.join(model_dir, "model_epoch_{}.npz".format(e))
    logging.info("latest snapshot is {}".format(latest))
    return latest


def prepare_setting(args):
    latest_snapshot = find_latest(args.trained)

    logger.info("> restore snapshot from {}".format(latest_snapshot))
    model = L.Classifier(ResNet50(num_classes=101))
    chainer.serializers.load_npz(latest_snapshot, model)
    logger.info("> load test set")
    test_dataset = get_food101_dataset(args.dataset, mode="test")

    if not chainer.backends.cuda.available:
        xp = np
    elif args.device >= 0:
        # use GPU
        chainer.backends.cuda.get_device_from_id(args.device).use()
        model.predictor.to_gpu()
        import cupy as xp
    else:
        # use CPU
        xp = np

    return model, xp, test_dataset


def predict(args):
    classes = np.genfromtxt(
        os.path.join(args.dataset, "meta", "classes.txt"),
        str,
        delimiter="\n"
    )
    model, xp, test_dataset = prepare_setting(args)

    top_1_counter = 0
    top_5_counter = 0
    top_10_counter = 0
    indices = list(range(len(test_dataset)))
    num_iteration = len(indices) if args.sample < 0 else args.sample
    random.shuffle(indices)
    with chainer.using_config('train', False),
         chainer.function.no_backprop_mode():
        for i in tqdm(indices[:num_iteration]):
            img, label = test_dataset.get_example(i)
            h = model.predictor(xp.expand_dims(xp.array(img), axis=0))
            prob = F.softmax(h).array.squeeze()
            prob = chainer.backends.cuda.to_cpu(prob)
            top_ten = np.argsort(-prob)[:10]
            top_five = top_ten[:5]
            if top_five[0] == label:
                top_1_counter += 1
                top_5_counter += 1
                top_10_counter += 1
                msg = "Bingo!"
            elif label in top_five:
                top_5_counter += 1
                top_10_counter += 1
                msg = "matched top 5"
            elif label in top_ten:
                top_10_counter += 1
                msg = "matched top 10"
            else:
                msg = "Boo, actual {}".format(classes[label])
            percent = list(map(int, (100 * prob[top_five])))
            logger.debug("> {} {} {}".format(classes[top_five], percent, msg))
        print('top1 accuracy', top_1_counter / num_iteration)
        print('top5 accuracy', top_5_counter / num_iteration)
        print('top10 accuracy', top_10_counter / num_iteration)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trained", type=str, help="path/to/snapshot e.g. trained")
    parser.add_argument("--sample", type=int, default=-1,
                        help="select num of --sample from test dataset to evaluate accuracy")
    parser.add_argument("--device", type=int, default=0,
                        help="specify GPU_ID. If negative, use CPU")
    parser.add_argument("--dataset", type=str, default=os.path.expanduser("~/dataset/food-101"))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_argument()
    predict(args)
