import argparse
import os
import time

import chainer
import chainercv
import chainer.functions as F
import cv2
import numpy as np

from predict import prepare_setting, restore_args
from food101_dataset import get_food101_dataset


def video(args):
    args_trained = restore_args(args.trained)
    inH, inW = args_trained["height"], args_trained["width"]
    dataset = get_food101_dataset(args.dataset, mode="test")
    idx2name = dataset.base.idx2name
    model, xp, _ = prepare_setting(args)

    cap = cv2.VideoCapture(1)
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    fps_time = 0
    with chainer.using_config('train', False), chainer.function.no_backprop_mode():
        while cap.isOpened():
            ret_val, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)
            _, capH, capW = img.shape
            crop_size = min(capH, capW)
            img = chainercv.transforms.center_crop(img, (crop_size, crop_size))
            img = chainercv.transforms.resize(img, (inH, inW))
            vis_img = img.copy().transpose(1, 2, 0)
            # RGB->BGR
            vis_img = vis_img[:, :, ::-1]
            start = time.time()
            h = model.predictor(xp.expand_dims(xp.array(img), axis=0))
            prediction = F.softmax(h)
            if args.device >= 0:
                prediction = xp.asnumpy(prediction[0].data)
            else:
                prediction = prediction[0].data
            top_ten = np.argsort(-prediction)[:10]
            end = time.time()
            print("Elapsed", end - start)
            blank = np.zeros((inH, 2 * inW, 3)).astype(img.dtype)
            for rank, label_idx in enumerate(top_ten):
                score = prediction[label_idx]
                name = idx2name[label_idx]
                print('{:>3d} {:>6.2f}% {}'.format(
                    rank + 1, score * 100, name))
                cv2.putText(blank, '{:>3d} {:>6.2f}% {}'.format(
                    rank + 1, prediction[label_idx] * 100, idx2name[label_idx]),
                            (10, 20 * (rank + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(blank, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            title = "Food-101"
            cv2.imshow(title, cv2.hconcat([vis_img, blank]))
            fps_time = time.time()
            """Hit esc key"""
            if cv2.waitKey(1) == 27:
                break


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trained", type=str, help="path/to/trained")
    parser.add_argument("--dataset", type=str, help="path/to/food-101",
                        default=os.path.expanduser("~/dataset/food-101"))
    parser.add_argument("--device", type=int, default=-1,
                        help="specify GPU_ID. If negative, use CPU")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_argument()
    video(args)
