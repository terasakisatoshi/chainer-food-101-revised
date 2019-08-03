import json
import os

import chainer


def save_args(args):
    if not os.path.exists(args.destination):
        os.mkdir(args.destination)

    for k, v in vars(args).items():
        print(k, v)

    with open(os.path.join(args.destination, "args.json"), 'w') as f:
        json.dump(vars(args), f)


def parse_device_list(device_list):
    device = {}
    for i, d in enumerate(device_list):
        if not chainer.backends.cuda.available:
            device = {"main": -1}
            break
        if i == 0:
            device["main"] = d
        else:
            name = "slave{}".format(i)
            device[name] = d
    return device
