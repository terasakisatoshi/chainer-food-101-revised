import os
import json

import numpy as np
from chainer.datasets import LabeledImageDataset




# these images is not suited for our purpose
BLACKLIST = [
    "lasagna/3787908",
    "steak/1340977",
    "bread_pudding/1375816"
]

IMG_EXT = ".jpg"


def load_annotations(dataset_dir, mode):
    meta_dir = os.path.join(dataset_dir, "meta")
    class_file = os.path.join(meta_dir, "classes.txt")
    food_names = np.genfromtxt(
        class_file,
        str,
        delimiter="\n"
    )
    idx2name = {food_id: food_name for food_id, food_name in enumerate(food_names)}
    json_name = "train.json" if mode == "train" else "test.json"
    json_file = os.path.join(meta_dir, json_name)
    with open(json_file, 'r') as f:
        food2path = json.load(f)
    pairs = []
    for food_id, food_name in idx2name.items():
        for path in food2path[food_name]:
            if path in BLACKLIST:
                # ignore
                continue
            path = os.path.join(dataset_dir, "images", path+IMG_EXT)
            pairs.append((path, food_id))
    
    name2idx={name:idx for idx,name in idx2name.items()}
    annotations={
        "pairs":pairs,
        "name2idx":name2idx,
        "idx2name":idx2name,
    }
    return annotations


class Food101Dataset(LabeledImageDataset):
    def __init__(self, dataset_dir, mode="train"):
        annotations = load_annotations(dataset_dir, mode)
        self.mode = mode
        super(Food101Dataset, self).__init__(annotations["pairs"])
        self.idx2name = annotations["idx2name"]
        self.name2idx = annotations["name2idx"]
        
def get_food101_dataset(dataset_dir,mode="train"):
    return Food101Dataset(dataset_dir,mode=mode)

if __name__=="__main__":
    import random

    from chainercv.visualizations import vis_image
    from matplotlib import pyplot as plt
    # specify dataset directory
    dataset_dir = os.path.expanduser("~/dataset/food-101")
    food101_dataset = get_food101_dataset(dataset_dir, mode="train")
    sample = np.random.randint(0, len(food101_dataset), size=100)
    i=random.randint(0,len(food101_dataset))
    example = food101_dataset.get_example(i)

    img, food_idx = example
    name = food101_dataset.idx2name[int(food_idx)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(name)
    vis_image(img, ax)
    plt.show()


