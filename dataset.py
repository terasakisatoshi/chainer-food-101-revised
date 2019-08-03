import random

import chainer
import chainercv
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation


def transform_image(image):
    fill = random.randint(0, 255)

    # color augmentation
    do_distort = random.choice([True, False])
    if do_distort:
        image = random_distort(image)
    # random rotate
    angle = random.randint(-90, 90)
    image = chainercv.transforms.rotate(
        image,
        angle=angle,
        expand=True,
        fill=(fill, fill, fill),
    )

    # random flip
    image = chainercv.transforms.random_flip(image, x_random=True)

    # random_expand
    do_expand = random.choice([True, False])
    if do_expand:
        image = chainercv.transforms.random_expand(
            image,
            max_ratio=1.5,
            fill=fill,
        )
    # random scale and random crop
    do_crop = random.choice([True, False])
    if do_crop:
        image = chainercv.transforms.random_sized_crop(
            image,
            scale_ratio_range=(0.3, 0.8),
            aspect_ratio_range=(8 / 10, 10 / 8),
        )
    return image


class FoodDataset(chainer.dataset.DatasetMixin):
    def __init__(self, base):
        self.do_augmentation = (base.mode == "train")
        self.base = base
        self.imsize = base.imsize

    def get_example(self, i):
        orig_image, label = self.base.get_example(i)
        # copy image object to prevent changing value 
        # via side effect on data augmentation.
        image = orig_image.copy()
        if self.do_augmentation:
            image = transform_image(image)
            image = resize_with_random_interpolation(image, size=self.imsize)
        image = chainercv.transforms.resize(image, size=self.imsize)
        return image, label

    def __len__(self):
        return len(self.base)


def main():
    import os
    import random
    from chainercv.visualizations import vis_image
    from matplotlib import pyplot as plt
    from food_101_dataset import get_food101_dataset
    
    dataset_dir = os.path.expanduser("~/dataset/food-101")
    food_dataset = get_food101_dataset(dataset_dir, mode="train")
    i = random.randint(0, len(food_dataset))
    orig_image, _ = food_dataset.base.get_example(i)
    image, label = food_dataset.get_example(i)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    vis_image(orig_image, ax=ax1)
    vis_image(image, ax=ax2)
    ax1.set_title("original")
    ax2.set_title("transformed")
    plt.show()


if __name__ == '__main__':
    main()
