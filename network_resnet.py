import chainer
import chainer.links as L
import chainer.functions as F
from chainercv.links.model import resnet
from chainer.backends.cuda import get_array_module


class ResNet50(chainer.Chain):
    def __init__(self, num_classes, **kwargs):
        super(ResNet50, self).__init__()
        with self.init_scope():
            # use ResNet provided by ChainerCV project as base net
            self.base = resnet.ResNet50(pretrained_model="imagenet")
            self.base.pick = 'pool5'
            self.mean = self.base.mean
            self.fc_1 = L.Linear(None, 1024)
            self.fc_2 = L.Linear(1024, num_classes)

    def __call__(self, x):
        h = self.prepare(x)
        h = self.base(h)
        h = F.dropout(F.relu(self.fc_1(h)))
        h = self.fc_2(h)
        return h

    def disable_target_layers(self):
        disables = ['conv1',
                    'res2',
                    'res3',
                    'res4',
                    # 'res5',
                    ]

        for layer in disables:
            self.base[layer].disable_update()

    def prepare(self, x):
        xp = get_array_module(x)
        """
        Args:
            x (~numpy.ndarray): An image. This is in RGB color and in CHW format.
                The range of its value is :math:`[0, 255]`.
        # note that the specification of the value mean equipped with base model is as follow:
        # RGB order
        # This is channel wise mean of mean image distributed at
        # https://github.com/KaimingHe/deep-residual-networks
        _imagenet_mean = np.array(
            [123.15163084, 115.90288257, 103.0626238],
            dtype=np.float32)[:, np.newaxis, np.newaxis]
        """
        return x - xp.asarray(self.mean)


def main():
    """runtime test"""
    import numpy as np
    resnet = ResNet50(num_classes=101)
    img = np.ones((2, 3, 224, 224), dtype=np.float32)
    resnet.disable_target_layers()
    ret = resnet(img)


if __name__ == '__main__':
    main()
