#coding=utf-8

from __future__ import print_function, division
import random
from torch.utils.data.sampler import Sampler   # 自定义的sample继承该基类

#sampler是在给dataloader使用的时候需要给每一个batch进行采样方式的调整

# 可以看到这里使用了随机打乱，按照image_aspect_ratio进行排序。

class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source  # data_source是自定义的cocoDataset对象
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group  #这里使用的yield

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]
