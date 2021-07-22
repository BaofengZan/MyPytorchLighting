#coding=utf-8
'''
实现不同的预处理操作 比如crop，resize,augment
这里都要实现成类 ，最后组成transform对象，作为参数传入Dataset对象。
'''

from __future__ import print_function, division
import torch
import numpy as np
import skimage


class Resizer(object):
    '''所有python的类。都是继承object的
    __call___: 使得类对象具有类似函数的功能。
    class A():
        def __call__(self, param):
            print('i can called like a function')
            print('掺入参数的类型是：', type(param))
    a = A()
    a('i') # 可以像函数一样调用
    '''
    # 通过放缩进行resize，这里最后的scale是放缩的倍数，相应的anno也需要放缩
    def __call__(self, sample, min_side=608, max_side=1024):
        # sample为dict 参看 coco.py中__getitem__函数
        image, annots = sample['img'], sample['annot']
        rows, cols, channels = image.shape # hwc
        smallest_side = min(rows, cols)  # 最短边
        scale = min_side / smallest_side
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side



