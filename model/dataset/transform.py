#coding=utf-8
'''
实现不同的预处理操作 比如crop，resize,augment
这里都要实现成类 ，最后组成transform对象，作为参数传入Dataset对象。
'''

from __future__ import print_function, division
import torch
import numpy as np
import skimage
import skimage.transform


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
        largest_side = max(rows, cols) # 长边
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        # 上面求scale的操作保证了bbox的短边不会小于608，长边不会大于1024

        # resize img
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, channels = image.shape

        # 并padding到32的倍数
        pw = (32 - rows % 32)%32
        ph = (32 - cols % 32)%32
        # 在右下角padding
        new_img = np.zeros((rows+pw, cols+ph, channels)).astype(np.float32)
        new_img[:rows, :cols, :] = image.astype(np.float32)

        # label也要放大
        annots[:, 4] *= scale
        return {"img":torch.from_numpy(new_img), "annot":torch.from_numpy(annots), 'scale':scale}

class Augmenter(object):
    # 图像翻转
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample
class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        # 这里不用在转成tensor么？
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


if __name__ == '__main__':
    img = np.random.randint(1, 255, size=(300, 556, 3))
    img = img.astype(np.float32)
    print(img.shape)
    anno = np.random.randint(1,255,size=(1,5))
    anno = anno.astype(np.float32)
    dic = {'img': img, 'annot': anno}
    t = Resizer()
    out = t(dic) # 调用 __call__
    print(out) #
    print(out['img'].shape) # torch.Size([576, 1024, 3])

    t2 = Augmenter()
    out = t2(dic)
    print(out['img'].shape)

    t3 = Normalizer()
    out = t3(dic)
    print(out['img'].shape)

    x = torch.randn([200,200,3])
    t4 = UnNormalizer()
    out = t4(x)
    print(out.shape)

