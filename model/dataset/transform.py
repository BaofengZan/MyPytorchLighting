#coding=utf-8
'''
实现不同的预处理操作 比如crop，resize,augment
这里都要实现成类 ，最后组成transform对象，作为参数传入Dataset对象。
'''

from __future__ import print_function, division
import torch
import cv2
import numpy as np
import random
import skimage
import skimage.transform

#在目标检测任务中，由于数据增强后图片上目标的位置可能发生变化，因此我们必须自己定义数据增强函数同时处理图片和目标的坐标

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
    def __init__(self,  min_side=608, max_side=1024):
        self.min_side = min_side
        self.max_side = max_side
    # 通过放缩进行resize，这里最后的scale是放缩的倍数，相应的anno也需要放缩
    def __call__(self, sample):
        # sample为dict 参看 coco.py中__getitem__函数
        image, annots = sample['img'], sample['annot']
        rows, cols, channels = image.shape # hwc
        smallest_side = min(rows, cols)  # 最短边
        scale = self.min_side / smallest_side
        largest_side = max(rows, cols) # 长边
        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side
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
        annots[:, :4] *= scale
        return {"img":torch.from_numpy(new_img), "annot":torch.from_numpy(annots), 'scale':scale}

# 图像翻转
class Augmenter(object):
    def __init__(self,  flip_prob=0.5):
        self.flip_prob = flip_prob
    def __call__(self, sample):
        if np.random.uniform(0, 1) < self.flip_prob:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x1

            sample = {'img': image, 'annot': annots, 'scale': sample['scale']}

        return sample

class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        # 这里不用在转成tensor么？
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots, 'scale': sample['scale']}
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

# 随机裁剪
class RandomCrop(object):
    def __init__(self, crop_prob=0.5):
        self.crop_prob = crop_prob
    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']
        if annots.shape[0] == 0:
            return sample
        if np.random.uniform(0, 1) < self.crop_prob:
            h, w, _ = image.shape # hwc  x1 y1 x2 y2
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ], axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_left_trans)))
            crop_ymin = max(0,
                            int(max_bbox[1] - random.uniform(0, max_up_trans)))
            crop_xmax = min(
                w, int(max_bbox[2] + random.uniform(0, max_right_trans)))
            crop_ymax = min(
                h, int(max_bbox[3] + random.uniform(0, max_down_trans)))

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            annots[:, [0, 2]] = annots[:, [0, 2]] - crop_xmin  # x1 x2
            annots[:, [1, 3]] = annots[:, [1, 3]] - crop_ymin

            sample = {'img': image, 'annot': annots, 'scale': scale}
            return sample

class RandomTranslate(object):
    def __init__(self, translate_prob=0.5):
        self.translate_prob = translate_prob

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.translate_prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ],
                                      axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            tx = random.uniform(-(max_left_trans - 1), (max_right_trans - 1))
            ty = random.uniform(-(max_up_trans - 1), (max_down_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])

            # src - 输入图像。
            # M - 变换矩阵。
            # dsize - 输出图像的大小。
            image = cv2.warpAffine(image, M, (w, h))
            annots[:, [0, 2]] = annots[:, [0, 2]] + tx
            annots[:, [1, 3]] = annots[:, [1, 3]] + ty

            sample = {'img': image, 'annot': annots, 'scale': scale}

        return sample


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

