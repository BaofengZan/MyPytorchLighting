#coding=utf-8
'''
实现自定义的collater函数
使用这个参数可以自己操作每个batch的数据 。为一个函数，输入为一批数据，输出处理后的数据。
'''
from __future__ import print_function, division
import torch
import numpy as np


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    heights = [int(s.shape[0]) for s in imgs]
    widths = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_height, max_width, 3) # 这里应该hwc的矩阵。

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    # hwc转chw
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    # 要转为tensor么？ 不用，因为该函数在DataLoader内部使用， DataLoader内部会自动转为tensort的
    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}