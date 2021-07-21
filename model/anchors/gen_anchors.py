#coding=utf-8
'''
用来生成anchors
'''
import torch
import torch.nn as nn
import  numpy as np

def generate_anchors(base_size=16, ratios=None, scales=None):
    '''
    给出base size ratio和scale 计算以原点为中心的所有anchors
    '''
    # retinanet特有的anchor
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # 初始化anchors
    anchors = np.zeros((num_anchors, 4))
    # 先计算 三种scale的anchor对应的宽(W) 高H 此时还不涉及ratios， 最后生成；basesize乘上不同scale的正方形
    anchors[:, 2:] = base_size*np.tile(scales, (2, len(ratios))).T # 最后是9*2的矩阵

    # 接下来将做ratio变换：**ratio变换的实质：将以上生成的正方形anchor按照一定比例缩放，
    # 求得缩放后的宽，保持scale不变，求得高。** 保持面积不变，生成不同的比例
    areas = anchors[:, 2] * anchors[:, 3] # 9*1    S1 S2 S3 S1 S2 S3 S1 S2 S3
    # np.repeat(ratios, len(scales)) == 0.5 0.5 0.5 1 1 1 2 2 2
    anchors[:, 2] = np.sqrt(areas/np.repeat(ratios, len(scales)))





class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        # 预设默认值
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]  # 下采样的倍数
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels] # 8 16 32 64 128
        else:
            self.strides = strides

        # 计算retinanet的anchor
        # basesize
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]  # 32 64 128 256 512
        else:
            self.sizes = sizes

        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        else:
            self.ratios = ratios

        if scales is None:
            self.scales = np.array([2, 2**(1.0/3.0), 2**(2.0/3.0)])
        else:
            self.scales = scales

    def forward(self, image):
        # 输入图像，计算所有的anchors
        image_shape = image.shape[2:] # nchw
        image_shape = np.array(image_shape)

        # 首先计算每个层的fature map大小
        # 向上取整
        image_shape = [np.ceil(image_shape / x) for x in self.pyramid_levels]

        #维度为0，说明可以无限制的扩充
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            # 3 4 5 6 7
            # 获取到每一层对应的anchor
            anchors = generate_anchors(base_size=self.sizes[idx],  ratios=self.ratios, scales=self.scales)
            # 映射到feature map
            shifted_anchors = shift(image_shape[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)  # 扩充
        all_anchors = np.expand_dims(all_anchors, axis=0) # 增加一个维度

        return torch.from_numpy(all_anchors.astype(np.float32)) # 返回的是Tensor



