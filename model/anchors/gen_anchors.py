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
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    # 以当前anchor的中心点为坐标原点建立直角坐标系，求出左上角坐标和右下角坐标，存入当前数组，格式为(x1,y1,x2,y2)。
    #np.tile(a, (m, n)) 将a拼成m行n列的数组
    # 下面一句就是 将上面得到的同面积不同尺度的W H ，以（0， 0） 为原点组成bbox
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T  # 第0列和第2列  x1 x2 相当于 x1=0-w/2 x2 = w-w/2
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T # 最后是 x1 y1 x2 y2
    return anchors  # 9*4

# 将上面计算到的anchor映射到每个feature map上得到真实坐标
def shift(shape, stride, anchors):
    '''
    shape为当前feature map大小
    stride为相较于原图的stride，用于将anchor 映射到原图的坐标
    '''
    # 首先将该fetaure map划分网格，并+0.5 得到每个网格的中心坐标 并映射到原图
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    # np.meshgrid(x, y)得到网格矩阵
    # x为 m y为n
    # shift_x 就为 n*m
    # shift_y 也为 n*m
    '''
    >>> shift_x = (np.arange(0, 3) + 0.5)
    >>> shift_y = (np.arange(0, 2) + 0.5)
    >>> shift_x
    array([0.5, 1.5, 2.5])
    >>> shift_y
    array([0.5, 1.5])
    >>> shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    >>> shift_x
    array([[0.5, 1.5, 2.5],
           [0.5, 1.5, 2.5]])
    >>> shift_y
    array([[0.5, 0.5, 0.5],
           [1.5, 1.5, 1.5]])
    最后形成的坐标是 (0.5, 0.5) (1.5, 0.5) （2.5, 0.5）...
    两个矩阵对应位置组成的（x，y）
    >>>

    '''
    shift_x, shift_y = np.meshgrid(shift_x, shift_y) # 所有网格的坐标

    # 再把shift——x拉成一行
    shift_x = shift_x.ravel() # np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])
    # stack
    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)
    # 对于每一个feature map 假设为 8*8  这里就形成4*64的矩阵 矩阵中的每一列为cx cy cx cy，为该feature map上的网格中心点 映射到原图上的位置。

    shifts = np.transpose(shifts) # 再转置 64*4

    '''
    我们得到的anchor坐标实际上可以看做是以（0， 0）为坐标原点得到的，
    那么坐标值实际上可以看做是左上角和右下角两个点对中心点的偏移量。那么如果我们将中心
    点换做p3的feature map的网格点，然后将偏移量叠加至上面，不就完成了anchor到feature map的映射嘛 。
    最后返回(x1 y1 x2 y2)
    '''
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0] # 9
    K = shifts.shape[0] # 8*8 = 64
    # (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)) == (1, k*A, 4)
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors

'''
所谓Anchor，就是一组不同长宽比、不同大小的先验框
需要注意的是每个网格上的9个先验框长宽都是一样的，只是框的中心点不同。 
RetinaNet使用了三种长宽比和三种放大比例先生成了9种长宽组合：
ratios = [0.5, 1, 2]
scales = [2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)]
aspects = [[[s * math.sqrt(r), s * math.sqrt(1 / r)] for s in scales]
           for r in ratios]
[[[0.7071067811865476, 1.4142135623730951], 
[0.8908987181403394, 1.7817974362806788], 
[1.122462048309373, 2.244924096618746]], 

[[1.0, 1.0],
 [1.2599210498948732, 1.2599210498948732], 
 [1.5874010519681994, 1.5874010519681994]], 
 
 [[1.4142135623730951, 0.7071067811865476],
 [1.7817974362806788, 0.8908987181403394], 
 [2.244924096618746, 1.122462048309373]]]
'''

class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        # 预设默认值
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]  # 下采样的倍数
            #self.pyramid_levels = [1, 2]  # 下采样的倍数
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels] # 8 16 32 64 128
        else:
            self.strides = strides

        # 计算retinanet的anchor
        # basesize
        if sizes is None:
            # areas
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]  # 32 64 128 256 512
        else:
            self.sizes = sizes

        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        else:
            self.ratios = ratios

        if scales is None:
            self.scales = np.array([2**0, 2**(1.0/3.0), 2**(2.0/3.0)])
        else:
            self.scales = scales

    def forward(self, image):
        # 输入图像，计算所有的anchors
        image_shape = image.shape[2:] # nchw
        image_shape = np.array(image_shape)

        # 首先计算每个层的fature map大小
        # 向上取整
        #image_shape = [np.ceil(image_shape / 2 ** x) for x in self.pyramid_levels]
        image_shape = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
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

        # # 输出的是anchor xyxy
        # return torch.from_numpy(all_anchors.astype(np.float32)) # 返回的是Tensor
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))


if __name__ == '__main__':
    # 改变c不变输出，说明通道没有影响
    C = torch.randn([6,1,16,16])
    model = Anchors()
    out = model(C)
    print(out.shape)
    print(out)