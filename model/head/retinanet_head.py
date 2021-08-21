'''
fpn的P3,P4,P5,P6,P7，每一个特征图都会进行5次的卷积操作，前面4次其实都是不改变tensor的shape的，最后一次要改变通道数的
回归输出的H*W*4A  A是anchor的个数
'''

import torch.nn as nn
import torch
from model.basic import ConvBnRelu, ConvRelu

class RegressionModule(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModule, self).__init__()

        # self.cba1 = ConvBnRelu(num_features_in, feature_size, kernel_size=3, stride=1, padding=1)
        # self.cba2 = ConvBnRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # self.cba3 = ConvBnRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # self.cba4 = ConvBnRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.cba1 = ConvRelu(num_features_in, feature_size, kernel_size=3, stride=1, padding=1)
        self.cba2 = ConvRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.cba3 = ConvRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.cba4 = ConvRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.output = nn.Conv2d(feature_size, 4*num_anchors, kernel_size=3, stride=1, padding=1)

        for M in self.modules():
            for m in M.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        out = self.cba1(x)
        out = self.cba2(out)
        out = self.cba3(out)
        out = self.cba4(out)

        out = self.output(out)

        # out -- n（4A）hw 要转为 n h*w*A 4
        # 1 n（4A）hw --》 nhw(4A)
        out = out.permute(0, 2, 3, 1)
        # 2 nhw(4A) --> n(h*w*A)4
        # # 通过contiguout().view变成 b , w*h, 4
        # https://zhuanlan.zhihu.com/p/64551412
        # is_contiguous直观的解释是Tensor底层一维数组元素的存储顺序与Tensor按行优先一维展开的元素顺序是否一致。
        # 为什么需要 contiguous ？  torch.view等方法操作需要连续的Tensor。

        # reshape方法会强制调用contiguous方法，在调用的时候，
        # 如果张量的形状和初始维度形状兼容（兼容的定义是新张量的两个连续维度的乘积等于原来张量的某一维度），
        # 则会返回原始张量。
        return out.reshape(out.shape[0], -1, 4)

# 分类分支
class ClassificationModule(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, feature_size=256):
        super(ClassificationModule, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # self.cba1 = ConvBnRelu(num_features_in, feature_size, kernel_size=3, stride=1, padding=1)
        # self.cba2 = ConvBnRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # self.cba3 = ConvBnRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # self.cba4 = ConvBnRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.cba1 = ConvRelu(num_features_in, feature_size, kernel_size=3, stride=1, padding=1)
        self.cba2 = ConvRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.cba3 = ConvRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.cba4 = ConvRelu(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # 这里和regression区别在于*num_classes
        # 每个anchor都有一个类别
        self.output = nn.Conv2d(feature_size, num_classes*num_anchors, kernel_size=3, stride=1, padding=1)
        # 还要进行一次sigmoid，为了不出现负数，将其映射到0-1之间
        self.output_act = nn.Sigmoid()

        for M in self.modules():
            for m in M.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        out = self.cba1(x)
        out = self.cba2(out)
        out = self.cba3(out)
        out = self.cba4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x H x W, with C = n_classes * n_anchors
        out1 = out.permute(0, 2, 3, 1)
        batch_size, height, width,  channels = out1.shape
        out1.view(batch_size, width, height, 9, self.num_classes)

        # 最后输出 B （h*w*A）num_classes
        return out1.reshape(x.shape[0], -1, self.num_classes)


if __name__ == '__main__':
    C = torch.randn([2,256,512,512])
    model = RegressionModule(256)
    out = model(C)
    print(out.shape)
    for i in range(len(out)):
        print(out[i].shape)
    # 说明用len可以得出第一维
    # torch.Size([2, 2359296, 4])
    # torch.Size([2359296, 4])
    # torch.Size([2359296, 4])
    print("------------------")
    C1 = torch.randn([2, 256, 64, 64])
    C2 = torch.randn([2, 256, 32, 32])
    C3 = torch.randn([2, 256, 16, 16])
    C4 = torch.randn([2, 256, 8, 8])
    C5 = torch.randn([2, 256, 4, 4])
    model = ClassificationModule(256)
    print(model(C1).shape)
    print(model(C2).shape)
    print(model(C3).shape)
    print(model(C4).shape)
    print(model(C5).shape)

    # torch.Size([2, 36864, 80])
    # torch.Size([2, 9216, 80])
    # torch.Size([2, 2304, 80])
    # torch.Size([2, 576, 80])
    # torch.Size([2, 144, 80])