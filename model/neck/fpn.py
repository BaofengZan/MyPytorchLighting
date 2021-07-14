'''
retinaNet的FPN从resnet50的c3 c4 c5开始
分别引出一个分支为 p3 p4 p5
并且从c5在下采样处一个p6.后接一个卷积得到p7
'''

import torch
import torch.nn as nn


class FPN(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, featuresize=256):
        '''
        https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/neck/FPN.py
        Args:
            C3_size:  c3的输入size 256
            C4_size:  512
            C5_size:  1024
            featuresize:  fpn的输出feature。并且输出都一样
        '''
        super(FPN, self).__init__()
        # p5_1由c5得到
        self.p5_1 = nn.Conv2d(C5_size, featuresize, kernel_size=1, stride=1, padding=0) # conv后面需要接bn层么？ 可以验证下
        self.p5_upsample = nn.Upsample(scale_factor=2, mode="nearest") # 和P4相加之后再送入
        self.p5_2 = nn.Conv2d(featuresize, featuresize, kernel_size=3, stride=1, padding=1)  # p5_1经过这个卷积，就输出到head 不改变shape
        #  add p5
        self.p4_1 = nn.Conv2d(C4_size, featuresize, kernel_size=1, stride=1, padding=0)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_2 = nn.Conv2d(featuresize, featuresize, kernel_size=3, stride=1, padding=1)

        # p3 不需要再upsample
        self.p3_1 = nn.Conv2d(C3_size, featuresize, kernel_size=1, stride=1, padding=0)
        self.p3_2 = nn.Conv2d(featuresize, featuresize, kernel_size=3, stride=1, padding=1)

        self.p6 = nn.Conv2d(C5_size, featuresize, kernel_size=3, stride=2, padding=1) # feature减半
        self.p7_1 = nn.ReLU()
        self.p7_2 = nn.Conv2d(featuresize, featuresize, kernel_size=3, stride=2, padding=1) # 再次减半
        # 减半不能整除的是向上取整

    def forward(self, x):
        _, C3, C4, C5 = x
        p5_x = self.p5_1(C5)
        p5_upsample_x = self.p5_upsample(p5_x)
        p5_x = self.p5_2(p5_x)

        p4_x = self.p4_1(C4)
        p4_x = p5_upsample_x + p4_x
        p4_upsampled_x = self.p4_upsample(p4_x)
        p4_x = self.p4_2(p4_x)

        P3_x = self.p3_1(C3)
        P3_x = P3_x + p4_upsampled_x
        P3_x = self.p3_2(P3_x)

        P6_x = self.p6(C5)

        P7_x = self.p7_1(P6_x)
        P7_x = self.p7_2(P7_x)

        return [P3_x, p4_x, p5_x, P6_x, P7_x]



if __name__ == '__main__':
    C2 = torch.randn([2, 16, 200, 200])
    C3 = torch.randn([2, 16, 200, 200])
    C4 = torch.randn([2, 32, 100, 100])
    C5 = torch.randn([2, 64, 50, 50])

    model = FPN(16, 32, 64)

    out = model([C2, C3, C4, C5])
    for i in range(len(out)):
        print(out[i].shape)