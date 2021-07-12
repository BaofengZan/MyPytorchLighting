import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

BatchNorm2d = nn.BatchNorm2d

'''
https://github.com/BaofengZan/DBNet.pytorch/blob/master/models/backbone/resnet.py

thon模块中的__all__，用于模块导入时限制，如：from module import *

此时被导入模块若定义了__all__属性，则只有__all__内指定的属性、方法、类可被导入；若没定义，则导入模块内的所有公有属性，方法和类
'''
__all__ = ['resnet50']

# 预训练模型
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

}
# 首先实现基本的block 1*1 @64  3*3@64 1*1@256
# 一个残差分支的一半
# K S P
# 3 1 1 不变
# 3 2 1 减半
# 1 1 0 不边
class Bottleneck(nn.Module):
    #  expansion是指经过这么一个block通道数变为原来的n倍
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        '''
        输入通道 中间层通道
        '''
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) # k=1 s=1 p=0 降维度
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 减半或者不变
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False) # 升维度
        self.bn3 = BatchNorm2d(planes*4)  # 注意这里的通道数
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # downsample为外面传入的残差的另一半
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            # 如果没有downsample，就没有右侧下采样分支，直接残差相加
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3):
        '''
        layes = [3 4 6 3] 为各stage循环的次数
        '''
        self.inplanes = 64 # layer1 stage1的输入通道 以及每个不含donsample的block的输入通道 后面会变化。
        self.out_channels = []  # 输出通道数 64 128 256 512 1024 2048
        super(ResNet, self).__init__()
        # layer1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) # 7x7@64 s=2 p=3
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # layer2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 开始循环block
        self.layer1 = self._make_layer(block, 64, layers[0]) # 输入64 输出通道就是256
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 输入为256 ， 中间输出为128， 输出为128*4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        # retinaNet只需要x3-x5
        return x2, x3, x4, x5

    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        这里的planes是每个block的中间层通道数，因为输出为该值的4倍
        blocks 为输入
        '''
        downsample = None
        # 只有在 需要下采样时，才需要downsample
        # 即s=2或输出的通道数要是输入的4倍(粗暴点 就是输入为64时，不需要下采样)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample =  nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1,stride=stride, bias=False),
                BatchNorm2d(planes*block.expansion)
                                        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  # 首先第一个resblok 一般都有downsample
        self.inplanes = planes*block.expansion # 为当前stage的输入*4
        for i in range(1, blocks):
            # 后面这都没有downsample
            layers.append(block(self.inplanes, planes)) #
        self.out_channels.append(planes*block.expansion) # 讲每一stage的输出通道数记录下来
        return nn.Sequential(*layers)

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


if __name__ == "__main__":
    # test
    # 产生一个512*512的tensor
    x = torch.randn([4, 3, 512, 512])
    model = resnet50(pretrained=True)  # 可以运行
    _, C3, C4, C5 = model(x)
    print(C3.shape)
    print(C4.shape)
    print(C5.shape)
    print(model.out_channels)

'''
torch.Size([4, 256, 128, 128])
torch.Size([4, 512, 64, 64])
torch.Size([4, 1024, 32, 32])
[256, 512, 1024, 2048]
'''