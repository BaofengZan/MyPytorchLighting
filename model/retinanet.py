from model.backbone.resnet import *
from model.neck.fpn import *
from model.head.retinanet_head import *
from model.anchors.gen_anchors import *
from model.utils.bboxtransform import *
from model.utils.clipbox import *
import torch
import torch.nn as nn
import math

class RetinaNet(nn.Module):
    def __init__(self, number_classes):
        super(RetinaNet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        # resnet输出得 c2 c3 c4 c5 [256 512 1024 2048]
        self.fpn = FPN(512, 1024, 2048)
        self.regression = RegressionModule(256)
        self.classification = ClassificationModule(256, num_classes=number_classes)

        # 初始化权重
        prior = 0.01
        self.classification.output.weight.data.fill_(0)
        self.classification.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regression.output.weight.data.fill_(0)
        self.regression.output.bias.data.fill_(0)

        # 产生anchors
        self.anchors = Anchors() # 创建对象
        # bbox转化
        self.boxTrans = BBoxTransform()
        self.clip = ClipBoxes()
    def forward(self, x):
        # 再我们构建网络时，也要生成该网络的anchors。
        # 生成的位置 肯定是在计算loss之前。（用anchor和网络的预测输出计算loss）
        # 这里的相当于 yolov5中的build_targets
        ## 预设值的anchors数和regressionmodel出来的anchors数是一样的！！！！！！！
        # 不过这个出来的shape的batchsize总是为1
        #这里输入的x应该为 nchw
        anchors = self.anchors(x)
        ##########
        _, C3, C4, C5 = self.resnet(x)
        # [P3_x, p4_x, p5_x, P6_x, P7_x]
        fpn_out_5_layer_list = self.fpn([_, C3, C4, C5])
        # 沿着列组合
        # fpn得结果经过回归分支，最后cat
        regression = torch.cat([self.regression(feature) for feature in fpn_out_5_layer_list], dim=1)
        classification = torch.cat([self.classification(feature) for feature in fpn_out_5_layer_list], dim=1)

        # 这里regression预测是(dx,dy,dw,dh)，要根据预设的anchors 计算出预测的bbox。
        transform_anchors = self.boxTrans(anchors, regression)
        # 这里预测的anchors 有可能越界， 边界的bbox 需要clip
        transform_anchors = self.clip(transform_anchors, x)
        return regression, classification, transform_anchors
if __name__ == '__main__':
    C = torch.randn([2, 3, 512, 512])
    model = RetinaNet(80)
    out = model(C)
    for i in range(len(out)):
        print(out[i].shape)
# torch.Size([2, 49104, 4])
# torch.Size([2, 49104, 80])

