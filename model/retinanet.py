from model.backbone.resnet import *
from model.neck.fpn import *
from model.head.retinanet_head import *
from model.anchors.gen_anchors import *
from model.utils.bboxtransform import *
from model.utils.clipbox import *
from model.nms.nms import *
from torchvision.ops import nms as torch_nms
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
        # 这里的输入 x 因该区分train和test：train带有annotation test可以不用带
        anchors = self.anchors(x)
        ##########
        _, C3, C4, C5 = self.resnet(x)
        # [P3_x, p4_x, p5_x, P6_x, P7_x]
        fpn_out_5_layer_list = self.fpn([_, C3, C4, C5])
        # 沿着列组合
        # fpn得结果经过回归分支，最后cat
        regression = torch.cat([self.regression(feature) for feature in fpn_out_5_layer_list], dim=1)
        classification = torch.cat([self.classification(feature) for feature in fpn_out_5_layer_list], dim=1)

        #
        classification = torch.randn([2, 49104, 80]) # 测试使用
        #### 从这开始都是test使用的。如果是训练，直接应该计算loss了
        # test时，前向的结果，需要转换为prebox并nms得到最后的结果。
        # 这里regression预测是(dx,dy,dw,dh)，要根据预设的anchors 计算出预测的bbox。
        transform_anchors = self.boxTrans(anchors, regression)
        # 这里预测的anchors 有可能越界， 边界的bbox 需要clip
        transform_anchors = self.clip(transform_anchors, x)

        # 这里我们已经得到所有的预测框。后面就开始nms
        # 首先创建几个Tensor 置信度， 类别  box
        finalScores = torch.Tensor([])
        finalAnchorBoxesIndexes = torch.Tensor([]).long() # 类别
        finalAnchorBoxesCoordinates = torch.Tensor([])

        # 针对一个类所有的box进行处理
        # classification shape: [Batch, K, 80]
        for i in range(classification.shape[2]):
            print("----", i)
            # squeeze得到我们的scores.shape == [2,...]
            # 拿到第一个类别的所有预测值
            scores = torch.squeeze(classification[:,:, i]) #去掉一个维度
            scores_over_thresh = (scores > 0.05)  # 初步筛选
            '''
            >>> a = torch.Tensor([1,2,3,4])
            >>> b = a > 2
            >>> b
            tensor([False, False,  True,  True])
            >>> b.sum()
            tensor(2)
            '''
            if scores_over_thresh.sum() == 0:
                continue

            # 只把刚刚判定为True的部分提取出来
            '''
            >>> a[b]
                tensor([3., 4.])
            '''
            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transform_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            # 到这里 scores和anchorBoxes的个数应该一样
            # nms 后面测试替换torch vision自带的
            #anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
            anchors_nms_idx = torch_nms(anchorBoxes, scores, 0.5)
            # 上面anchors_nms_idx得到所有anchors的id.利用该id可以找到对应的anchors

            # 得分
            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            #分类的类别信息 anchors_nms_idx.shape[0]为最后得到几个结果 并且赋值为当前类标
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            # finalAnchorBoxesIndexes里面存储的应该是 [0,0,1,1,1, 2, 2,79] 假设该图共有8个目标：两个0 三个1 两个2 一个79
            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            # 得到最后的bbox
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
        # finalScores finalAnchorBoxesIndexes finalAnchorBoxesCoordinates 这三个的 shape[0]应该相同 即当前图像中检测到的目标个数
        return finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates
if __name__ == '__main__':
    C = torch.randn([2, 3, 512, 512])
    model = RetinaNet(80)
    out = model(C)
    for i in range(len(out)):
        print(out[i].shape)
# torch.Size([2, 49104, 4])
# torch.Size([2, 49104, 80])  # 这里80是类别置信度。

