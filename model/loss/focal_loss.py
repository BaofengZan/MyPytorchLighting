#coding=utf-8
'''
retinaNet的focal loss
整体而言，相当于增加了分类不准确样本在损失函数中的权重。

实际上这里应该叫做总损失，其中的分类损失是focal loss。回归损失有好几种，我们这里用smooth l1 loss。

'''
import torch
import torch.nn as nn
class FocalLoss(nn.Module):
    # 标签coco是xywh但是在加载的时候转化成了xyxy
    # anno torch.Size([1, 9, 5]) xyxy catagory
    def forward(self, classifications, regressions, anchors, annotations):
        '''
        classifications分类分支输出[batch, K, 80]
        regressions会回归分支输出[Batch, K, 4]
        anchors 预设的anchors [1, M, 4]
        annotations真实标签
        alpha * (1-p)^ gamma
        '''
        alpha = 0.25
        gamma = 0.5
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_loss = []
        anchor = anchors[0, :, :] # anchor: xyxy
        #首先转为 cx cy w h


