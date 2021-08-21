#coding=utf-8
'''
retinaNet的focal loss
整体而言，相当于增加了分类不准确样本在损失函数中的权重。

实际上这里应该叫做总损失，其中的分类损失是focal loss。回归损失有好几种，我们这里用smooth l1 loss。

'''
import torch
import torch.nn as nn

# 可视化用的
from model.dataset.transform import *
import cv2
import random
unnormalize = UnNormalizer()
# a ,b 格式为 x1,y1,x2,y2
def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    '''
    >>> a
    tensor([1., 2., 3., 4., 5., 6.])
    >>> a = torch.unsqueeze(a, dim=1)
    >>> a
    tensor([[1.],
            [2.],
            [3.],
            [4.],
            [5.],
            [6.]])
    >>> b = torch.Tensor([3,4,5])
    >>> torch.min(a,b)
    tensor([[1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
            [3., 4., 4.],
            [3., 4., 5.],
            [3., 4., 5.]])
            
    这里计算广播后的最小：
    [
    1   345  --> 111
    2   345  --> 222
    3   345  ->  333
    4   345  ->  344
    5   345  ->  345
    6   345  ->  345
    ]
    '''
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    # 标签coco是xywh但是在加载的时候转化成了xyxy
    # anno torch.Size([1, 9, 5]) xyxy catagory
    def forward(self, classifications, regressions, anchors, annotations, srcimg=None):
        '''
        classifications分类分支输出[batch, K, 80]
        regressions会回归分支输出[Batch, K, 4]
        anchors 预设的anchors [1, K, 4]
        annotations真实标签 [1, M, 5]
        alpha * (1-p)^ gamma
        '''
        alpha = 0.25 #  0.5表示均衡
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        anchor = anchors[0, :, :] # anchor: xyxy
        #首先转为 cx cy w h
        #因为模型的输出需要利用cx cy w h转换成bbox
        anchor_w = anchor[:, 2] -anchor[:, 0]
        anchor_h = anchor[:, 3]-anchor[:, 1]
        anchor_cx = anchor[:, 0] + 0.5 * anchor_w
        anchor_cy = anchor[:, 1] + 0.5 * anchor_h

        # 开始遍历batch
        for j in range(batch_size):

            # 可视化
            ssss = srcimg[j, :, :, :].clone()
            img = np.array(255 * unnormalize(ssss).cpu())
            img[img < 0] = 0
            img[img > 255] = 255
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            #拿到一个图的模型输出
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annot = annotations[j, :, :] # (N, 5) [x1, y1, x2, y2 id]  真实标签
            # 去除id != -1
            bbox_annot = bbox_annot[bbox_annot[:, 4] != -1]
            # 将分类置信度clamp到0-1
            # sigmoid
            #classification = torch.sigmoid(classification) # [N, 80]  N为生成的anchor个数
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            # 如果没有GT，那就看作为负样本，-(1-alpha) * p ^ gamma log(1-p)  # p是模型预测的概率值
            if bbox_annot.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha
                    alpha_factor = 1 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    # 交叉熵
                    bce = -(torch.log(1.0 - classification))
                    cls_loss = focal_weight * bce  # focal loss
                    classification_losses.append(cls_loss.mean()) # TODO 用mean还是sum需要验证
                    regression_losses.append(torch.tensor(0).float().cuda())  # 负样本只分类loss
                else:
                    alpha_factor = torch.ones(classification.shape) * alpha
                    alpha_factor = 1 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    # 交叉熵
                    bce = -(torch.log(1.0 - classification))
                    cls_loss = focal_weight * bce  # focal loss
                    classification_losses.append(cls_loss.mean())
                    regression_losses.append(torch.tensor(0).float()) # 负样本只分类loss

                continue
            # 存在GT
            # 先计算所有anchor和gt的iou，区分出正样本和负样本
            # 假设共有4567个anchor。gt由15个。 要计算出（4567， 15）维度的矩阵
            # xyxy
            IOU = calc_iou(anchors[0, :, :], bbox_annot[:, :4]) # num_anchors x num_annotations
            '''
            >>> torch.Tensor([[1,3,2,4], [2,8,4,2]])
            tensor([[1., 3., 2., 4.],
                    [2., 8., 4., 2.]])
            >>> a=torch.Tensor([[1,3,2,4], [2,8,4,2]])
            >>> b,c = torch.max(a, dim=1)  # 每一行的最大值
            >>> b   #为每一行最大值
            tensor([4., 8.])
            >>> c    # 最大值的下标
            tensor([3, 1])
            
            >>> b,c = torch.max(a, dim=0)  # 每一列的最大值
            >>> b
            tensor([2., 8., 4., 4.])
            >>> c
            tensor([1, 1, 1, 0])
            '''
            IoU_max, IoU_argmax = torch.max(IOU, dim=1)  # num_anchors x 1

            # 开始计算loss
            targets = torch.ones(classification.shape) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            # iou小于0.4(即和gt的box 离得远的) 全部设置为0
            # target==1说明为正样本。target==0为负样本
            targets[torch.lt(IoU_max, 0.4), :] = 0
            # 找到正样本 返回[f, t, t, f, ....]
            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum() # 正样本个数
            # 将正样本target设置为1,注意这里部分设置为1
            assigned_annotations = bbox_annot[IoU_argmax, :] # 拿到每行中iou最大的bbox 真实标签
            targets[positive_indices, :] = 0
            # 仅仅把 iou>0.5 并和gt iou最大框的 id设为1
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            # 正样本 -（alpha）（1-p） ^ gamma log(p)
            # 负样本 - (1-alpha) p ^ gamma lpg(1-p)
            # torch.where()函数的作用是按照一定的规则合并两个tensor类型。
            # 将alpha_factor中target==1对应位置的值保留。 其余位置替换为1. - alpha_factor
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            # 正样本 weight: 1. - classification    负样本： classification
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)

            # * 对应元素相乘
            # @ 对应元素相乘在相加
            # torch.mm数学上的矩阵乘法，要满足矩阵维度
            #torch.matmul   torch.mm的broadcast版本.
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            #交叉熵 正负样本都要计算
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            # 此时bce也是个矩阵，大小为classification.shape
            cls_loss = focal_weight * bce  #ele-wise
            if torch.cuda.is_available():
                # 如果不等于-1
                # torch.ne 两个元素不等返回true
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                # [4567,80]
                # 将目标为-1的loss值赋值为0
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
            #classification_losses.append(cls_loss.mean())

            # 开始计算回归损失
            # 只有正样本才计算回归损失
            if positive_indices.sum() > 0:
                # assigned_annotations时iou最大的anchor
                assigned_annotations = assigned_annotations[positive_indices, :]  # 真是类标

                anchor_widths_pi = anchor_w[positive_indices]
                anchor_heights_pi = anchor_h[positive_indices]
                anchor_ctr_x_pi = anchor_cx[positive_indices]
                anchor_ctr_y_pi = anchor_cy[positive_indices]

                # 可视化正样本anchors
                a_x1 = anchor_ctr_x_pi - anchor_widths_pi /2
                a_y1 = anchor_ctr_y_pi - anchor_heights_pi /2
                a_x2 = anchor_ctr_x_pi + anchor_widths_pi /2
                a_y2 = anchor_ctr_y_pi + anchor_heights_pi /2
                for idx in range(len(a_x1)):
                    cv2.rectangle(img, (int(a_x1[idx].item()), int(a_y1[idx].item())),
                                  (int(a_x2[idx].item()), int(a_y2[idx].item())), (random.randint(0,255),
                                                                                   random.randint(0,255),
                                                                                   random.randint(0,255)))
                # gt
                for gt in assigned_annotations:
                    cv2.rectangle(img, (int(gt[0].item()), int(gt[1].item())),
                                  (int(gt[2].item()), int(gt[3].item())), (0, 0, 255), 4)

                cv2.imshow("anchor", img)
                cv2.waitKey(100)

                # 计算gt xyxy -> xywh
                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                # smooth l1 loss
                # -1<（y-y^）<1 : 0.5*（y-y^）**2
                # other  abs(（y-y^）)-0.5
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                # Concatenates a sequence of tensors along a new dimension.
                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    # 权重
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                regression_diff = torch.abs(targets - regression[positive_indices, :])  # y-y^
                # 小于和大于0.111的用不同方式计算
                '''
                按照公式
                regression_loss = torch.where(
                    # torch.le <=
                    torch.le(regression_diff, 1.0),
                    0.5 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5
                )
                '''
                regression_loss = torch.where(
                    # torch.le <=
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                # 全部stack后平均，饭后最后的cls loss ,reg loss
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


if __name__ == '__main__':
    # def forward(self, classifications, regressions, anchors, annotations):
    c = torch.randn([1,4567,80]).cuda()
    r = torch.randn([1,4567,4]).cuda()
    a = torch.randn([1,4567,4]).cuda()
    anno = torch.randn([1,15,5]).cuda()
    model = FocalLoss().cuda()
    out = model(c,r,a,anno)
    for i in range(len(out)):
        print(out[i]) # 输出的是 cls loss和regre loss