#coding=utf-8

'''
进行regression和anchors的计算输出！！
在retinanet model出来的是 dx dy dw dh
其中: dx 是targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths 由 gt的cx-anchor的cx 后与anchor的宽的比例
dh 类似yolov3：   targets_dw = torch.log(gt_widths / anchor_widths)

该文件作用：由模型的 dx dy dw dh 反推出预测的bbox
'''


import torch
import torch.nn as nn
import numpy as np


class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        # 均值 归一化作用？
        #？？？
    def forward(self, anchors, deltas):
        # anchors  [1, K*A, 4]
        width = anchors[:, :, 2] - anchors[:, :, 0]
        height = anchors[:, :, 3] - anchors[:, :, 1]
        c_x = anchors[:, :, 0] + 0.5 * width
        c_y = anchors[:, :, 1] + 0.5*height

        device = anchors.device
        factor = torch.tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)
        deltas = deltas * factor

        pred_cx = c_x + deltas[:, :, 0] * width
        pred_cy = c_y + deltas[:, :, 1] * height
        pred_w = torch.exp(deltas[:, :, 2].float()) * width
        pred_h = torch.exp(deltas[:, :, 3].float()) * height

        # 转换成 x1 y1 x2 y2
        pred_boxes_x1 = pred_cx - 0.5 * pred_w
        pred_boxes_y1 = pred_cy - 0.5 * pred_h
        pred_boxes_x2 = pred_cx + 0.5 * pred_w
        pred_boxes_y2 = pred_cy + 0.5 * pred_h
        # [:, :, 1] stack
        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes
if __name__ == '__main__':
    x1 = torch.randint(3, 10, (1,100,4))
    # print(x1)
    x2 = torch.randint(3, 5, (3,100,4))
    # print(x2)

    model = BBoxTransform()
    out = model(x1,x2)
    print(out.shape)
    #torch.Size([3, 100, 4])