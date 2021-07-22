#coding=utf-8

import torch
from torch import Tensor

def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] # N中一个和M个比较； 所以由N，M 个
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]  #小于0的为0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  交集

    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # NxM， boxes1中每个框和boxes2中每个框的IoU值；

def nms(boxes, scores, iou_thresh):
    '''
    Args:
        boxes:传进来的框，[N, 4]
        scores: [N]  类别置信度
        iou_thresh:
    Returns: 返回的是保留的框的下标
    '''

    keep = []
    idx = scores.argsort()  # 将置信度从小到大排列
    '''
    >>> a = np.array([2,3,1,5,3,6,8])
    >>> a.argsort()
    array([2, 0, 1, 4, 3, 5, 6], dtype=int64)  返回的是下标。  从小到大排列
    '''
    while len(idx) > 0:
        # 先得到得分最大的框的索引以及对应坐标
        max_score_idx = idx[-1]
        # 在数组索引中，加入None就相当于在对应维度加一维
        '''
        >>> a
            array([2, 3, 1, 5, 3, 6, 8])
        >>> a[None, :]
            array([[2, 3, 1, 5, 3, 6, 8]])
        '''
        max_score_box = boxes[max_score_idx][None, :] # [1, 4]
        keep.append(max_score_idx)
        if len(idx) == 1:
            break
        idx = idx[:-1] # 删除最后一个框
        # 剩余索引对应的框 和 得分最大框 计算IoU
        other_box = boxes[idx] # [?, 4]
        ious = box_iou(max_score_box, other_box) # 一个框和其余框比较 1XM
        idx = idx[ious[0]<=iou_thresh] # 保留小于阈值的框
    keep = idx.new(keep) # 转换为tensor  要测试下 是否等价keep=torch.from_numpy(keep)
    return keep

