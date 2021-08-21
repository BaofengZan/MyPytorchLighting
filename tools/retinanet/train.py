# coding=utf-8
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from model import retinanet
from model.dataset.coco import COCODataset
from model.dataset.voc import VocDetection
from model.dataset.collator import collater
from model.dataset.transform import *
from model.dataset.sampler import *

from torch.utils.data import DataLoader  # 加载COCODataset对象的

import eval_coco
import eval_voc
from model.retinanet import RetinaNet

import cv2


def main(args=None):
    # 先解析参数
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--jsonPath', default="/mnt/f/dataset/tiny_coco/tiny_coco-master/annotations/instances_train2017.json")
    parser.add_argument('--imgPath', default="/mnt/f/dataset/tiny_coco/tiny_coco-master/images/train2017")
    parser.add_argument('--Dataset', default="coco", help="coco/voc")
    parser.add_argument('--epochs', default=100, type=int)
    parser = parser.parse_args(args)

    # 创建datalaoder
    transform = transforms.Compose([Normalizer(), Augmenter(), Resizer(400, 667)])

    if (parser.Dataset == "coco"):
        dataset_train = COCODataset(parser.jsonPath, parser.imgPath, transform=transform)
        # dataset_val = COCODataset(parser.jsonPath, parser.imgPath, transform=transform)
        dataset_val = COCODataset("/mnt/f/dataset/tiny_coco/tiny_coco-master/annotations/instances_val2017.json",
                                  "/mnt/f/dataset/tiny_coco/tiny_coco-master/images/val2017", transform=transform)
    elif (parser.Dataset == "voc"):
        dataset_train = VocDetection("/mnt/f/dataset/voc2007/VOCtrainval_06-Nov-2007/VOCdevkit", image_sets=[('2007', 'val')], transform=transform)
        dataset_val = VocDetection("/mnt/f/dataset/voc2007/VOCtrainval_06-Nov-2007/VOCdevkit",
                                   image_sets=[('2007', 'val')], transform=transform)
    else:
        print("不支持的数据集")
        exit(-1)

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)

    retinanet = RetinaNet(number_classes=dataset_train.num_classes())

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

    # 优化函数
    optimizer = optim.Adam(retinanet.parameters(), lr=0.0001)
    # 学习率调整规则
    # 添加warm up
    warm_up = True
    if warm_up:
        lr_func = lambda epoch: 0.1 ** len([m for m in [8, 11] if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    retinanet.train()
    retinanet.freeze_bn()

    for epoch_num in range(parser.epochs):
        retinanet.training = True
        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            # debug
            # print(data)
            optimizer.zero_grad()  # 清空梯度
            # 前向
            if torch.cuda.is_available():
                # 这里img annot都要cuda()
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss
            print("epoch: {} -- loss: {}".format(epoch_num, loss))
            if (loss == 0):
                continue
            loss.backward()  # loss 反传
            # clip作用？？？
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()  # 优化器优化
            epoch_loss.append(float(loss))
            del classification_loss
            del regression_loss

        print('Evaluating dataset')
        if (parser.Dataset == "coco"):
            retinanet.training = False
            eval_coco.evaluate_coco(dataset_val, retinanet)
        elif (parser.Dataset == "voc"):
            retinanet.training = False
            eval_voc.evaluate_voc(dataset_val, retinanet)
        scheduler.step()
        if (epoch_num % 10 == 0):
            torch.save(retinanet, 'retinanet_epoch{}.pt'.format(epoch_num))
    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
    # https://github.com/yatengLG/Retinanet-Pytorch
    # imgpath = r"E:\Datasets\oneimg\aachen_000003_000019_rightImg8bit.png"
    # im = cv2.imread(imgpath)
    # cv2.imshow("re", im)
    # cv2.waitKey(0)
