import argparse
import torch
from torchvision import transforms

import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from model import retinanet
from model.dataset.coco import COCODataset
from model.dataset.collator import collater
from model.dataset.transform import *
from model.dataset.sampler import *

from torch.utils.data import DataLoader

import eval_coco
from model.retinanet import RetinaNet

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple validation script for training a RetinaNet network.')

    parser.add_argument('--json_path', help='Path to COCO directory',
                        default='E:/01_Code/02_Python/01_CV/dataset/coco/instances_val2017.json')
    parser.add_argument('--img_path', help='Path to COCO directory',
                        default='E:/01_Code/02_Python/01_CV/dataset/coco/val2017')
    parser.add_argument('--model_path', help='Path to model (.pt) file.',
                        default='D:/07_codeD/MyCvCode/RetinaNet/weights/model_final.pt')

    parser = parser.parse_args(args)

    dataset_val = COCODataset(parser.json_path, img_path=parser.img_path,
                              transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
    # retinanet = torch.load(parser.model)

    # Create the model
    # retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet = RetinaNet(num_classes=dataset_val.num_classes())

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.load(parser.model_path)

        # retinanet.load_state_dict(torch.load(parser.model_path))
        # retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    # retinanet.module.freeze_bn()
    retinanet.freeze_bn()


    eval_coco.evaluate_coco(dataset_val, retinanet)


if __name__ == '__main__':
    main()