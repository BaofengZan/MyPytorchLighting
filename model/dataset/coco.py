#coding=utf-8
'''
解析coco数据集
'''

from __future__ import print_function, division
import os
import numpy as np
import skimage
import skimage.io
import skimage.transform
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2

class COCODataset(Dataset):
    def __init__(self, jsonPath, imgPath, transform=None):
        self.json_path = jsonPath
        self.img_path = imgPath
        #在建立dataset的时候，我们这里可以进行数据的预处理了，也就是transform部分 torchvision.transforms
        self.transform = transform

        self.coco = COCO(self.json_path)
        #image_ids = [397133, 37777, 252219, 87038, 174482, 403385, 6818, 480985, 458054,...]
        self.img_ids = self.coco.getImgIds()
        # 提取coco的类标 所有的类标 这里是80个 其中一个值 {'supercategory': 'person', 'id': 1, 'name': 'person'}
        # 这里面的id就是类标的id即category_id。 比如：0-79 coco里面id是0-90.因为有些跳过了。但是总数是80个
        categories = self.coco.loadCats(self.coco.getCatIds()) # categories是一个dict
        categories.sort(key=lambda x: x['id']) # 按照id排序
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories: #
            self.coco_labels[len(self.classes)] = c['id']  # 这个是 0-79和josn的id对应map 比如 79:90
            self.coco_labels_inverse[c['id']] = len(self.classes)  # 和上面相反  90：79
            self.classes[c['name']] = len(self.classes) # "person":0   "toolbrash":79
        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key  #  刚好和self.classes相反
    def __len__(self):
        return len(self.img_ids)  # 多少个objs

    def load_img(self, idx):
        # 根据下标读取图像
        imginfo = self.coco.loadImgs(self.img_ids[idx])[0]
        path = os.path.join(self.img_path, imginfo["file_name"])
        '''
        保存后的都是numpy格式，但cv2的存储格式是BGR，而skimage的存储格式是RGB
        都是hwc
        '''
        # 这里用skimage
        img = skimage.io.imread(path)
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)  #灰度图转rgb

        return img.astype(np.float32) / 255.0  # 归一化
    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]
    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def num_classes(self):
        # 换其他数据集需要更改！！
        return 80

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.img_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_annotations(self, idx):
        annotations_ids = self.coco.getAnnIds(imgIds=self.img_ids[idx], iscrowd=False)
        annotations = np.zeros((0, 5))  # 类标  以及 坐标
        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # 解析annot
        # 一个图上可能有多个目标
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            anno = np.zeros((1, 5)) # 1行5列
            anno[0, :4] = a['bbox']
            anno[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, anno, axis=0)

        ## transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        return annotations

    def __getitem__(self, idx):
        # 根据idx拿值
        img = self.load_img(idx)
        annot = self.load_annotations(idx)
        sample= {"img":img, "annot":annot}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    jsonpath = r"E:\Datasets\tiny_coco\tiny_coco-master\annotations\instances_train2017.json"
    coco = COCODataset(jsonpath, r"E:\Datasets\tiny_coco\tiny_coco-master\images\train2017")
    print(len(coco))

    # DataLoader加载数据集
    dataloader_train = DataLoader(coco, num_workers=0) # dataloader(也可能是dataset)默认会把array变成tensor
    # 一个一个拿
    train = next(iter(dataloader_train))
    print(type(train))
    print(train)