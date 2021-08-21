import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_CLASSES_COLOR = [(174, 175, 133), (17, 84, 219), (46, 138, 54),
                     (253, 151, 96), (75, 242, 162), (173, 150, 67),
                     (232, 46, 160), (83, 226, 155), (80, 150, 1),
                     (127, 246, 43), (167, 126, 221), (132, 20, 125),
                     (192, 240, 135), (111, 67, 22), (56, 53, 178),
                     (74, 215, 29), (14, 69, 126), (191, 60, 67),
                     (56, 119, 196), (84, 48, 194)]


class VocDetection(Dataset):
    def __init__(self,
                 root_dir,
                 image_sets=[('2007', 'train')],
                 transform=None,
                 keep_difficult=False):
        self.root_dir = root_dir
        self.image_set = image_sets
        self.transform = transform
        self.cats = VOC_CLASSES
        #self.num_classes =
        self.cat_to_voc_label = {cat: i for i, cat in enumerate(self.cats)}
        self.voc_label_to_cat = {i: cat for i, cat in enumerate(self.cats)}

        self.keep_difficult = keep_difficult

        self.annotpath = os.path.join('%s', 'Annotations', '%s.xml')
        self.imagepath = os.path.join('%s', 'JPEGImages', '%s.jpg')

        self.ids = list()
        self.img_ids = []
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root_dir, 'VOC' + year)
            img_id = 0
            for line in open(
                    os.path.join(rootpath, 'ImageSets', 'Main',
                                 name + '.txt')):
                self.ids.append((rootpath, line.strip()))
                self.img_ids.append(img_id)
                img_id += 1

    def coco_label_to_label(self, coco_label):
        # # 90 : 79  cat id ： coco id
        return self.cat_to_voc_label[coco_label]
    def label_to_coco_label(self, label):
        return self.voc_label_to_cat[label]
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        annots, origin_hw = self.load_annots(idx)

        sample = {
            'img': image,
            'annot': annots,
            'scale': np.float32(1.),
            'origin_hw': origin_hw,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        img = cv2.imread(self.imagepath % self.ids[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # hwc rgb
        return img.astype(np.float32) / 255.

    def image_aspect_ratio(self, image_index):
        img = cv2.imread(self.imagepath % self.ids[image_index])
        # hwc
        return float(img.shape[1]) / float(img.shape[0])

    def num_classes(self):
        # 换其他数据集需要更改！！
        return len(self.cats)
    def load_annots(self, idx):
        target = ET.parse(self.annotpath % self.ids[idx]).getroot()
        annots = []

        size = target.find('size')
        h, w = int(size.find('height').text), int(size.find('width').text)
        origin_hw = np.array([h, w])

        for obj in target.iter('object'):
            difficult = (int(obj.find('difficult').text) == 1)
            if not self.keep_difficult and difficult:
                continue

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for pt in pts:
                cur_pt = float(bbox.find(pt).text)
                bndbox.append(cur_pt)

            if bndbox[2] - bndbox[0] < 1 or bndbox[3] - bndbox[1] < 1:
                continue

            if bndbox[0] < 0 or bndbox[1] < 0 or bndbox[2] > w or bndbox[3] > h:
                continue

            if name not in self.cats:
                continue

            bndbox.append(self.cat_to_voc_label[name])
            # [xmin, ymin, xmax, ymax, voc_label]
            annots += [bndbox]

        # format:[[x1, y1, x2, y2, voc_label], ... ]
        annots = np.array(annots)

        return annots.astype(np.float32), origin_hw.astype(np.float32)


if __name__ == '__main__':
    jsonpath = r"/mnt/f/dataset/voc2007/VOCtrainval_06-Nov-2007/VOCdevkit"
    voc = VocDetection(jsonpath)
    print(len(voc))

    # DataLoader加载数据集
    from torch.utils.data import DataLoader
    dataloader_train = DataLoader(voc, num_workers=0)  # dataloader(也可能是dataset)默认会把array变成tensor
    # 一个一个拿
    train = next(iter(dataloader_train))
    print(type(train))
    print(train)