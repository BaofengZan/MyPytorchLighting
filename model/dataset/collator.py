#coding=utf-8
'''
实现自定义的collater函数
使用这个参数可以自己操作每个batch的数据 。为一个函数，输入为一批数据，输出处理后的数据。
'''
from __future__ import print_function, division
import torch
import numpy as np


# 数据预读取
# 所谓数据预读取就是模型在进行本次batch的前向计算和反向传播时就预先加载下一个batch的数据，
# 这样就节省了下次加载数据的时间（相当于加载数据与前向计算和反向传播并行了）

class COCODataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_annot = sample['img'], sample['annot']
        except StopIteration:
            self.next_input = None
            self.next_annot = None
        with torch.cuda.stream(self.stream):
            '''
            non_blocking默认值为False, 通常我们会在加载数据时，将DataLoader的参数pin_memory设置为True,
            DataLoader中参数pin_memory的作用是：将生成的Tensor数据存放在哪里，值为True时，意味着生成的
            Tensor数据存放在锁页内存中，这样内存中的Tensor转义到GPU的显存会更快。
            主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟
            内存进行交换（注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。
            显卡中的显存全部是锁页内存,当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，
            或者交换内存使用过多的时候，设置pin_memory=False。
            如果pin_memory=True的话，将数据放入GPU的时候，也应该把non_blocking打开，
            这样就只把数据放入GPU而不取出，访问时间会大大减少。
            '''
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_annot = self.next_annot.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        annot = self.next_annot
        self.preload()
        return input, annot

# 对于一个batch的images和annotations，
# 我们最后还需要用collater函数将images和annotations的shape全部对齐后才能输入模型进行训练
'''
对于images，由于我们前面的Resize类已经将其shape对齐了，
所以这里不再做处理。对于annotations，由于每张图片标注的object数量都不一样，
还有可能出现某张图上没有标注object的情况。
我们取一个batch中所有图片里单张图片中标注object数量的最大值，
然后用值-1填充其他图片的annotations，
使得所有图片的annotations中object数量都等于这个最大值。
在进行训练时，我们会在loss部分处理掉这部分值-1的annotations。
'''
def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    heights = [int(s.shape[0]) for s in imgs]
    widths = [int(s.shape[1]) for s in imgs]

    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_height, max_width, 3) # 这里应该hwc的矩阵。

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    #padded_imgs = torch.from_numpy(np.stack(imgs, axis=0))
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    # hwc转chw
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    # 要转为tensor么？ 不用，因为该函数在DataLoader内部使用， DataLoader内部会自动转为tensort的
    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
