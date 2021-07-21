初衷：根据博客学习retinanet以及pytorch的使用。并构建自己的pytorch model习惯

知乎地址：https://zhuanlan.zhihu.com/p/384123384

![image-20210713170452200](imgs/image-20210713170452200.png)

1 backbone

![preview](imgs/v2-3652f91ecc0683fa73d03fa99c26ab82_r.jpg)

![preview](imgs/v2-b1ac9497249c5de6b812b1af729f4c44_r.jpg)

图片来源[ResNet50网络结构图及结构详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/353235794)

这里注意只有在每个Stage的第一步，ResBlock才会有downsample操作。这里只有stride!=1 或者输出的通道数要是输入的4倍，才是每个stage的第一步。

2 FPN

![image-20210713133235722](imgs/image-20210713133235722.png)

![image-20210713165336952](imgs/image-20210713165336952.png)

[目标检测系列一：RetinaNet之anchor_何以解忧 唯有专注-CSDN博客](https://blog.csdn.net/qq_36251958/article/details/105024133)

