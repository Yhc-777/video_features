这是frok自@v-iashin 大佬的库，我魔改了一点点，加上了CLIP特征提取还有一些保存的参数

文档暂时还没时间弄呢，版本也有点混乱QWQ

假如想用的可以暂时参考下面的源文档和`main.py`里的argparse代码，等我论文发出来了就完善下这个=. =

下面是原库的README

# Extract Video Features Using Multiple GPUs

This is the source code for `video_features`, a small library that allows you to extract features from raw videos using the pre-trained nets. So far, it supports several extractors that capture visual appearance, calculates optical flow, and, even, audio features.

The source code was intended to support the feature extraction pipeline for two of my papers ([BMT](https://arxiv.org/abs/2005.08271) and [MDVC](https://arxiv.org/abs/2003.07758)). This small library somehow emerged out of that code and now has more models implemented.

### [Documentation is here](https://iashin.ai/video_features/)
