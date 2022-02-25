import collections.abc
from itertools import repeat


###### 定义用到的部分工具函数 ######


# 产生维度为n的tuple
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

# 转化预训练模型
def _conv_filter(state_dict, patch_size = 16):
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict