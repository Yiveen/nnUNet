from typing import Tuple, Union, List

from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np


class DownsampleSegForDSTransform2(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    '''
        data_dict['output_key'] 将是一个列表，包含按 ds_scales 缩放的分割图。
        '''

    def __init__(self, ds_scales: Union[List, Tuple],
                 order: int = 0, input_key: str = "seg",
                 output_key: str = "seg", axes: Tuple[int] = None):
        """
        根据 ds_scales 下采样 data_dict[input_key]。ds_scales 中的每个条目指定一个深度监督输出及其相对于原始数据的分辨率，
        例如 0.25 表示原始形状的 1/4。ds_scales 也可以是元组的元组，例如 ((1, 1, 1), (0.5, 0.5, 0.5))，以独立指定每个轴的下采样。
        """
        self.axes = axes  # 指定进行下采样的轴
        self.output_key = output_key  # 输出键，用于在 data_dict 中存储输出
        self.input_key = input_key  # 输入键，用于在 data_dict 中获取输入数据
        self.order = order  # 插值的顺序，0 表示最近邻插值
        self.ds_scales = ds_scales  # 下采样比例

    def __call__(self, **data_dict):
        if self.axes is None:
            axes = list(range(2, data_dict[self.input_key].ndim))  # 如果没有指定轴，则默认对除了第一维和第二维以外的所有维度进行下采样
        else:
            axes = self.axes  # 使用指定的轴进行下采样

        output = []  # 初始化输出列表
        for s in self.ds_scales:  # 遍历所有下采样比例
            if not isinstance(s, (tuple, list)):
                s = [s] * len(axes)  # 如果 s 不是元组或列表，则将其扩展为与轴数量相同的列表
            else:
                assert len(s) == len(axes), f'If ds_scales is a tuple for each resolution (one downsampling factor ' \
                                            f'for each axis) then the number of entried in that tuple (here ' \
                                            f'{len(s)}) must be the same as the number of axes (here {len(axes)}).'

            if all([i == 1 for i in s]):
                output.append(data_dict[self.input_key])  # 如果所有下采样比例都是 1，则直接添加原始数据到输出列表
            else:
                new_shape = np.array(data_dict[self.input_key].shape).astype(float)  # 获取输入数据的形状，并转换为浮点数, 这里只是shape，不是元素值！！！
                for i, a in enumerate(axes):  # 遍历每个轴和对应的下采样比例
                    new_shape[a] *= s[i]  # 根据下采样比例调整形状
                new_shape = np.round(new_shape).astype(int)  # 将新形状四舍五入并转换为整数
                out_seg = np.zeros(new_shape, dtype=data_dict[self.input_key].dtype)  # 创建新形状的全零数组
                for b in range(data_dict[self.input_key].shape[0]):  # 遍历第一维
                    for c in range(data_dict[self.input_key].shape[1]):  # 遍历第二维
                        # 对每个分割图进行下采样，并将结果存储在 out_seg 中
                        out_seg[b, c] = resize_segmentation(data_dict[self.input_key][b, c], new_shape[2:], self.order)
                output.append(out_seg)  # 将下采样后的分割图添加到输出列表
        data_dict[self.output_key] = output  # 将输出列表存储在 data_dict 的输出键下
        return data_dict  # 返回更新后的 data_dict
