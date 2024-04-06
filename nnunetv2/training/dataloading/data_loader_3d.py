import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices() #得到的bacth的索引
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        if self.stage == 1:
            seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        if self.key_shape is not None:
            key_all = np.zeros(self.key_shape, dtype=np.float32)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j) #False

            data, properties, key = self._data.load_case(i, stage=self.stage)#hard-code!!
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:] #(143,215,215) 这个不同的输入会不一样
            dim = len(shape)
            all_zero_count = -1
            while True:
                all_zero_count += 1
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['category_k_locations'])

                # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
                # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
                # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
                # later
                # 以下代码先裁剪数据到实际位于数据中的边界框区域，这将导致一个更小的数组，随后更快地进行填充
                valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]  # 计算有效的边界框下界
                valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]  # 计算有效的边界框上界，实际debug的情况是范围超了

                # 使用有效的边界框裁剪数据和分割标签
                this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                data_sliced = data[this_slice]  # 裁剪图像数据
                if self.stage == 1:
                    this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                    seg_sliced = seg[this_slice]  # 裁剪分割标签

                if key is not None and self.stage != 1:
                    this_slice = tuple([slice(0, key.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                    key_sliced = key[this_slice]  # 裁剪分割标签
                    non_zero_indices = np.where(key_sliced > 0)
                    # if len(non_zero_indices[0]) != 0 or all_zero_count >= 2:
                    if len(non_zero_indices[0]) != 0 or len(non_zero_indices[1]) != 0 or len(non_zero_indices[2]) != 0 or len(non_zero_indices[3]) != 0 or all_zero_count >= 2:
                        break
                else:
                    break

            # 计算填充的大小，保证裁剪后的数据和分割标签与补丁大小匹配， 因为实际上并没有对need_to_pad进行处理
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data_sliced, ((0, 0), *padding), 'constant', constant_values=0)  # 填充图像数据
            if self.stage == 1:
                seg_all[j] = np.pad(seg_sliced, ((0, 0), *padding), 'constant', constant_values=-1)  # 填充分割标签

            if key is not None:
                # for category in range(key.shape[0]):
                key_all[j] = np.pad(key_sliced, ((0, 0), *padding), 'constant', constant_values=0)  # 填充key标签
        if self.stage != 1:
            # print('11111', key_all.shape)
            return {'data': data_all, 'properties': case_properties, 'keys': selected_keys,
                'key_points': key_all}
        else:
            # 返回包含图像数据、分割标签、样本属性和样本键的字典
            return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
