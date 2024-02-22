import os
from typing import List

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from nnunetv2.training.dataloading.utils import get_case_identifiers
import h5py
from scipy.sparse import csr_matrix, save_npz

class nnUNetDataset(object):
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        """
        This does not actually load the dataset. It merely creates a dictionary where the keys are training case names and
        the values are dictionaries containing the relevant information for that case.
        dataset[training_case] -> info
        Info has the following key:value pairs:
        - dataset[case_identifier]['properties']['data_file'] -> the full path to the npz file associated with the training case
        - dataset[case_identifier]['properties']['properties_file'] -> the pkl file containing the case properties

        In addition, if the total number of cases is < num_images_properties_loading_threshold we load all the pickle files
        (containing auxiliary information). This is done for small datasets so that we don't spend too much CPU time on
        reading pkl files on the fly during training. However, for large datasets storing all the aux info (which also
        contains locations of foreground voxels in the images) can cause too much RAM utilization. In that
        case is it better to load on the fly.

        If properties are loaded into the RAM, the info dicts each will have an additional entry:
        - dataset[case_identifier]['properties'] -> pkl file content

        IMPORTANT! THIS CLASS ITSELF IS READ-ONLY. YOU CANNOT ADD KEY:VALUE PAIRS WITH nnUNetDataset[key] = value
        USE THIS INSTEAD:
        nnUNetDataset.dataset[key] = value
        (not sure why you'd want to do that though. So don't do it)
        """
        super().__init__()
        # print('loading dataset')
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers_filtered = [f for f in case_identifiers if 'key' not in f] #TODO:记住这个写法！！

        case_identifiers_filtered.sort()
        #所有训练数据,没有进行k折交叉验证分开['arota_001', 'arota_003', 'arota_005', 'arota_006', 'arota_008', 'arota_009', 'arota_010', 'arota_011', 'arota_012', 'arota_013', 'arota_014', 'arota_016', 'arota_017', 'arota_018', 'arota_019', 'arota_020', 'arota_021', 'arota_022', 'arota_023', 'arota_024', 'arota_025', 'arota_026', 'arota_027', 'arota_029', 'arota_030', 'arota_033', 'arota_035', 'arota_036', 'arota_037']


        self.dataset = {}
        for c in case_identifiers_filtered:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder, f"{c}.npz") #对应processed文件夹下面的npz文件
            self.dataset[c]['properties_file'] = join(folder, f"{c}.pkl") #对应processed文件夹下面的pkl文件

            key_files_path = join(folder, f"{c}_key.npz")
            if os.path.exists(key_files_path):
                self.dataset[c]['key'] = join(folder, f"{c}key.npz")
            else:
                self.dataset[c]['key'] = None

            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.npz")

        if len(case_identifiers_filtered) <= num_images_properties_loading_threshold:#num_images_properties_loading_threshold==0
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and \
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))
        # print(f'nnUNetDataset.keep_files_open: {self.keep_files_open}')

    def __getitem__(self, key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def load_case(self, key, stage):
        entry = self[key]
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r') #(1,170,241,241)
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
                # print('saving open data file')
        else:
            data = np.load(entry['data_file'])['data']

        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
                # print('saving open seg file')
        else:
            seg = np.load(entry['data_file'])['seg']

        key_label = None

        if stage == 2 or stage == 3:
            if 'open_key_file' in entry.keys():
                seg = entry['open_key_file']
            elif isfile(entry['data_file'][:-4] + "_key.h5"):
                key_label = np.load(entry['data_file'][:-4] + "_key.npz")['key']
                # with h5py.File(entry['data_file'][:-4] + "_key.h5", 'r') as f:
                #     # 从文件中读取稀疏矩阵的组成部分
                #     data_h5 = f['data'][:]
                #     indices = f['indices'][:]
                #     indptr = f['indptr'][:]
                #     sparse_shape = f['shape'][:]
                #
                #     # 重构稀疏矩阵
                #     sparse_matrix = csr_matrix((data_h5, indices, indptr), shape=sparse_shape)
                #     dense_array = sparse_matrix.toarray()
                #     # 读取原始形状
                #     original_shape = f.attrs['original_shape']
                #     key_label = dense_array.reshape(original_shape)

                if self.keep_files_open:
                    self.dataset[key]['open_key_file'] = key_label
            else:
                key_label = None

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev[None]))
        # if key_label is None:
        #     return data, seg, entry['properties']
        # else:
        return data, seg, entry['properties'], key_label


if __name__ == '__main__':
    # this is a mini test. Todo: We can move this to tests in the future (requires simulated dataset)

    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset003_Liver/3d_lowres'
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0) # this should not load the properties!
    # this SHOULD HAVE the properties
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # amazing. I am the best.

    # this should have the properties
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=1000)
    # now rename the properties file so that it does not exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now we should still be able to access the properties because they have already been loaded
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # move file back
    shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

    # this should not have the properties
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0)
    # now rename the properties file so that it does not exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now this should crash
    try:
        ks = ds['liver_0'].keys()
        raise RuntimeError('we should not have come here')
    except FileNotFoundError:
        print('all good')
        # move file back
        shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

