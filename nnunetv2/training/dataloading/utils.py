import multiprocessing
import os
from multiprocessing import Pool
from typing import List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes
from scipy.sparse import csr_matrix, save_npz
import h5py

def load_npz_file(file_path):
    # 检查文件是否存在
    if os.path.exists(file_path):
        try:
            # 尝试加载文件
            data = np.load(file_path)
            return data
        except Exception as e:
            # 如果加载过程中出现任何错误，返回None
            print(f"Error loading {file_path}: {e}")
            return None
    else:
        # 如果文件不存在，返回None
        return None

def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False) -> None:
    try:
        a = np.load(npz_file)  # inexpensive, no compression is done here. This just reads metadata
        b = load_npz_file(npz_file[:-4] + "_key.npz")
        c = load_npz_file(npz_file[:-4] + "_categoryk.npz")
        d = load_npz_file(npz_file[:-4] + "_aaa.npz")
        e = load_npz_file(npz_file[:-4] + "seg_down_sampled.npz")
        key_as_float32 = b['key'].astype(np.float32)
        # a_seg = a['seg'] #读取进来会带一个1
        # a_data = a['data']
        if overwrite_existing or not isfile(npz_file[:-3] + "npy"):
            np.save(npz_file[:-3] + "npy", a['data'])
        if unpack_segmentation and (overwrite_existing or not isfile(npz_file[:-4] + "_seg.npy")):
            np.save(npz_file[:-4] + "_seg.npy", a['seg'])
        if (overwrite_existing or not isfile(npz_file[:-4] + "_key.npy")) and b is not None:
            pass
            # np.save(npz_file[:-4] + "_key.npy", key_as_float32)
            # key_save_name = npz_file[:-4] + "_key.h5"
            # num_class, z, x, y = key_as_float32.shape
            # key_as_float32_reshaped = key_as_float32.reshape(num_class*z, x*y)
            # sparse_matrix = csr_matrix(key_as_float32_reshaped)
            # with h5py.File(key_save_name, 'w') as f:
            #     # 存储稀疏矩阵的数据、行索引和列指针
            #     f.create_dataset('data', data=sparse_matrix.data)
            #     f.create_dataset('indices', data=sparse_matrix.indices)
            #     f.create_dataset('indptr', data=sparse_matrix.indptr)
            #     f.create_dataset('shape', data=sparse_matrix.shape)
            #     # 将原始形状作为属性存储
            #     f.attrs['original_shape'] = key_as_float32.shape
        # if (overwrite_existing or not isfile(npz_file[:-4] + "_categoryk.npy")) and c is not None:
        #     np.save(npz_file[:-4] + "_categoryk.npy", c['key'])
        if (overwrite_existing or not isfile(npz_file[:-4] + "_aaa.npy")) and d is not None:
            np.save(npz_file[:-4] + "_aaa.npy", d['seg'])
        if (overwrite_existing or not isfile(npz_file[:-4] + "seg_down_sampled.npy")) and e is not None:
            np.save(npz_file[:-4] + "seg_down_sampled.npy", e['seg'])
            # key_save_dir = npz_file[:-4] + "_key.h5"
            #
            #
            #
            # for i in range(key_as_float32.shape[0]):
            #     # 遍历第二维度
            #     for j in range(key_as_float32.shape[1]):
            #         # 获取二维切片
            #         slice_2d = key_as_float32[i, j, :, :]
            #         # 将二维切片转换为稀疏矩阵
            #         sparse_matrix = csr_matrix(slice_2d)
            #         # 保存稀疏矩阵，文件名反映了它的维度索引
            #         save_npz(key_save_dir + '/' + f'sparse_matrix_{i}_{j}.npz', sparse_matrix)
            # np.save(npz_file[:-4] + "_key.npy", key_as_float32)
    except KeyboardInterrupt:
        if isfile(npz_file[:-3] + "npy"):
            os.remove(npz_file[:-3] + "npy")
        if isfile(npz_file[:-4] + "_seg.npy"):
            os.remove(npz_file[:-4] + "_seg.npy")
        if isfile(npz_file[:-4] + "_key.npy"):
            os.remove(npz_file[:-4] + "_key.npy")
        raise KeyboardInterrupt


def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        npz_files_filtered = [f for f in npz_files if 'key' not in f and 'category' not in f and 'aaa' not in f and 'sampled' not in f]
        p.starmap(_convert_to_npy, zip(npz_files_filtered,
                                       [unpack_segmentation] * len(npz_files_filtered),
                                       [overwrite_existing] * len(npz_files_filtered))
                  )


def get_case_identifiers(folder: str) -> List[str]:
    """
    finds all npz files in the given folder and reconstructs the training case names from them
    """
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/2d')