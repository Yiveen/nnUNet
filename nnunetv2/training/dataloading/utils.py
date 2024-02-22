from __future__ import annotations
import multiprocessing
import os
from typing import List
from pathlib import Path
from warnings import warn

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

def find_broken_image_and_labels(
    path_to_data_dir: str | Path,
) -> tuple[set[str], set[str]]:
    """
    Iterates through all numpys and tries to read them once to see if a ValueError is raised.
    If so, the case id is added to the respective set and returned for potential fixing.

    :path_to_data_dir: Path/str to the preprocessed directory containing the npys and npzs.
    :returns: Tuple of a set containing the case ids of the broken npy images and a set of the case ids of broken npy segmentations. 
    """
    content = os.listdir(path_to_data_dir)
    unique_ids = [c[:-4] for c in content if c.endswith(".npz")]
    failed_data_ids = set()
    failed_seg_ids = set()
    for unique_id in unique_ids:
        # Try reading data
        try:
            np.load(path_to_data_dir / (unique_id + ".npy"), "r")
        except ValueError:
            failed_data_ids.add(unique_id)
        # Try reading seg
        try:
            np.load(path_to_data_dir / (unique_id + "_seg.npy"), "r")
        except ValueError:
            failed_seg_ids.add(unique_id)

    return failed_data_ids, failed_seg_ids


def try_fix_broken_npy(path_do_data_dir: Path, case_ids: set[str], fix_image: bool):
    """ 
    Receives broken case ids and tries to fix them by re-extracting the npz file (up to 5 times).

    :param case_ids: Set of case ids that are broken.
    :param path_do_data_dir: Path to the preprocessed directory containing the npys and npzs.
    :raises ValueError: If the npy file could not be unpacked after 5 tries. --
    """
    for case_id in case_ids:
        for i in range(5):
            try:
                key = "data" if fix_image else "seg"
                suffix = ".npy" if fix_image else "_seg.npy"
                read_npz = np.load(path_do_data_dir / (case_id + ".npz"), "r")[key]
                np.save(path_do_data_dir / (case_id + suffix), read_npz)
                # Try loading the just saved image.
                np.load(path_do_data_dir / (case_id + suffix), "r")
                break
            except ValueError:
                if i == 4:
                    raise ValueError(
                        f"Could not unpack {case_id + suffix} after 5 tries!"
                    )
                continue


def verify_or_stratify_npys(path_to_data_dir: str | Path) -> None:
    """
    This re-reads the npy files after unpacking. Should there be a loading issue with any, it will try to unpack this file again and overwrites the existing.
    If the new file does not get saved correctly 5 times, it will raise an error with the file name to the user. Does the same for images and segmentations.
    :param path_to_data_dir: Path to the preprocessed directory containing the npys and npzs.
    :raises ValueError: If the npy file could not be unpacked after 5 tries. --
      Otherwise an obscured error will be raised later during training (depending when the broken file is sampled)
    """
    path_to_data_dir = Path(path_to_data_dir)
    # Check for broken image and segmentation npys
    failed_data_ids, failed_seg_ids = find_broken_image_and_labels(path_to_data_dir)

    if len(failed_data_ids) != 0 or len(failed_seg_ids) != 0:
        warn(
            f"Found {len(failed_data_ids)} faulty data npys and {len(failed_seg_ids)}!\n"
            + f"Faulty images: {failed_data_ids}; Faulty segmentations: {failed_seg_ids})\n"
            + "Trying to fix them now."
        )
        # Try to fix the broken npys by reextracting the npz. If that fails, raise error
        try_fix_broken_npy(path_to_data_dir, failed_data_ids, fix_image=True)
        try_fix_broken_npy(path_to_data_dir, failed_seg_ids, fix_image=False)


def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False) -> None:
    try:
        a = np.load(npz_file)  # inexpensive, no compression is done here. This just reads metadata
        b = load_npz_file(npz_file[:-4] + "_key.npz")
        c = load_npz_file(npz_file[:-4] + "_categoryk.npz")
        key_as_float32 = b['key'].astype(np.float32)
        # a_seg = a['seg'] #读取进来会带一个1
        # a_data = a['data']
        if overwrite_existing or not isfile(npz_file[:-3] + "npy"):
            np.save(npz_file[:-3] + "npy", a['data'])
        if unpack_segmentation and (overwrite_existing or not isfile(npz_file[:-4] + "_seg.npy")):
            np.save(npz_file[:-4] + "_seg.npy", a['seg'])
        if (overwrite_existing or not isfile(npz_file[:-4] + "_key.npy")) and b is not None:
            key_save_name = npz_file[:-4] + "_key.h5"
            num_class, z, x, y = key_as_float32.shape
            key_as_float32_reshaped = key_as_float32.reshape(num_class*z, x*y)
            sparse_matrix = csr_matrix(key_as_float32_reshaped)
            with h5py.File(key_save_name, 'w') as f:
                # 存储稀疏矩阵的数据、行索引和列指针
                f.create_dataset('data', data=sparse_matrix.data)
                f.create_dataset('indices', data=sparse_matrix.indices)
                f.create_dataset('indptr', data=sparse_matrix.indptr)
                f.create_dataset('shape', data=sparse_matrix.shape)
                # 将原始形状作为属性存储
                f.attrs['original_shape'] = key_as_float32.shape
        if (overwrite_existing or not isfile(npz_file[:-4] + "_categoryk.npy")) and c is not None:
            np.save(npz_file[:-4] + "_categoryk.npy", c['key'])
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
        npz_files_filtered = [f for f in npz_files if 'key' not in f and 'category' not in f]
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