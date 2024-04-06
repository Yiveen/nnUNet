#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import multiprocessing
import shutil
from time import sleep
from typing import Union, Tuple

import nnunetv2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape, compute_new_spacing, resample_data_or_seg_to_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    create_lists_from_splitted_dataset_folder, get_filenames_of_train_images_and_targets
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json



from scipy import mgrid, exp, square, pi, dot
from numpy import zeros, ravel, uint8
import copy
import SimpleITK as sitk


class DefaultPreprocessor(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], aaa: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        data = np.copy(data)
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        #这里我们认为aaa文件一定有，所以这里不做更多的判断
        aaa = np.copy(aaa)

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]]) #plans_manager.transpose_forward : [0,1,2]
        aaa = aaa.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])

        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:] #到这里datat还是没有进行任何处理的data
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox, aaa = crop_to_nonzero(data, seg, aaa)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing # [1.0, 0.7871090173721313, 0.7871090173721313]
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        aaa = configuration_manager.resampling_fn_seg(aaa, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)

            aaa_data_json = '/home/yiwen/guidedresearch/nnUNet/nnUNet_raw/Dataset027_Aorta' + '/AAA.json'  # Hard-code here
            aaa_data_json = load_json(aaa_data_json)
            aaa_label_manager = plans_manager.get_label_manager(aaa_data_json)
            collect_for_this_aaa = aaa_label_manager.foreground_regions if aaa_label_manager.has_regions \
                else aaa_label_manager.foreground_labels
            properties['aaa_locations'] = self._sample_foreground_locations(aaa, collect_for_this_aaa,
                                                                              verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
            aaa = aaa.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
            aaa = aaa.astype(np.int8)
        return data, seg, aaa

    def run_case_original(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str]):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properties = rw.read_images(image_files)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        data, seg, aaa = self.run_case_npy(data, seg, data_properties, plans_manager, configuration_manager,
                                      dataset_json)
        return data, seg, data_properties, aaa

    def run_case(self, image_files: List[str], seg_file: Union[str, None], key_file: Union[str, None], aaa_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str]):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        dataset_json_path = dataset_json
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properties = rw.read_images(image_files) #在内部处理中会vstack一下，然后会产生一个新的维度。data [1, 326,512,512] data_properties就是simpleitk的属性
        original_data_shape = data.shape

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file) #seg文件的大小
        else:
            seg = None
        # original_seg = seg.astype(np.int8)
        if aaa_file is not None:
            aaa, _ = rw.read_seg(aaa_file) #seg文件的大小
        else:
            aaa = None

        data, seg, aaa = self.run_case_npy(data, seg, aaa, data_properties, plans_manager, configuration_manager,
                                      dataset_json)

        original_image = []
        seg_img = sitk.ReadImage(seg_file)
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelType(sitk.sitkBall)
        dilate_filter.SetKernelRadius(1)  # 设置膨胀核半径
        seg_cta_dilated = dilate_filter.Execute(seg_img)


        # seg_cta_dilated_npy_image = sitk.GetArrayFromImage(seg_cta_dilated)
        # original_image.append(seg_cta_dilated_npy_image)
        # original_seg = np.vstack(original_image)
        # original_seg = np.expand_dims(original_seg, axis=0)

        original_spacing = [data_properties['spacing'][i] for i in plans_manager.transpose_forward]
        down_sample_spacing = compute_new_spacing(original_data_shape[1:], original_spacing,
                                                  plans_manager.plans['configurations']['3d_fullres']['patch_size'])
        new_s = plans_manager.plans['configurations']['3d_fullres']['patch_size']
        new_s_zyx = [new_s[2], new_s[1], new_s[0]]
        seg_down_sampled = self.resample_volume(seg_cta_dilated, down_sample_spacing[[2,1,0]], new_s_zyx, sitk.sitkNearestNeighbor)
        seg_down_sampled = sitk.GetArrayFromImage(seg_down_sampled)
        seg_down_sampled = np.expand_dims(seg_down_sampled, axis=0)


        # seg_down_sampled = resample_data_or_seg_to_shape(original_seg, plans_manager.plans['configurations']['3d_fullres']['patch_size'],
        #                                                  original_spacing, down_sample_spacing, is_seg=True) #check一下如何seg的

        if configuration_manager.data_identifier == 'nnUNetPlans_3d_fullres' and key_file is not None:
            with open(key_file, 'r') as key_file_path:
                key_json = json.load(key_file_path)
            key_points_ras = np.array(key_json['ras_points'])
            direction_matrix = np.array(key_json['IJKtoRASDirectionMatrix'])
            origin = np.array(key_json['ImageOrigin'])

            key_points_ras_zyx = key_points_ras[:, [2, 1, 0]]

            direction_diag = np.diag(direction_matrix)
            direction_diag_zyx = direction_diag[[2, 1, 0]]
            direction_matrix_zyx = np.diag(direction_diag_zyx)
            origin_zyx = origin[[2, 1, 0]]
            target_spacing = configuration_manager.spacing

            key_points = self.transform_points(key_points_ras_zyx, target_spacing, origin_zyx, direction_matrix_zyx)
            k_size = plans_manager.plans['k_size']
            sigma = plans_manager.plans['sigma']
            biases = plans_manager.plans['biases']

            volume1 = np.zeros(data.shape)
            key_initial = np.zeros((4,) + data.shape[1:])
            key = np.zeros((4,) + original_data_shape[1:])

            # key = self.apply_gaussian_to_1:keypoints(copy.deepcopy(volume1), key_points, k_size, sigma, biases) #选取特定元素
            # for channel in range(4):
            #     key_initial[channel] = self.make_gauss(self.normalize_point(key_points[channel], data.shape[1:]), data.shape[1:], sigma=40)
            #     expanded_arr = np.expand_dims(key_initial[channel], axis=0)
            #     resampled_key = configuration_manager.resampling_fn_data(expanded_arr, original_data_shape[1:], target_spacing, original_spacing)
            #     key[channel] = resampled_key.squeeze(0)
            for channel in range(4):
                key_initial[channel] = self.make_gauss(self.normalize_point(key_points[channel], data.shape[1:]), data.shape[1:], sigma=40)
                # expanded_arr = np.expand_dims(key_initial[channel], axis=0)
                # resampled_key = configuration_manager.resampling_fn_data(expanded_arr, original_data_shape[1:], target_spacing, original_spacing)
                # key[channel] = resampled_key.squeeze(0)
            category_k = self.apply_category_to_keypoints(copy.deepcopy(volume1), key_points, 10, sigma, biases)

            key_data_json = '/home/yiwen/guidedresearch/nnUNet/nnUNet_raw/Dataset027_Aorta' + '/key_points.json' #Hard-code here
            key_data_json = load_json(key_data_json)
            key_label_manager = plans_manager.get_label_manager(key_data_json)
            collect_for_this = key_label_manager.foreground_regions if key_label_manager.has_regions \
                else key_label_manager.foreground_labels
            data_properties['category_k_locations'] = self._sample_foreground_locations(category_k, collect_for_this,
                                                                              verbose=self.verbose)

            key_points_downsampled = self.transform_points(key_points_ras_zyx, down_sample_spacing, origin_zyx, direction_matrix_zyx)
            volume2 = np.zeros(seg_down_sampled.shape)

            key_down_sampled = np.zeros((4,) + seg_down_sampled.shape[1:])
            # key = self.apply_gaussian_to_1:keypoints(copy.deepcopy(volume1), key_points, k_size, sigma, biases) #选取特定元素
            for channel in range(4):
                # Now you can use volume2.shape
                key_down_sampled[channel] = self.make_gauss(
                    self.normalize_point(key_points_downsampled[channel], seg_down_sampled.shape[1:]),
                    seg_down_sampled.shape[1:],
                    sigma=10
                )
            # key_down_sampled = self.apply_gaussian_to_keypoints(copy.deepcopy(volume2), key_points_downsampled, 14, sigma, biases)  # 选取特定元素
            category_k_down_sampled = self.apply_category_to_keypoints(copy.deepcopy(volume2), key_points_downsampled, 14, sigma, biases) #TODO:需要确定下采样的高斯核大小
            data_properties['category_k_locations_down_sampled'] = self._sample_foreground_locations(category_k_down_sampled, collect_for_this,
                                                                                        verbose=self.verbose)

            # volume2 = np.zeros(data.shape)
            # key2 = self.apply_gaussian_to_keypoints(volume2, key_points[[1,4,5],:], k_size, sigma, biases)
            # key = np.concatenate((key1, key2), axis=0)
        else:
            key = None


        return data, seg, data_properties, key_initial, category_k, aaa, seg_down_sampled, key_down_sampled, category_k_down_sampled

    def normalize_point(self, point, shape):
        normalized_point = [p/(s + 1e-24) for p, s in zip(point, shape)]
        return np.array(normalized_point)

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str, key_file:str, aaa_file:str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        print('now process', output_filename_truncated)
        list = ['/home/yiwen/guidedresearch/nnUNet/nnUNet_preprocessed/Dataset027_Aorta/nnUNetPlans_3d_fullres/arota_009', '/home/yiwen/guidedresearch/nnUNet/nnUNet_preprocessed/Dataset027_Aorta/nnUNetPlans_3d_fullres/arota_017', '/home/yiwen/guidedresearch/nnUNet/nnUNet_preprocessed/Dataset027_Aorta/nnUNetPlans_3d_fullres/arota_018', '/home/yiwen/guidedresearch/nnUNet/nnUNet_preprocessed/Dataset027_Aorta/nnUNetPlans_3d_fullres/arota_019','/home/yiwen/guidedresearch/nnUNet/nnUNet_preprocessed/Dataset027_Aorta/nnUNetPlans_3d_fullres/arota_024']
        # if output_filename_truncated in list:
        # if output_filename_truncated == '/home/yiwen/guidedresearch/nnUNet/nnUNet_preprocessed/Dataset027_Aorta/nnUNetPlans_3d_fullres/arota_012':
        data, seg, properties, key, category_k, aaa, seg_down_sampled, key_down_sampled, category_k_down_sampled = self.run_case(image_files, seg_file, key_file, aaa_file, plans_manager, configuration_manager, dataset_json)
        # print('dtypes', data.dtype, seg.dtype)
        #TODO:在experiment里面规划key file的路径存储，然后在这里读取，转换成目标大小，然后生成高斯核
        # 还需要check怎么样分成patch
        print('now save', output_filename_truncated)
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        np.savez_compressed(output_filename_truncated + '_aaa.npz', seg=aaa)
        np.savez_compressed(output_filename_truncated + '_key.npz', key=key)
        # np.savez_compressed(output_filename_truncated + '_categoryk.npz', key=category_k)
        np.savez_compressed(output_filename_truncated + 'seg_down_sampled.npz', seg=seg_down_sampled)
        np.savez_compressed(output_filename_truncated + 'key_down_sampled_key.npz', key=key_down_sampled)
        # np.savez_compressed(output_filename_truncated + 'category_k_down_sampled_key.npz', key=category_k_down_sampled)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
        return class_locs

    def _normalize(self, data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'nnunetv2.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError(f'Unable to locate class \'{scheme}\' for normalization')
            normalizer = normalizer_class(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                                          intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], seg[0])#只是对image进行归一化，没有对seg进行归一化
        return data

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        """
        data identifier = configuration name in plans. EZ.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
        if self.verbose:
            print(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)

        # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
        # output_filenames_truncated = [join(output_directory, i) for i in identifiers]

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for k in dataset.keys():
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'], dataset[k]['key'], dataset[k]['aaa'],
                                           plans_manager, configuration_manager,
                                           dataset_json),)))
            remaining = list(range(len(dataset)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        # this function will be called at the end of self.run_case. Can be used to change the segmentation
        # after resampling. Useful for experimenting with sparse annotations: I can introduce sparsity after resampling
        # and don't have to create a new dataset each time I modify my experiments
        return seg

    def gen_gaussian_kernel_3d(self, k_size, sigma, bias):
        center = k_size // 2
        x, y, z = np.mgrid[0 - center:k_size - center, 0 - center:k_size - center, 0 - center:k_size - center]
        distance_squared = square(x) + square(y) + square(z)  # 计算距离中心的平方距离

        # 在球内应用高斯函数，在球外设置为0
        g = np.where(distance_squared <= square(center),
                     50 / (2 * pi * sigma) * exp(-distance_squared / (2 * square(sigma))) + bias,
                     0)

        return g
    # def gen_category_kernel_3d(self, k_size, sigma, bias):
    #     category_kernel = np.zeros((k_size, k_size, k_size), dtype=np.int16)
    #     category_kernel = category_kernel + bias
    #     return category_kernel

    def gen_category_kernel_3d_sphere(self, k_size, sigma, bias):
        center = k_size // 2
        radius = k_size // 2
        category_sphere = np.zeros((k_size, k_size, k_size), dtype=np.int16)  # 初始化全零数组

        # 生成三维坐标网格
        x, y, z = np.ogrid[-center:k_size - center, -center:k_size - center, -center:k_size - center]
        # 计算每个点到中心的欧几里得距离的平方
        distance_squared = x ** 2 + y ** 2 + z ** 2

        # 将球体内的点设置为1
        mask = distance_squared <= radius ** 2
        category_sphere[mask] = bias

        return category_sphere

    def gen_gaussian_sphere(self, k_size, sigma, bias):
        # 确定球心的位置
        center = k_size // 2
        # 生成三维坐标网格
        x, y, z = np.mgrid[0 - center:k_size - center, 0 - center:k_size - center, 0 - center:k_size - center]
        # 计算每个点到球心的欧几里得距离
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        # 应用高斯函数
        g = (50 / (2 * pi * sigma ** 3)) * exp(-(r ** 2) / (2 * sigma ** 2))

        return g

    def transform_points(self, points: np.ndarray, spacing: np.ndarray, origin: np.ndarray, direction_matrix: np.ndarray):
        scale_matrix = np.diag(spacing)
        transform_matrix = np.dot(direction_matrix, scale_matrix)
        ijk_points = []
        for point in points:
            ijk_point = np.dot(np.linalg.inv(transform_matrix),
                               np.array(point) - np.array(origin))
            ijk_points.append(ijk_point)
        return np.array(ijk_points)

    def apply_category_to_keypoints(self, volume, keypoints, k_size, sigma, biases):
        """
        在3D体数据中的特定关键点位置上应用高斯核。

        :param volume: 3D numpy array, 输入的体数据。
        :param keypoints: List of tuples, 关键点的坐标列表，每个元组表示一个点的(z, y, x)坐标。
        :param k_size: int, 高斯核的大小。
        :param sigma: float, 高斯核的标准差。
        """
        # 生成高斯核
        # gaussian_kernel = self.gen_gaussian_kernel_3d(k_size, sigma, biases)
        kernel_center = k_size // 2

        for point, bias in zip(keypoints, biases):
            gaussian_kernel = self.gen_category_kernel_3d_sphere(k_size, sigma, bias)  # 生成带有bias的高斯核

            z, y, x = point
            z = self.int_number(z)
            y = self.int_number(y)
            x = self.int_number(x)


            # 计算高斯核应用区域的边界，确保不超出体数据范围
            z_min, z_max = max(0, z - kernel_center), min(volume.shape[1], z + kernel_center)
            y_min, y_max = max(0, y - kernel_center), min(volume.shape[2], y + kernel_center)
            x_min, x_max = max(0, x - kernel_center), min(volume.shape[3], x + kernel_center)

            # 应用高斯核，同时更新最大值数组
            # 计算当前高斯核在体数据中的位置
            kernel_slice = gaussian_kernel[
                           kernel_center - (z - z_min):kernel_center + (z_max - z),
                           kernel_center - (y - y_min):kernel_center + (y_max - y),
                           kernel_center - (x - x_min):kernel_center + (x_max - x)
                           ]

            # 更新体数据：仅当高斯核的值大于体数据中的值时才更新
            volume[0, z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                volume[0, z_min:z_max, y_min:y_max, x_min:x_max], kernel_slice
            )
        return volume


    def apply_gaussian_to_keypoints(self, volume, keypoints, k_size, sigma, biases):
        """
        在3D体数据中的特定关键点位置上应用高斯核。

        :param volume: 3D numpy array, 输入的体数据。
        :param keypoints: List of tuples, 关键点的坐标列表，每个元组表示一个点的(z, y, x)坐标。
        :param k_size: int, 高斯核的大小。
        :param sigma: float, 高斯核的标准差。
        """
        # 生成高斯核
        # gaussian_kernel = self.gen_gaussian_kernel_3d(k_size, sigma, biases)
        kernel_center = k_size // 2
        overlap_count = np.zeros_like(volume, dtype=np.float32)  # 用于记录每个体素的高斯核重叠次数

        # # 遍历所有关键点
        # for z, x, y in keypoints:
        #     # 计算高斯核应用区域的边界
        #     z_min, z_max = max(0, z - kernel_center), min(volume.shape[1], z + kernel_center + 1)
        #     y_min, y_max = max(0, y - kernel_center), min(volume.shape[2], y + kernel_center + 1)
        #     x_min, x_max = max(0, x - kernel_center), min(volume.shape[3], x + kernel_center + 1)
        #
        #     # 计算高斯核在体数据中的对应区域,在计算左下和右上角的坐标
        #     # 在体数据上叠加高斯核，重叠区域取最大值
        #     volume[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
        #         volume[z_min:z_max, y_min:y_max, x_min:x_max],
        #         gaussian_kernel[
        #         kernel_center - (z - z_min):kernel_center + (z_max - z),
        #         kernel_center - (y - y_min):kernel_center + (y_max - y),
        #         kernel_center - (x - x_min):kernel_center + (x_max - x)
        #         ]
        #     )

        for point, bias in zip(keypoints, biases):
            gaussian_kernel = self.gen_gaussian_kernel_3d(k_size, sigma, bias)  # 生成带有bias的高斯核

            z, y, x = point
            z = self.int_number(z)
            y = self.int_number(y)
            x = self.int_number(x)


            # 计算高斯核应用区域的边界，确保不超出体数据范围
            z_min, z_max = max(0, z - kernel_center), min(volume.shape[1], z + kernel_center)
            y_min, y_max = max(0, y - kernel_center), min(volume.shape[2], y + kernel_center)
            x_min, x_max = max(0, x - kernel_center), min(volume.shape[3], x + kernel_center)

            # 应用高斯核，同时更新最大值数组
            # 计算当前高斯核在体数据中的位置
            kernel_slice = gaussian_kernel[
                           kernel_center - (z - z_min):kernel_center + (z_max - z),
                           kernel_center - (y - y_min):kernel_center + (y_max - y),
                           kernel_center - (x - x_min):kernel_center + (x_max - x)
                           ]

            # 更新体数据：仅当高斯核的值大于体数据中的值时才更新
            volume[0, z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                volume[0, z_min:z_max, y_min:y_max, x_min:x_max], kernel_slice
            )

            # # 计算高斯核的切片索引
            # z_kernel_min = max(kernel_center - (z - z_min), 0)
            # z_kernel_max = min(kernel_center + (z_max - z), gaussian_kernel.shape[0])
            # x_kernel_min = max(kernel_center - (x - x_min), 0)
            # x_kernel_max = min(kernel_center + (x_max - x), gaussian_kernel.shape[1])
            # y_kernel_min = max(kernel_center - (y - y_min), 0)
            # y_kernel_max = min(kernel_center + (y_max - y), gaussian_kernel.shape[2])
            #
            # # 更新体数据和重叠计数
            # volume[0, z_min:z_max, x_min:x_max, y_min:y_max] += gaussian_kernel[
            #                                                     z_kernel_min:z_kernel_max,
            #                                                     x_kernel_min:x_kernel_max,
            #                                                     y_kernel_min:y_kernel_max
            #                                                     ]
            # overlap_count[0, z_min:z_max, x_min:x_max, y_min:y_max] += 1

            # 更新体数据和重叠计数
            # volume[0, z_min:z_max, y_min:y_max, x_min:x_max] += gaussian_kernel[
            #                                                     kernel_center - (z - z_min):kernel_center + (z_max - z),
            #                                                     kernel_center - (y - y_min):kernel_center + (y_max - y),
            #                                                     kernel_center - (x - x_min):kernel_center + (x_max - x)
            #                                                     ]
            # overlap_count[0, z_min:z_max, y_min:y_max, x_min:x_max] += 1

            # 处理重叠区域：取平均值
        # volume /= np.maximum(overlap_count, 1)  # 避免除以零
        return volume

    def int_number(self, number):
        return np.floor(number).astype(int)

    def resample_volume(self, sitk_volume, new_spacing, new_size, interpolator):

        # Set up the resampling transform and parameters
        resampling_transform = sitk.Transform()  # Identity transform by default
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetReferenceImage(sitk_volume)
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetSize(new_size)
        resample_filter.SetTransform(resampling_transform)
        resample_filter.SetInterpolator(interpolator)
        resample_filter.SetOutputOrigin(sitk_volume.GetOrigin())
        resample_filter.SetOutputDirection(sitk_volume.GetDirection())
        resample_filter.SetDefaultPixelValue(0)  # Default value for pixels outside the original volume

        # Perform the resampling
        resampled_volume = resample_filter.Execute(sitk_volume)

        return resampled_volume

    import numpy as np
    from functools import reduce
    from operator import mul

    def normalized_linspace(self, length):
        """Generate a vector with values ranging from 0 to 1 in NumPy."""
        # 生成的向量的首尾元素分别比0和1稍小一点，以对应单元格的中心
        return np.linspace(1.0 / length / 2, 1 - 1.0 / length / 2, num=length)

    def make_gauss(self, means, size, sigma, normalize=False):
        """Draw Gaussians in NumPy."""
        dim_range = range(-1, -(len(size) + 1), -1)
        coords_list = [self.normalized_linspace(s) for s in size]

        # Calculate distances squared
        dists = [(x - mean) ** 2 for x, mean in
                 zip(coords_list, np.split(means, indices_or_sections=means.shape[-1], axis=-1))]

        # Calculate constants for Gaussian formula
        stddevs = [2 * sigma / s for s in size]
        ks = [-0.5 * (1 / stddev) ** 2 for stddev in stddevs]

        # Calculate exponent part of the Gaussian
        exps = [np.exp(dist * k) for k, dist in zip(ks, dists)]

        # Combine dimensions of the Gaussian
        gauss = exps[0]
        for exp in exps[1:]:
            gauss = np.expand_dims(gauss, axis=-1)  # Add a new axis to the last dimension of gauss
            exp = np.expand_dims(exp, axis=0)  # Add a new axis to the first dimension of exp
            gauss = gauss * exp  # Multiply expanded arrays

        # Move axes back to their original order if necessary

        if not normalize:
            return gauss

        # Normalize the Gaussians
        val_sum = gauss.sum(axis=tuple(dim_range), keepdims=True) + 1e-24
        return gauss / val_sum


def example_test_case_preprocessing():
    # (paths to files may need adaptations)
    plans_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/nnUNetPlans.json'
    dataset_json_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/dataset.json'
    input_images = ['/home/isensee/drives/e132-rohdaten/nnUNetv2/Dataset219_AMOS2022_postChallenge_task2/imagesTr/amos_0600_0000.nii.gz', ]  # if you only have one channel, you still need a list: ['case000_0000.nii.gz']

    configuration = '3d_fullres'
    pp = DefaultPreprocessor()

    # _ because this position would be the segmentation if seg_file was not None (training case)
    # even if you have the segmentation, don't put the file there! You should always evaluate in the original
    # resolution. What comes out of the preprocessor might have been resampled to some other image resolution (as
    # specified by plans)
    plans_manager = PlansManager(plans_file)
    data, _, properties = pp.run_case(input_images, seg_file=None, plans_manager=plans_manager,
                                      configuration_manager=plans_manager.get_configuration(configuration),
                                      dataset_json=dataset_json_file)

    # voila. Now plug data into your prediction function of choice. We of course recommend nnU-Net's default (TODO)
    return data


if __name__ == '__main__':
    example_test_case_preprocessing()
    # pp = DefaultPreprocessor()
    # pp.run(2, '2d', 'nnUNetPlans', 8)

    ###########################################################################################################
    # how to process a test cases? This is an example:
    # example_test_case_preprocessing()
