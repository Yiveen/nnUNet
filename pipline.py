import torch
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
import inspect
import multiprocessing
import os
import re
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
import pickle

from typing import Union

from post_process_class import Postprocess1, Postprocess2, Postprocess3
# from pipline_utilities import get_network_from_plans
# from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
# from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
# from nnunetv2.utilities.helpers import empty_cache, dummy_context
# from nnunetv2.inference.sliding_window_prediction import compute_gaussian
# from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
# from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
#     compute_steps_for_sliding_window
# from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape, compute_new_spacing, resample_data_or_seg_to_shape, resample_data_or_seg_to_shape
# from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save, export_prediction_key
# from tqdm import tqdm
# from acvl_utils.cropping_and_padding.padding import pad_nd_image
# from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json


class Predictor:
    def __init__(self, category, input_data_path, plans, dataset_json, pretrained_path, save_path):
        '''
        category : 1代表seg类型，2代表key类型
        save_path 不用给到文件类型 .nii等会自动补上

        '''
        self.category = category
        self.input_data_path = input_data_path
        self.device = torch.device('cuda')
        self.pretrain_path = pretrained_path
        self.plans_manager = PlansManager(plans)
        configuration = '3d_fullres'
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        self.dataset_json = dataset_json

        self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                               self.dataset_json)
        self.enable_deep_supervision = True

        self.network = self.build_network_architecture(
                    self.configuration_manager.network_arch_class_name,
                    self.configuration_manager.network_arch_init_kwargs,
                    self.configuration_manager.network_arch_init_kwargs_req_import,
                    self.num_input_channels,
                    self.label_manager.num_segmentation_heads,
                    self.enable_deep_supervision)

        self.save_path = save_path
        self.use_gaussian = True
        self.allow_tqdm = True
        self.load_pretrain()

        self.result = self.perform_actual_validation()

    @ staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
    def load_pretrain(self):
        saved_model = torch.load(self.pretrain_path)
        pretrained_dict = saved_model['network_weights']

        skip_strings_in_pretrained = [
            '.seg_layers.',
        ]

        mod = self.network
        model_dict = mod.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys() and all(
                [i not in k for i in skip_strings_in_pretrained])}
        model_dict.update(pretrained_dict)

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        mod = self.network
        mod.decoder.deep_supervision = enabled

    def load_data(self):
        data, data_properties = SimpleITKIO.read_images(self.input_data_path)
        original_data_shape = data.shape
        img = sitk.ReadImage(self.input_data_path)
        if self.category == 2:
            dilate_filter = sitk.BinaryDilateImageFilter()
            dilate_filter.SetKernelType(sitk.sitkBall)
            dilate_filter.SetKernelRadius(1)  # 设置膨胀核半径
            seg_cta_dilated = dilate_filter.Execute(img)
            original_spacing = [data_properties['spacing'][i] for i in self.plans_manager.transpose_forward]
            down_sample_spacing = compute_new_spacing(original_data_shape[1:], original_spacing,
                                                      self.plans_manager.plans['configurations']['3d_fullres']['patch_size'])
            new_s = self.plans_manager.plans['configurations']['3d_fullres']['patch_size']
            new_s_zyx = [new_s[2], new_s[1], new_s[0]]
            final = self.resample_volume(seg_cta_dilated, down_sample_spacing[[2, 1, 0]], new_s_zyx,
                                                    sitk.sitkNearestNeighbor)
        else:
            final = img
        img_array = sitk.GetArrayFromImage(final)
        img_array = np.expand_dims(img_array, axis=0) # here z y x
        if self.category == 1:
            return self.run_case_npy(img_array,properties=data_properties), data_properties
        else:
            return self.run_case_npy(seg=img_array, properties=data_properties), data_properties

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None],
                     properties: dict):
        # let's not mess up the inputs!
        original_spacing = [properties['spacing'][i] for i in self.plans_manager.transpose_forward]
        target_spacing = self.configuration_manager.spacing  # this should already be transposed
        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing  # [1.0, 0.7871090173721313, 0.7871090173721313]
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
        if self.category == 1:
            data = np.copy(data)
            data = data.transpose([0, *[i + 1 for i in self.plans_manager.transpose_forward]])
            data, seg, bbox, aaa = crop_to_nonzero(data)
            data = self.configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
            return data
        else:
            seg = np.copy(seg)
            seg = seg.transpose(
                [0, *[i + 1 for i in self.plans_manager.transpose_forward]])  # plans_manager.transpose_forward : [0,1,2]

            data, seg, bbox, aaa = crop_to_nonzero(seg=seg)
            seg = self.configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
            if np.max(seg) > 127:
                seg = seg.astype(np.int16)
            else:
                seg = seg.astype(np.int8)
            return seg

    def perform_actual_validation(self):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        data, data_properties = self.load_data()
        data = data.to('cuda').half()

        if self.category == 1:
            prediction = self.predict_sliding_window_return_logits(data)
            prediction = prediction.cpu()
            result = export_prediction_from_logits(prediction, data_properties, self.configuration_manager, self.plans_manager,
                 self.dataset_json, self.save_path, False)
        else:
            prediction_key = self.predict_sliding_window_return_logits(data)
            prediction_key = prediction_key.cpu()
            result = export_prediction_key(prediction_key, data_properties, self.configuration_manager, self.plans_manager,
                 self.dataset_json, self.save_path, False)
        return result

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[
                Union[np.ndarray, torch.Tensor], Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                # preallocate results and num_predictions
                results_device = self.device
                try:
                    data = data.to(self.device)
                    predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                                   # (2,298,385,385)
                                                   dtype=torch.half,
                                                   device=results_device)
                    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                                device=results_device)  # (298,385,385)

                    predicted_keys = torch.zeros((4, *data.shape[1:]),  # hard-code here
                                                 dtype=torch.half,
                                                 device=results_device)
                    if self.use_gaussian:
                        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                                    value_scaling_factor=1000,
                                                    device=results_device)
                except RuntimeError:
                    # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                    results_device = torch.device('cpu')
                    data = data.to(results_device)
                    predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                                   dtype=torch.half,
                                                   device=results_device)
                    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                                device=results_device)
                    if self.use_gaussian:
                        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                                    value_scaling_factor=1000,
                                                    device=results_device)
                finally:
                    empty_cache(self.device)

                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    workon = data[sl][
                        None]  # 即在原有的一维数组前面增加了一个新的轴 在 NumPy 中，None 用于此目的与使用 np.newaxis 相同。这通常用于增加数组的维数，使得一维数组变成二维的行向量或列向量，或者在更高维的数组中添加更多维度
                    workon = workon.to(self.device, non_blocking=False)

                    if self.category == 1:
                        prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                    else:
                        prediction, prediction_key = self._internal_maybe_mirror_and_predict(workon)
                        prediction = prediction[0].to(results_device)
                        prediction_key = prediction_key[0].to(results_device)
                        predicted_keys[sl] += (prediction_key * gaussian if self.use_gaussian else prediction_key)
                    predicted_logits[sl] += (prediction * gaussian if self.use_gaussian else prediction)
                    n_predictions[sl[1:]] += (gaussian if self.use_gaussian else 1)

                if self.category == 1:
                    predicted_logits /= n_predictions
                else:
                    predicted_keys /= n_predictions
        empty_cache(self.device)
        if self.category == 1:
            return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]
        else:
            return predicted_keys[
                tuple([slice(None), *slicer_revert_padding[1:]])]
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        key_prediction = None
        if self.category == 1:
            prediction = self.network(x)
        else:
            prediction, key_prediction = self.network(x)
        if self.category == 1:
            return prediction
        else:
            return prediction, key_prediction
    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     0.5)

            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     0.5)
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

class VesselSegmentation:
    def __init__(self, input_cta_path, plans, dataset_json, pretrained_path,
                 save_path):
        self.PredictorVessel = Predictor(1, input_cta_path, plans, dataset_json, pretrained_path,
                 save_path)
        self.result = None
    def predict(self):
        self.result = self.PredictorVessel.result

class AAASegmentation:
    def __init__(self, input_cta_path, plans, dataset_json, pretrained_path,
                 save_path):
        self.PredictorAAA = Predictor(1, input_cta_path, plans, dataset_json, pretrained_path,
                                    save_path)
        self.result = None
    def predict(self):
        self.result = self.PredictorAAA.result

class KeyPointRegression:
    def __init__(self, input_cta_path, plans, dataset_json, pretrained_path,
                 save_path):
        self.PredictorKey = Predictor(2, input_cta_path, plans, dataset_json, pretrained_path,
                                      save_path)
        self.result = None
    def predict(self):
        self.result = self.PredictorKey.result


# class CriterionDist:
#     def __init__(self, ):



if __name__ == '__main__':
    # plans = load_json(r'E:\GuidedResearchProject\nnUNet\nnUNet_preprocessed\Dataset027_Aorta\nnUNetPlans.json')
    # dataset_json = load_json(r'E:\GuidedResearchProject\nnUNet\nnUNet_preprocessed\Dataset027_Aorta\dataset.json')

    base_path = r'E:\results'

    # Vessel process
    # raw_vessel_path = os.path.join(base_path, 'vessel', 'raw')
    # vessel_processed_save_path = os.path.join(base_path, 'vessel', 'processed')
    # Vesselprocess = Postprocess1(raw_vessel_path, vessel_processed_save_path)
    # for file in os.listdir(raw_vessel_path):
    #     vessel_result = Vesselprocess.run(file)
    #     Vesselprocess.save()

    # AAA process
    # raw_AAA_path = os.path.join(base_path, 'AAA', 'raw')
    # AAA_processed_save_path = os.path.join(base_path, 'AAA', 'processed')
    # AAAprocess = Postprocess1(raw_AAA_path, AAA_processed_save_path)
    # AAA_position = []
    # for file in os.listdir(raw_AAA_path):
    #     AAA_result = AAAprocess.run(file)
    #     AAA_position.append(AAAprocess.get_center())
    #
    # AAA_position = np.array(AAA_position)

    # # Key proecess
    raw_key_path = os.path.join(base_path, 'key', 'raw')
    key_ijk_points = {}
    properties_dict = {}
    dist = []
    mse = []
    mae = []
    group_id = [1,4,6,9,17,18,19,24,26,28,30,35,36,37]
    json_base = r'E:\key_points\Patient_0\Patient_0.json'
    Keyprocess = Postprocess3(raw_key_path, json_base)

    file_grouped = {}
    for id in group_id:
        grouped_name = []
        for file in os.listdir(raw_key_path):
            if f"arota_{id:03}" in file:
                grouped_name.append(file)
        file_grouped[id] = grouped_name

    for id, file in file_grouped.items():
        AAA_result = Keyprocess.run(file)
        # key_ijk_points[f"arota_{id:03}"] = np.array(Keyprocess.key_points) #or
        properties_dict[f"arota_{id:03}"] = {'spacing': Keyprocess.spacing_zxy, 'origin': Keyprocess.origin_zxy}
        key_ijk_points[f"arota_{id:03}"] = np.array(Keyprocess.gt_points)
        dist.append(Keyprocess.dist)
        mse.append(Keyprocess.mse)
        mae.append(Keyprocess.mae)
        AAA_position = None
    dist = np.mean(np.array(dist))
    mse = np.mean(np.array(mse))
    mae = np.mean(np.array(mae))
    print('dist',dist)
    print('mse', mse)
    print('mae', mae)

    data_to_save = [AAA_position, key_ijk_points, properties_dict]
    # 保存这个列表为 .npy 文件
    with open(os.path.join(base_path, 'processed.npy'), 'wb') as file:
        pickle.dump(data_to_save, file)
    # Final process, must after run Matlab

