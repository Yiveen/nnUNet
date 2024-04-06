import multiprocessing
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, \
    determine_reader_writer_from_file_ending
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        split = key.split(',')
        return tuple([int(i) for i in split if len(i) > 0])


def save_summary_json(results: dict, output_file: str):
    """
    stupid json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        for k in results["metric_per_case"][i].keys():
            if 'total_distance' in k:
                results_converted["metric_per_case"][i]['total_distance'] = {results["metric_per_case"][i]['total_distance']}
            else:
                results_converted["metric_per_case"][i]['metrics'] = \
                    {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
                     for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    def convert_sets_to_lists(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets_to_lists(x) for x in obj]
        else:
            return obj

    results_converted = convert_sets_to_lists(results_converted)
    save_json(results_converted, output_file, sort_keys=True)


def load_summary_json(filename: str):
    results = load_json(filename)
    # convert keys in mean metrics
    results['mean'] = {key_to_label_or_region(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results["metric_per_case"])):
        results["metric_per_case"][i]['metrics'] = \
            {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    return results


def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: Optional[int] = None, key_name: str = None, biases: list = None) -> dict:
    path_parts = reference_file.split('/')
    # 找到 "gt_segmentations" 的索引并获取其后的文件名
    file_stem = os.path.join(*path_parts[:path_parts.index('gt_segmentations')])
    file_name_with_extension = path_parts[path_parts.index('gt_segmentations') + 1]
    # 从文件名中移除扩展名
    file_name, _ = os.path.splitext(file_name_with_extension)
    if key_name is not None:
        key_file_path = os.path.join("/" + file_stem, key_name, file_name[:-4] + "_key.npz")

    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    key_pred = None
    seg_pred = None
    if 'key' in prediction_file:
        key_pred, key_pred_dict = image_reader_writer.read_seg(prediction_file)
        key_ref = np.load(key_file_path)['key']
    else:
        seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)

    results = {}

    # spacing = seg_ref_dict['spacing']
    if seg_pred is not None:

        ignore_mask = seg_ref == ignore_label if ignore_label is not None else None
        results['reference_file'] = reference_file
        results['prediction_file'] = prediction_file
        results['metrics'] = {}
        for r in labels_or_regions:
            results['metrics'][r] = {}
            mask_ref = region_or_label_to_mask(seg_ref, r)
            mask_pred = region_or_label_to_mask(seg_pred, r)
            tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
            if tp + fp + fn == 0:
                results['metrics'][r]['Dice'] = np.nan
                results['metrics'][r]['IoU'] = np.nan
            else:
                results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
                results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
            results['metrics'][r]['FP'] = fp
            results['metrics'][r]['TP'] = tp
            results['metrics'][r]['FN'] = fn
            results['metrics'][r]['TN'] = tn
            results['metrics'][r]['n_pred'] = fp + tp
            results['metrics'][r]['n_ref'] = fn + tp

    if key_pred is not None:
        b, c, h, w = key_ref.shape
        center = np.zeros((b, len(biases), 3), dtype=np.float32)
        center_predict = np.zeros((b, len(biases), 3), dtype=np.float32)

        total_distance = 0

        for channel_idx in range(b):  # 使用 range(b)
            kernel = key_ref[channel_idx]
            output = key_pred[channel_idx]

            for i, bias in enumerate(biases):
                adjusted_kernel = kernel - bias - 0.7  # hard-coded
                adjusted_output = output - bias

                mask = (adjusted_kernel > 0) & (adjusted_kernel < 1)  # 不再使用non_zero
                if not np.any(mask):  # 如果mask全为False，则跳过
                    continue
                # 获取满足条件的元素的索引
                gt_indices = np.array(np.nonzero(mask))

                max_index = np.argmax(adjusted_kernel[mask])
                peak_coords_abs = gt_indices[:, max_index]
                center[channel_idx, i, :] = peak_coords_abs

                mask_output = (adjusted_output > 0) & (adjusted_output < 1)  # 不再使用non_zero
                if not np.any(mask_output):  # 如果mask全为False，则跳过
                    continue
                # 获取满足条件的元素的索引
                gt_indices = np.array(np.nonzero(mask_output))

                max_index_predict = np.argmax(adjusted_output[mask_output])
                peak_coords_abs_predict = gt_indices[:, max_index_predict]
                center_predict[channel_idx, i, :] = peak_coords_abs_predict
        non_zero_mask = np.logical_and(center != 0, center_predict != 0)
        non_zero_center = center[non_zero_mask]
        non_zero_center_predict = center_predict[non_zero_mask]

        if non_zero_center.size > 0:
            total_distance = np.linalg.norm(non_zero_center - non_zero_center_predict, axis=-1)
            total_distance = np.mean(total_distance)  # 取平均
        else:
            total_distance = 0.0  # 或者任何适当的默认值

        results['total_distance'] = total_distance

    return results


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]], #1
                              ignore_label: int = None, #None
                              num_processes: int = default_num_processes,
                              chill: bool = True,
                              configuration_manager: ConfigurationManager = None,
                              plans_manager: PlansManager =None,
                              stage: int = 1) -> dict:
    """
    output_file must end with .json; can be None
    """
    if plans_manager is not None:
        biases = plans_manager.plans['biases']
    if configuration_manager is not None:
        key_name = configuration_manager.data_identifier
    else:
        key_name = None
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_pred exist in folder_ref"
    files_ref = [join(folder_ref, i) for i in files_pred if i in files_ref]
    if stage == 2:
        files_ref = [item for item in files_ref for _ in range(2)]
    files_pred = [join(folder_pred, i) for i in files_pred]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_metrics(*i)
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred),
                     [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred), [key_name] * len(files_pred), [biases]* len(files_pred)))
        )

    # results = compute_metrics(files_ref[0], files_pred[0], image_reader_writer,regions_or_labels,ignore_label,key_name, biases)


    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for num, i in enumerate(results) if num / 2 == 0])
    total_dist = None
    if 'total_distance' in results[0].keys():
        total_dist = np.mean(i['total_distance'] for num, i in enumerate(results) if num / 2 == 1)

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    if total_dist is not None:
        result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean, 'total_dist': total_dist}
    else:
        result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result
    # print('DONE')


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False):
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                              num_processes, chill=chill)


def compute_metrics_on_folder_simple(folder_ref: str, folder_pred: str, labels: Union[Tuple[int, ...], List[int]],
                                     output_file: str = None,
                                     num_processes: int = default_num_processes,
                                     ignore_label: int = None,
                                     chill: bool = False):
    example_file = subfiles(folder_ref, join=True)[0]
    file_ending = os.path.splitext(example_file)[-1]
    rw = determine_reader_writer_from_file_ending(file_ending, example_file, allow_nonmatching_filename=True,
                                                  verbose=False)()
    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              labels, ignore_label=ignore_label, num_processes=num_processes, chill=chill)


def evaluate_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np, chill=args.chill)


def evaluate_simple_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-l', type=int, nargs='+', required=True,
                        help='list of labels')
    parser.add_argument('-il', type=int, required=False, default=None,
                        help='ignore label')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')

    args = parser.parse_args()
    compute_metrics_on_folder_simple(args.gt_folder, args.pred_folder, args.l, args.o, args.np, args.il, chill=args.chill)


if __name__ == '__main__':
    folder_ref = '/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/labelsTr'
    folder_pred = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation'
    output_file = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/summary.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([1, 2])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions, ignore_label,
                              num_processes)
