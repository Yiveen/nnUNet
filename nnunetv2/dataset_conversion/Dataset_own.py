import os
import shutil
from pathlib import Path
import nibabel as nib

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json,generate_key_json,generate_AAA_json
from nnunetv2.paths import nnUNet_raw


def make_out_dirs(dataset_id: int, task_name="Aorta"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"
    out_key_dir = out_dir / "keypoints"
    out_AAA_dir = out_dir / "AAA"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)
    os.makedirs(out_key_dir, exist_ok=True)
    os.makedirs(out_AAA_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir, out_key_dir, out_AAA_dir


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path, test_list: list, out_key_dir: Path, out_AAA_dir: Path):
    """Copy files from the ACDC dataset to the nnUNet dataset folder. Returns the number of training cases."""
    patients_train = sorted([f for f in (src_data_folder).iterdir() if (f.is_dir() and str(f) not in test_list and 'points' not in str(f))])
    # for item in src_data_folder.iterdir() :
    #     if (str(item) in test_list):
    #         print(True)
    patients_test = test_list

    num_training_cases = 0
    # Copy training files and corresponding labels.
    for patient_dir in patients_train:
        print(patient_dir)
        for file in patient_dir.iterdir(): # patient_dir 是一个Path类型的数据，然后Path类型的数据才可以.iterdir()
            if file.suffix == ".nii" and "label" not in file.name and "CTA_2" in file.name:
                # The stem is 'patient.nii', and the suffix is '.gz'.
                # We split the stem and append _0000 to the patient part.

                # Load the .nii file
                nii_file = nib.load(str(patient_dir) + "/" + file.name)
                # Save the file as .nii.gz
                nib.save(nii_file, train_dir / f"arota_{str(patient_dir).split('_')[1].zfill(3)}_0000.nii.gz")

                original_json_file = str(src_data_folder) + '/key_points/' + str(patient_dir.name) + '/' + str(patient_dir.name) + '.json'
                shutil.copy(original_json_file, out_key_dir / f"arota_{str(patient_dir).split('_')[1].zfill(3)}.json")

                original_AAA_file = str(src_data_folder) + '/AAA/' + str(patient_dir.name) + '/' + 'AAA1.nii'
                aaa_nii_file = nib.load(original_AAA_file)
                nib.save(aaa_nii_file, out_AAA_dir / f"arota_{str(patient_dir).split('_')[1].zfill(3)}.nii.gz")

                num_training_cases += 1
            elif file.suffix == ".nii" and "VesselMask_2_label" in file.name: #如果考虑更新label，那么旧的label文件就不要带label了
                # Load the .nii file
                nii_file = nib.load(str(patient_dir) + "/" + file.name)
                # Save the file as .nii.gz
                nib.save(nii_file, labels_dir / f"arota_{str(patient_dir).split('_')[1].zfill(3)}.nii.gz")

                # shutil.copy(file, labels_dir / f"arota_{str(patient_dir).split('_')[1].zfill(3)}.nii")
    # Copy test files.
    #    file.name 将返回 a_file.txt（完整的文件名和扩展名）。The .name attribute specifically returns the last component of the path, regardless of whether it's a file name or a directory name
    #    file.stem 将返回 a_file（仅文件名，不包括扩展名）。
    #    file.suffix 将返回 .txt（仅扩展名）。
    for patient_dir in patients_test:
        print(patient_dir)
        for file in Path(patient_dir).iterdir():
            if file.suffix == ".nii":
                # Load the .nii file
                nii_file = nib.load(str(patient_dir) + "/" + file.name)
                # Save the file as .nii.gz
                nib.save(nii_file, test_dir / f"arota_{str(patient_dir).split('_')[1].zfill(3)}_0000.nii.gz")

                # shutil.copy(file, test_dir / f"arota_{str(patient_dir).split('_')[1].zfill(3)}_0000.nii")

    return num_training_cases


def convert_own(src_data_folder: str, test_list: list, dataset_id=100):
    out_dir, train_dir, labels_dir, test_dir, out_key_dir, out_AAA_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir, test_list, out_key_dir, out_AAA_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "CT",
        },#not sure here!!
        labels={
            "background": 0,
            "Arota": 1,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
        key_ending=".json"
    )

    generate_key_json(
        str(out_dir),
        channel_names={
            0: "CT",
        },  # not sure here!!
        labels={
            "background": 0,
            "Key_point1": 1,
            "Key_point2": 2,
            "Key_point3": 3,
            "Key_point4": 4,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
        key_ending=".json"
    )

    generate_AAA_json(
        str(out_dir),
        channel_names={
            0: "CT",
        },  # not sure here!!
        labels={
            "background": 0,
            "AAA": 1,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
        key_ending=".json"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        default='/home/yiwen/FinalDataset/',
        type=str,
        help="The downloaded our own dataset dir. Should contain extracted 'training' and 'testing' folders.",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=27, help="nnU-Net Dataset ID, default: 100"
    )
    args = parser.parse_args()
    test_list_ori = ['Patient_4','Patient_15','Patient_28']
    test_list = [args.input_folder + item for item in test_list_ori]
    print('111',test_list)
    print("Converting...")
    convert_own(args.input_folder, test_list, args.dataset_id)
    print("Done!")