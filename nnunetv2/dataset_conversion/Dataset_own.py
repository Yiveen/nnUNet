import os
import shutil
from pathlib import Path
import nibabel as nib

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def make_out_dirs(dataset_id: int, task_name="Aorta"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path, test_list: list, test_list_ori:list):
    """Copy files from the ACDC dataset to the nnUNet dataset folder. Returns the number of training cases."""
    patients_train = sorted([f for f in (src_data_folder).iterdir() if (f.is_dir() and str(f) not in test_list)])
    # for item in src_data_folder.iterdir() :
    #     if (str(item) in test_list):
    #         print(True)
    patients_test = test_list

    num_training_cases = 0
    # Copy training files and corresponding labels.
    for patient_dir in patients_train:
        print(patient_dir)
        for file in patient_dir.iterdir():
            if file.suffix == ".nii" and "label" not in file.name and "CTA_2" in file.name:
                # The stem is 'patient.nii', and the suffix is '.gz'.
                # We split the stem and append _0000 to the patient part.

                # Load the .nii file
                nii_file = nib.load(str(patient_dir) + "/" + file.name)
                # Save the file as .nii.gz
                nib.save(nii_file, train_dir / f"arota_{str(patient_dir).split('_')[1].zfill(3)}_0000.nii.gz")
                # shutil.copy(file, train_dir / f"arota_{str(patient_dir).split('_')[1].zfill(3)}_0000.nii")
                num_training_cases += 1
            elif file.suffix == ".nii" and "label" in file.name:
                # Load the .nii file
                nii_file = nib.load(str(patient_dir) + "/" + file.name)
                # Save the file as .nii.gz
                nib.save(nii_file, labels_dir / f"arota_{str(patient_dir).split('_')[1].zfill(3)}.nii.gz")

                # shutil.copy(file, labels_dir / f"arota_{str(patient_dir).split('_')[1].zfill(3)}.nii")
    # Copy test files.
    #    file.name 将返回 a_file.txt（完整的文件名和扩展名）。
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


def convert_own(src_data_folder: str, test_list: list, test_list_ori:list, dataset_id=100):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir, test_list, test_list_ori)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "CT",
        },#not sure here!!
        labels={
            "background": 0,
            "AORTA": 1,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        default='/home/yiwen/dataset/Newdataset/',
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
    convert_own(args.input_folder, test_list, test_list_ori, args.dataset_id)
    print("Done!")