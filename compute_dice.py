import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt

# custom
from data_utilities import *

# data directory

# file_name = "sub-gl428"

def dice(dataset, year, file_name):
    print(f"computing file {file_name}")

    # load files
    msk_nib = nib.load(f"./nnUNet_predicted/Dataset{dataset}/{file_name}.nii.gz")
    true_msk_nib = nib.load(f"../01_data/02_VerseNewData/dataset-verse{year}test/dataset-03test/derivatives/{file_name}/{file_name}_dir-ax_seg-vert_msk.nii.gz")

    # msk_iso = resample_nib(msk_nib, voxel_spacing=(1, 1, 1), order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
    # true_msk_iso = resample_nib(true_msk_nib, voxel_spacing=(1, 1, 1), order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)

    # msk_iso = reorient_to(msk_iso, axcodes_to=('I', 'P', 'L'))
    # true_msk_iso = reorient_to(true_msk_iso, axcodes_to=('I', 'P', 'L'))

    msk_np = msk_nib.get_fdata()
    true_msk_np = true_msk_nib.get_fdata()


    # compute dice coeff w/ background

    matching = msk_np == true_msk_np
    dice = np.size(matching[matching == True]) / np.size(msk_np)
    print(f"The dice coeff. of predicted {file_name} is {dice}.")

    # compute dice coeff w/o background

    matching_on_true_label_area = matching[true_msk_np != 0.]
    print(matching_on_true_label_area)
    label_dice = np.size(matching_on_true_label_area[matching_on_true_label_area == True]) / np.size(matching_on_true_label_area)
    print(f"The label only dice coeff. of predicted {file_name} is {label_dice}.")

    return dice, label_dice

def test_data(year, dataset, label_only):
    if (label_only):
        flag = 1
    else:
        flag = 0
    num_files = 0
    dice_sum = 0
    pred = f"./nnUNet_predicted/Dataset{dataset}"
    for (_, _, files) in os.walk(pred):
        for file in files:
            if file.endswith(".nii.gz"):
                file_name = file.split('.')[0]
                num_files += 1
                dice_sum += dice(dataset, year, file_name)[flag]
    print(f"Total average dice of {dice_sum/num_files}.")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', type=str, help="data year. 19 or 20")
    parser.add_argument('-d', type=int, help='nnU-Net Dataset ID, default: 100')
    parser.add_argument('-l', '--label-only', action='store_true')
    args = parser.parse_args()
    if (args.y == None or args.d == None):
        print("Need arguments. Check -h.")
    else:
        test_data(args.y, args.d, args.label_only)
