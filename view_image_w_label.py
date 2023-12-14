import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt

# custom
from data_utilities import *

# data directory

file_name = "sub-gl428"

def dice_and_image(dataset, year, file_name):
    print(f"computing file {file_name}")

    # load files
    img_nib = nib.load(f"./nnUNet_raw/Dataset{dataset}_VerSe20{year}/imagesTs/{file_name}_0000.nii.gz")
    msk_nib = nib.load(f"./nnUNet_predicted/Dataset{dataset}/{file_name}.nii.gz")
    true_msk_nib = nib.load(f"../01_data/02_VerseNewData/dataset-verse{year}test/dataset-03test/derivatives/{file_name}/{file_name}_dir-ax_seg-vert_msk.nii.gz")

    #check img zooms 
    zooms = img_nib.header.get_zooms()
    # print('img zooms = {}'.format(zooms))

    #check img orientation
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine))
    # print('img orientation code: {}'.format(axs_code))


    # Resample and Reorient data
    img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
    msk_iso = resample_nib(msk_nib, voxel_spacing=(1, 1, 1), order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
    true_msk_iso = resample_nib(true_msk_nib, voxel_spacing=(1, 1, 1), order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)

    img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
    msk_iso = reorient_to(msk_iso, axcodes_to=('I', 'P', 'L'))
    true_msk_iso = reorient_to(true_msk_iso, axcodes_to=('I', 'P', 'L'))

    #check img zooms 
    zooms = img_iso.header.get_zooms()

    #check img orientation
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_iso.affine))
    # print('img orientation code: {}'.format(axs_code))


    # get vocel data
    im_np  = img_iso.get_fdata()
    msk_np = msk_iso.get_fdata()
    true_msk_np = true_msk_iso.get_fdata()


    # get the mid-slice of the scan and mask in both sagittal and coronal planes

    im_np_sag = im_np[:,:,int(im_np.shape[2]/2)]
    im_np_cor = im_np[:,int(im_np.shape[1]/2),:]

    msk_np_sag = msk_np[:,:,int(msk_np.shape[2]/2)]
    msk_np_sag[msk_np_sag==0] = np.nan

    msk_np_cor = msk_np[:,int(msk_np.shape[1]/2),:]
    msk_np_cor[msk_np_cor==0] = np.nan

    true_msk_np_sag = true_msk_np[:,:,int(true_msk_np.shape[2]/2)]
    true_msk_np_sag[true_msk_np_sag==0] = np.nan

    true_msk_np_cor = true_msk_np[:,int(true_msk_np.shape[1]/2),:]
    true_msk_np_cor[true_msk_np_cor==0] = np.nan

    # compute dice coeff w/ background

    matching = msk_np == true_msk_np
    dice = np.size(matching[matching == True]) / np.size(msk_np)
    print(f"The dice coeff. of predicted {file_name} is {dice}.")

    # compute dice coeff w/o background


    # matching_on_true_label_area = matching[true_msk_np != 0.]
    # print(matching_on_true_label_area)
    # dice = np.size(matching_on_true_label_area[matching_on_true_label_area == True]) / np.size(matching_on_true_label_area)
    # print(f"The label only dice coeff. of predicted {file_name} is {dice}.")

    # plot 
    fig, axs = create_figure(96,im_np_sag, im_np_cor)

    axs[0].imshow(im_np_sag, cmap=plt.cm.gray, norm=wdw_sbone)
    axs[0].imshow(msk_np_sag, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)

    axs[1].imshow(im_np_cor, cmap=plt.cm.gray, norm=wdw_sbone)
    axs[1].imshow(msk_np_cor, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)

    plt.savefig(f'./nnUNet_predicted/Dataset{dataset}/{file_name}.png')
    print("predicted image generated.")

    plt.clf()

    fig, axs = create_figure(96,im_np_sag, im_np_cor)

    axs[0].imshow(im_np_sag, cmap=plt.cm.gray, norm=wdw_sbone)
    axs[0].imshow(true_msk_np_sag, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)

    axs[1].imshow(im_np_cor, cmap=plt.cm.gray, norm=wdw_sbone)
    axs[1].imshow(true_msk_np_cor, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)

    plt.savefig(f'./nnUNet_predicted/Dataset{dataset}/{file_name}_true.png')
    print("true image generated.")

    return dice

def test_data(year, dataset):
    num_files = 0
    dice_sum = 0
    pred = f"./nnUNet_predicted/Dataset{dataset}"
    for (_, _, files) in os.walk(pred):
        for file in files:
            if file.endswith(".nii.gz"):
                file_name = file.split('.')[0]
                num_files += 1
                dice_sum += dice_and_image(dataset, year, file_name)
    print(f"Total average dice of {dice_sum/num_files}.")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', type=str, help="data year. 19 or 20")
    parser.add_argument('-d', type=int, help='nnU-Net Dataset ID, default: 100')
    # parser.add_argument('-u', '--use-transform', action='store_true')
    args = parser.parse_args()
    test_data(args.y, args.d)
