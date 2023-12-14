import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt

# custom
from data_utilities import *

# data directory

file_name = "sub-verse511"

print(f"computing file {file_name}")

# load files
img_nib = nib.load(f"./nnUNet_raw/Dataset101_VerSe2020/imagesTr/{file_name}_0000.nii.gz")
msk_nib = nib.load(f"./nnUNet_raw/Dataset101_VerSe2020/labelsTr/{file_name}.nii.gz")


img_iso = reorient_to(img_nib, axcodes_to=('I', 'P', 'L'))
msk_iso = reorient_to(msk_nib, axcodes_to=('I', 'P', 'L'))

# get vocel data
im_np  = img_iso.get_fdata()
msk_np = msk_iso.get_fdata()


# get the mid-slice of the scan and mask in both sagittal and coronal planes

im_np_sag = im_np[:,:,int(im_np.shape[2]/2)]
im_np_cor = im_np[:,int(im_np.shape[1]/2),:]

msk_np_sag = msk_np[:,:,int(msk_np.shape[2]/2)]
msk_np_sag[msk_np_sag==0] = np.nan

msk_np_cor = msk_np[:,int(msk_np.shape[1]/2),:]
msk_np_cor[msk_np_cor==0] = np.nan


# plot 
fig, axs = create_figure(96,im_np_sag, im_np_cor)

axs[0].imshow(im_np_sag, cmap=plt.cm.gray, norm=wdw_sbone)
axs[0].imshow(msk_np_sag, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)

axs[1].imshow(im_np_cor, cmap=plt.cm.gray, norm=wdw_sbone)
axs[1].imshow(msk_np_cor, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)

plt.savefig(f'./{file_name}_vanilla.png')
print("vanilla image generated.")

plt.clf()

img_nib = nib.load(f"./nnUNet_raw/Dataset105_VerSe2020/imagesTr/{file_name}_0000.nii.gz")
msk_nib = nib.load(f"./nnUNet_raw/Dataset105_VerSe2020/labelsTr/{file_name}.nii.gz")


# get vocel data
im_np  = img_nib.get_fdata()
msk_np = msk_nib.get_fdata()


# get the mid-slice of the scan and mask in both sagittal and coronal planes

im_np_sag = im_np[:,:,int(im_np.shape[2]/2)]
im_np_cor = im_np[:,int(im_np.shape[1]/2),:]

msk_np_sag = msk_np[:,:,int(msk_np.shape[2]/2)]
msk_np_sag[msk_np_sag==0] = np.nan

msk_np_cor = msk_np[:,int(msk_np.shape[1]/2),:]
msk_np_cor[msk_np_cor==0] = np.nan


fig, axs = create_figure(96,im_np_sag, im_np_cor)

axs[0].imshow(im_np_sag, cmap=plt.cm.gray, norm=wdw_sbone)
axs[0].imshow(msk_np_sag, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)

axs[1].imshow(im_np_cor, cmap=plt.cm.gray, norm=wdw_sbone)
axs[1].imshow(msk_np_cor, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)

plt.savefig(f'./{file_name}_transformed.png')
print("transformed image generated.")