import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt

from matplotlib.patheffects import withStroke

# custom
from data_utilities import *

# data directory

file_name = "sub-gl108"
# file_name = "sub-verse511"

img_nib = nib.load(f"./{file_name}_0000.nii.gz")
msk_nib = nib.load(f"./{file_name}.nii.gz")

print(f"computing file {file_name}")

# # load files
img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
msk_iso = resample_nib(msk_nib, voxel_spacing=(1, 1, 1), order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)

img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
msk_iso = reorient_to(msk_iso, axcodes_to=('I', 'P', 'L'))

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

## Get the median info for each spine, if there exists over THRESHOLD pixels in the plane.
# use median to avoid impact by outliers

med_sag = np.empty((28,2,))
med_sag[:] = np.nan

med_cor = np.empty((28,2,))
med_cor[:] = np.nan

THRESHOLD = 300
for i in range(1,29):
    if (np.size(msk_np_sag[msk_np_sag==i]) >= THRESHOLD):
        row, col = np.where(msk_np_sag == i)
        med_sag[i,0] = np.median(row)
        med_sag[i,1] = np.median(col)

    if (np.size(msk_np_cor[msk_np_cor==i]) >= THRESHOLD):
        row, col = np.where(msk_np_cor == i)
        med_cor[i,0] = np.median(row)
        med_cor[i,1] = np.median(col)

# plot 
fig, axs = create_figure(96,im_np_sag, im_np_cor)


label = ["background", "C1","C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "T12",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
        "sacrum",
        "cocygis",
        "T13"]

axs[0].imshow(im_np_sag, cmap=plt.cm.gray, norm=wdw_sbone)
axs[1].imshow(im_np_cor, cmap=plt.cm.gray, norm=wdw_sbone)
plt.savefig(f'./{file_name}_vanilla.png')

axs[0].imshow(msk_np_sag, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
axs[1].imshow(msk_np_cor, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
plt.savefig(f'./{file_name}_label.png')

for i in range(28):
    if (med_sag[i,0] != np.nan):
        axs[0].plot(med_sag[i,1], med_sag[i,0], 'ro', markersize=7)
        axs[0].text(med_sag[i,1] + 10,  med_sag[i,0], label[i], color='red', fontsize=10, va='center', ha='left', alpha = 0.6, path_effects=[withStroke(linewidth=5, foreground='white')])


for i in range(28):
    if (med_cor[i,0] != np.nan):
        axs[1].plot(med_cor[i,1], med_cor[i,0], 'ro', markersize=7)
        axs[1].text(med_cor[i,1] + 10,  med_cor[i,0], label[i], color='red', fontsize=10, va='center', ha='left',  alpha = 0.6, path_effects=[withStroke(linewidth=5, foreground='white')])

plt.savefig(f'./{file_name}_label_centroid.png')
print("images generated.")

