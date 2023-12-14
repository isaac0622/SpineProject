import shutil, os
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw # raw path

import nibabel as nib
import nibabel.orientations as nio
# custom by VerSe authors
from data_utilities import resample_nib, reorient_to

def rescale_reorient_and_save_image(file_dir: str, dest_dir: str):
    # takes in path to file
    # read it by nibabel and rescale/reorient
    # then save it as dest_dir 
    img_nib = nib.load(file_dir)
    img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
    img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
    nib.save(img_iso, dest_dir)

def rescale_reorient_and_save_mask(file_dir: str, dest_dir: str):
    # takes in path to file
    # read it by nibabel and rescale/reorient
    # then save it as dest_dir 
    img_nib = nib.load(file_dir)
    img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=0) # difference here!
    img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
    # change label 28 to 26
    labels = img_iso.get_fdata()
    labels[labels == 28] = 26
    new_img = nib.Nifti1Image(labels, img_iso.affine, img_iso.header)
    nib.save(new_img, dest_dir)


def datagen_verse_vanilla(verse_base_dir: str, year: str, nnunet_dataset_id: int = 100, use_resample: bool = False):
    """
    VerSe has an extra validation set. We include them to the training set. 
    verse_base_dir: directory path for data folder.
    year: either 20 or 19
    nnunet_datased_id: use 101, 102, ...
    """

    # training data consists of:
    # ./dataset-verse##training/dataset-01training
    # ./dataset-verse##validation/dataset-02validation

    # test set is:
    # ./dataset-verse##test/dataset-03test
    if year == "20":
        train_dir = os.path.join(verse_base_dir, f"dataset-verse{year}training/dataset-01training")
        val_dir = os.path.join(verse_base_dir, f"dataset-verse{year}validation/dataset-02validation")
        test_dir = os.path.join(verse_base_dir, f"dataset-verse{year}test/dataset-03test")
    elif year == "19":
        train_dir = os.path.join(verse_base_dir, f"dataset-verse{year}training/dataset-verse{year}training")
        val_dir = os.path.join(verse_base_dir, f"dataset-verse{year}validation/dataset-verse{year}validation")
        test_dir = os.path.join(verse_base_dir, f"dataset-verse{year}test/dataset-verse{year}test")

    # Dataset101_VerSe2020
    nnunet_dataset_id %= 1000 
    foldername = f"Dataset{nnunet_dataset_id:03d}_VerSe20{year}"

    # setting up nnU-Net folders
    out_base = os.path.join(nnUNet_raw, foldername)
    imagestr = os.path.join(out_base, "imagesTr")
    imagests = os.path.join(out_base, "imagesTs")
    labelstr = os.path.join(out_base, "labelsTr")
    if not os.path.exists(imagestr): 
        os.makedirs(imagestr) 
    if not os.path.exists(imagests): 
        os.makedirs(imagests)
    if not os.path.exists(labelstr): 
        os.makedirs(labelstr)

    raw_ctr = 0
    label_ctr = 0


    # make training set data
    for base_dir in [train_dir, val_dir]:
        # traverse directory & copy nii.gz file. rawdata dir contains raw .nii.gz data only.
        for (root, _, files) in os.walk(os.path.join(base_dir, "rawdata")):
            # root: file's path
            # files: iterative of file names in root
            # There are very few roots that contain multiple files.
            for file in files:
                if file.endswith(".nii.gz"):
                    raw_ctr += 1
                    name = file.split('_')[0]
                    if not use_resample:
                        shutil.copy(os.path.join(root, file), os.path.join(imagestr, f'{name}_0000.nii.gz'))
                    else:
                        rescale_reorient_and_save_image(os.path.join(root, file), os.path.join(imagestr, f'{name}_0000.nii.gz'))
                   
        # traverse directory & copy nii.gz file. deriv. dir contains masking .nii.gz data & centroid json file
        for (root, _, files) in os.walk(os.path.join(base_dir, "derivatives")):
            # root: file's path
            # files: iterative of file names in root
            # There are very few roots that contain multiple files.
            for file in files:
                if file.endswith(".nii.gz"):
                    label_ctr += 1
                    name = file.split('_')[0]
                    if not use_resample:
                        shutil.copy(os.path.join(root, file), os.path.join(labelstr, f'{name}.nii.gz'))
                        # img_iso = nib.load(os.path.join(root, file))
                        # labels = img_iso.get_fdata()
                        # labels[labels == 28] = 26
                        # new_img = nib.Nifti1Image(labels, img_iso.affine, img_iso.header)
                        # nib.save(new_img, os.path.join(labelstr, f'{name}.nii.gz'))
                    else:
                        rescale_reorient_and_save_mask(os.path.join(root, file), os.path.join(labelstr, f'{name}.nii.gz'))

    tr_ctr = min(raw_ctr, label_ctr)

    base_dir = test_dir
    # test set
    for (root, _, files) in os.walk(os.path.join(base_dir, "rawdata")):
        # root: file's path
        # files: iterative of file names in root
        # There are very few roots that contain multiple files. Those will be overwritten.
        for file in files:
            if file.endswith(".nii.gz"):
                name = file.split('_')[0]
                if not use_resample:
                    shutil.copy(os.path.join(root, file), os.path.join(imagests, f'{name}_0000.nii.gz'))
                else:
                    rescale_reorient_and_save_image(os.path.join(root, file), os.path.join(imagests, f'{name}_0000.nii.gz'))
        
        # mask is not copied for test data
    # for (root, _, files) in os.walk(os.path.join(base_dir, "derivatives")):
    #     # root: file's path
    #     # files: iterative of file names in root
    #     # There are very few roots that contain multiple files. Those will be overwritten.
    #     for file in files:
    #         if file.endswith(".nii.gz"):
    #             label_ctr += 1
    #             shutil.copy(os.path.join(root, file), os.path.join(labelstr, f'{root}.nii.gz'))
    if not use_resample:
        generate_dataset_json(out_base, {0: "CT"}, 
                            labels = {"background": 0,
                                        "C1": 1,
                                        "C2": 2,
                                        "C3": 3,
                                        "C4": 4,
                                        "C5": 5,
                                        "C6": 6,
                                        "C7": 7,
                                        "T1": 8,
                                        "T2": 9,
                                        "T3": 10,
                                        "T4": 11,
                                        "T5": 12,
                                        "T6": 13,
                                        "T7": 14,
                                        "T8": 15,
                                        "T9": 16,
                                        "T10": 17,
                                        "T11": 18,
                                        "T12": 19,
                                        "L1": 20,
                                        "L2": 21,
                                        "L3": 22,
                                        "L4": 23,
                                        "L5": 24,
                                        "L6": 25,
                                        "sacrum": 26, # not labeled in this dataset => need to destroy!
                                        "cocygis": 27, # not labeled in this dataset => need to destroy!
                                        "T13": 28},
                            num_training_cases=tr_ctr, file_ending='.nii.gz',
                            overwrite_image_reader_writer='NibabelIOWithReorient')
    else:
        generate_dataset_json(out_base, {0: "CT"}, 
                            labels = {"background": 0,
                                        "C1": 1,
                                        "C2": 2,
                                        "C3": 3,
                                        "C4": 4,
                                        "C5": 5,
                                        "C6": 6,
                                        "C7": 7,
                                        "T1": 8,
                                        "T2": 9,
                                        "T3": 10,
                                        "T4": 11,
                                        "T5": 12,
                                        "T6": 13,
                                        "T7": 14,
                                        "T8": 15,
                                        "T9": 16,
                                        "T10": 17,
                                        "T11": 18,
                                        "T12": 19,
                                        "L1": 20,
                                        "L2": 21,
                                        "L3": 22,
                                        "L4": 23,
                                        "L5": 24,
                                        "L6": 25,
                                        "T13": 26}, #changed label
                            num_training_cases=tr_ctr, file_ending='.nii.gz',
                            overwrite_image_reader_writer='NibabelIOWithReorient')
    
def datagen_verse_2d(verse_base_dir: str, year: str, nnunet_dataset_id: int = 102):
    pass

# place file in data folder and run
# python datagen_verse.py vanilla 20 -d=101 --dir=../01_data/02_VerseNewData
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help="either 'vanilla' or '2D'")
    parser.add_argument('y', type=str, help="data year. 19 or 20")
    parser.add_argument('--dir', required=False, type=str, default="../01_data/02_VerseNewData", help="Directory path for data.")
    parser.add_argument('-d', required=False, type=int, default=100, help='nnU-Net Dataset ID, default: 100')
    parser.add_argument('-u', '--use-transform', action='store_true')
    args = parser.parse_args()
    if args.type == "vanilla":
        datagen_verse_vanilla(args.dir, args.y, args.d, args.use_transform)
        print("Successfully imported.")
    elif args.type == "2D":
        datagen_verse_2d(args.dir, args.y, args.d, args.use_transform)
        print("Successfully imported.")
    else:
        print("wrong argument")