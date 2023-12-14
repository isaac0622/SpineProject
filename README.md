# Spine Labeling Project
2023-2 Creative Integrated Design 1 Project Team A

팀장 2020- 전민주
팀원 2020-19851 오이석

## Repository
`Dataset101`: Processed & Trained Result of VerSe2020 Dataset. Trained CT image (`.nii.gz`) excluded due to large memory.   

`results`: Images (sagittal plane, etc) of true/inferenced data. 

`README.md`: This file.

`compute_dice.py`: Computing dice coefficient for resulting CT images.

`data_utilities.py`: Data utilities for taking planed images.

`datagen_verse.py`: Preprocessing of VerSe dataset for nnUNet use. Includes various preprocessing options.

`example_image.py`: Draw plane image from CT images.

`example_image_centroid.py`: Inference centroid of each spine, using mean.

`example_image_median.py`: Inference centroid of each spine, using median, a robust statistic.

`view_image_w_label.py`: Simultaneously compute dice coefficient and draw plane images.

## Dataset
실제 VerSe Dataset은 그 용량이 매우 크기에 다음 GitHub Repository에서 받을 수 있다.

[VerSe: Large Scale Vertebrae Segmentation Challenge](https://github.com/anjany/verse)

## Deep Learning Library

학습의 기반이 된 딥러닝 라이브러리는 nnUNet으로, 다음 GitHub Repository에서 받을 수 있다.

[nnUNet](https://github.com/MIC-DKFZ/nnUNet)
