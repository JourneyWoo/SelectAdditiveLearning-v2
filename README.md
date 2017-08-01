# SAL_ext_BioImage

## about Data
lung_image_mask.py is about reading the data, lung images and masks, from the dataset, which is downloaded from 
https://www.kaggle.com/kmader/finding-lungs-in-ct-data

lung_age_contrast.py is about reading and dealing with the data, lung images and the contrast, from the dataset full_archive.npz, which is downloaded from 
https://www.kaggle.com/kmader/siim-medical-image-analysis-tutorial

## about Model
CNN_lung_image_mask.py and CNN_lung_age_contrast.py use the same cnn model to deal with the lung images and the masks and contrast. Since the input and output is not same, the shapes of input and output of these two models are different. 
