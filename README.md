# SAL_ext_BioImage

## about Data
lung_image_mask.py is about reading the data, lung images and masks, from the dataset, which is downloaded from 
https://www.kaggle.com/kmader/finding-lungs-in-ct-data

lung_age_contrast.py is about reading and dealing with the data, lung images and the contrast, from the dataset full_archive.npz, which is downloaded from 
https://www.kaggle.com/kmader/siim-medical-image-analysis-tutorial

lung_healthy_cancer.py has two nain functions: Read the cancer and healthy data sets, containing about 10000 images; Use the TFrecord to make a dataset containing all the training images and its corresponding labels.
Healthy dataset: http://www.via.cornell.edu/databases/lungdb.html

Cancer dataset: https://wiki.cancerimagingarchive.net/display/Public/LungCT-Diagnosis;jsessionid=B2E28A358D38041800E1D90F4C7D108F#84dcce1ae70b450fa7c3cbabc6dc5164

## about Model
CNN_lung_image_mask.py and CNN_lung_age_contrast.py use the same cnn model to deal with the lung images and the masks and contrast. Since the input and output is not same, the shapes of input and output of these two models are different. 

lung_healthy_cancer.py use the data from lung_healthy_cancer.py and then train the net model.

## lung_h/c_contrast_classification
LUNGpreprocess.py contains some image preprocessing methods, such as Gray value conversion, Re-sampling, segmentation, Normalized and 0 value centering. 

LUNGdataset.py uses some methods in the LUNGpreprocess.py and make a TFrecord dataset, which are feed into CNN_cancer_healthy.py, containing healthy and cancer images anf corresponding labels.

LUNGcontrast_dataset.py uses some methods in the LUNGpreprocess.py and make a TFrecord dataset, which are feed into CNN_contrast.py, containing lung images anf corresponding contrast labels.

CNN_cancer_healthy.py and CNN_contrast.py are the classifier models.

