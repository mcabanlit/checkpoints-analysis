# Checkpoints Analysis [![gcash donation][1]][2] [![paypal donation][3]][4]
[![python version][7]][8]
 
 
This repository contains two main documents, that analyzes checkpoints implementation in sk-learn's Random Forest and as well as in TensorFlow.

## Checkpoints Analysis on Scikit Learn using Heart Disease Dataset
There is this warm_start parameter that is [comparable to checkpoints](https://datascience.stackexchange.com/questions/49012/checkpoints-in-sklearn) in TF. I implemented it in the heart disease dataset and produced an output where we add more trees per checkpoint. For each checkpoint, it will produce a pkl file like the one below:
 * [random_forest_ckp_9.pkl](https://github.com/mcabanlit/checkpoints-analysis/blob/main/saved_models/random-forest/random_forest_ckp_9.pkl)

### Dataset
The dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
Uses the following predictors to predict if a person has or does not have heart disease:
  * Age
  * Sex
  * Chest Pain Type
  * Resting Blood Pressure
  * Cholesterol
  * Fasting Blood Sugar
  * Resting ECG
  * Maximum Heart Rate
  * Exercise Induced Angina
  * Previous Peak
  * Slope of Peak
  * Number of Major Vessels
  * Thalassemia
 
### Accuracy
The accuracy of the exported models per checkpoint has increased from 0.99 on the first checkpoint to 1.0 on the 10 checkpoint. 

###  
  
    

## Checkpoints Analysis on Keras using Microscopy Dataset
For each improvement in the accuracy of an epoch, a checkpoint would be generated. It will output a file like the one below:
* [jupnote_model.05-0.72.h5](https://github.com/mcabanlit/checkpoints-analysis/blob/main/saved_models/jupnote_model.05-0.72.h5)

### Dataset
The dataset was retrieved from the [Python for Microscopists Github Repository of @bnsreenu](https://github.com/bnsreenu/python_for_microscopists). The data are images of cells that has and does not have parasites such as the ones below:

_Figure 1. Cell with Parasite [(C33P1thinF_IMG_20150619_114756a_cell_179)](https://github.com/mcabanlit/checkpoints-analysis/blob/main/data/cell_images/test/Parasitized/C33P1thinF_IMG_20150619_114756a_cell_179%20-%20Copy%20-%20Copy.png)_ 

![C33P1thinF_IMG_20150619_114756a_cell_179 - Copy - Copy](https://user-images.githubusercontent.com/102983286/176599274-b0171903-e31e-4081-a178-4341a7fd330b.png)



_Figure 2. Cell without Parasite [(C1_thinF_IMG_20150604_104722_cell_115)](https://github.com/mcabanlit/checkpoints-analysis/blob/main/data/cell_images/test/Uninfected/C1_thinF_IMG_20150604_104722_cell_115%20-%20Copy%20-%20Copy.png)_

![C1_thinF_IMG_20150604_104722_cell_9 - Copy - Copy](https://user-images.githubusercontent.com/102983286/176599296-90f57da1-5175-4025-a0ad-9039a7f50466.png)



### Accuracy
I added a check for the accuracy of my exported loaded checkpoint against validation, training and both. The accuracy seems to be in proper percentages:
* Accuracy After Training  - 0.68 ([malaria_augmented_model.h5](https://github.com/mcabanlit/checkpoints-analysis/blob/main/malaria_augmented_model.h5))
* Loaded Model            - 0.72 ([jupnote_model.05-0.72.h5](https://github.com/mcabanlit/checkpoints-analysis/blob/main/saved_models/jupnote_model.05-0.72.h5)) 
* Validation Data          - 0.72 ([loaded_model_malaria_augmented_model.h5](https://github.com/mcabanlit/checkpoints-analysis/blob/main/saved_models/loaded_model_malaria_augmented_model.h5)) 
* Training Data            - 1.00 ([loaded_model_malaria_augmented_model.h5](https://github.com/mcabanlit/checkpoints-analysis/blob/main/saved_models/loaded_model_malaria_augmented_model.h5))
* Merged Training and Val - 0.9933 ([loaded_model_malaria_augmented_model.h5](https://github.com/mcabanlit/checkpoints-analysis/blob/main/saved_models/loaded_model_malaria_augmented_model.h5))


## Conclusion
Checkpoints are very helpful for processing large datasets. For `warm_start` in the random forest model, it is a good method of improving the accuracy of an already fitted model. For deep learning, the `checkpoints` were very useful on creating restore points during the training.


[1]: https://img.shields.io/badge/donate-gcash-green
[2]: https://drive.google.com/file/d/1JeMx5_S7VBBT-3xO7mV9YOMfESeV3eKa/view

[3]: https://img.shields.io/badge/donate-paypal-blue
[4]: https://www.paypal.com/paypalme/mcabanlitph

[7]: https://img.shields.io/badge/python-3.10-blue
[8]: https://www.python.org/
