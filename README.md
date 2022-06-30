# Checkpoints Analysis [![gcash donation][1]][2] [![paypal donation][3]][4]
[![license][5]][6] [![python version][7]][8]
 
 
This repository contains two main documents, that analyzes checkpoints implementation in sk-learn's Random Forest and as well as in TensorFlow.

## Checkpoints Analysis on Scikit Learn using Heart Disease Dataset
There is this warm_start parameter that is comparable to checkpoints in TF. I implemented it in the heart disease dataset and produced an output where we add more trees per checkpoint. For each checkpoint, it will produce a pkl file like the one below:
 * [random_forest_ckp_9.pkl](https://github.com/mcabanlit/checkpoints-analysis/blob/main/saved_models/random-forest/random_forest_ckp_9.pkl)

### Dataset

### Accuracy
The accuracy of the exported models per checkpoint were as follows: 

## Checkpoints Analysis on Keras using Microscopy Dataset
For each improvement in the accuracy of an epoch, a checkpoint would be generated. It will output a file like the one below:
* [jupnote_model.05-0.72.h5](https://github.com/mcabanlit/checkpoints-analysis/blob/main/saved_models/jupnote_model.05-0.72.h5)

### Dataset


### Accuracy
I added a check for the accuracy of my exported loaded checkpoint against validation, training and both. The accuracy seems to be in proper percentages:
* Loaded Model                  - 0.72 (Accuracy when fitting)
* Validation Data                - 0.72
* Training Data                   - 1.00
* Merged Training and Val - 0.9933


## Conclusion

[1]: https://img.shields.io/badge/donate-gcash-green
[2]: https://drive.google.com/file/d/1JeMx5_S7VBBT-3xO7mV9YOMfESeV3eKa/view

[3]: https://img.shields.io/badge/donate-paypal-blue
[4]: https://www.paypal.com/paypalme/mcabanlitph

[5]: https://img.shields.io/badge/license-GNUGPLv3-blue.svg
[6]: https://github.com/mcabanlit/heart-disease/blob/main/LICENSE.md

[7]: https://img.shields.io/badge/python-3.10-blue
[8]: https://www.python.org/
