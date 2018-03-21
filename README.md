# logistic-regression-using-python
Implementation of logistic regression in python from scratch

The objective of this logistic regression model is to classify dogs and cats images
The dataset is from a competition on Kaggle on classifying cats and dogs

Dataset: https://www.kaggle.com/c/dogs-vs-cats

Out of around 25000 images, only 200 images are used, each category having 100 images

File:     image_preprocess.py                - for preprocessing the raw images using OPENCV and store the processed images

File:     images.pickle                      - the processed images are stored in pickle format

File:     classify.py                        - implementation of the model, training and testing the model

The model is implemented with regularization

With regularisation:

  Training accuracy : (fluctuates around 99 %)
  
  Testing accuracy :  (fluctuates around 62 %)

The model can be bettered but the main objective was to implement logictic regression from scratch.

