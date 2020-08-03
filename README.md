# deep-learning-final-project
An ablation study of different semi-supervised CNN frameworks applied to the SVHN dataset.


# Abstract

This project seeks to explore a body of research concerned with noisy semi-supervised approaches to training convolutional neural networks. This project is focused on experimenting with different settings on the classic student-teacher method described in Google’s noisy student paper [3]. The three areas where we experiment with different settings are with the type of noise used, the number of student-teacher iterations, and the cutoff threshold for assigned labels to unclassified data. All of our methods are tested on the The Street View House Numbers (SVHN) dataset. Our goal is to discover which set of the hyperparameters described above leads to the most accurate model. 
