from SVHN_Model_final import SVHN_Model as Model
import pandas as pd
import numpy as np

"""
The code below can be broken down into 3 steps:

Step 1: Obtain the base model. This model will later be used to provide
        pseudo-labels to the unlabeled training data. We will then
        save the accuracy of this model and use it as a baseline to interpret
        how our student-teacher iterations will improve upon the model.

Step 2: Establish hyperamaters for the student-teacher iterations. Here me are
        interested in seeing the effect of adjusting 3 hyperparamenters.
        These hyperparameters are:
            A: The threshold of confidence of the teacher model on its prediction of the 
               unlabeled data.
            B: The type of noise that is applied to the pseudo-labeled data.
            C: The number of iterations of student-teacher training applied.

Step 3: Commence student-teacher training. In this process each preceding model ("the teacher")
        assigns labels to the unlabaled training data. This data is then noised and fed,
        along with the labeled training data, to the next model ("the student")  to train. We 
        then check the accuracy of the student on the dataset. At thit point the process starts
        again and the student becomes the new teacher.
"""


if __name__== "__main__":

    accuracy = Model.train_base_model()
    df = pd.DataFrame(accuracy)
    df.to_csv('Data/base_accuracy.csv')

    thresholds = [97,98,99]
    noises = ['Rand','Traditional','']
    iterations = 3

    for thresh in thresholds:
        
        for noise in noises:

            accuracy = []
            for iteration in range(iterations):
                model = '{}_{}_{}'.format(thresh,noise,iteration)
                print(model)
                
                if iteration==0:
                    accuracy.append(Model.student_training(augment = noise,
                    percentile = thresh, model_type = 'ResNext' ,save = True, cycle = False, epochs=30))
               
                else:
                    accuracy.append(Model.student_training(augment = noise,
                    percentile = thresh, model_type = 'ResNext' ,save = True, cycle = True, epochs=30))

                df = pd.DataFrame(accuracy)
                df.to_csv('Data/'+model+'.csv')
