from SVHN_Model_final import SVHN_Model as Model
import pandas as pd
import numpy as np

if __name__== "__main__":
   accuracy = Model.train_base_model()

   df = pd.DataFrame(accuracy)
   df.to_csv('Data/base_accuracy.csv')

    for thresh in [97,98,99]:
        
        for noise in ['Rand','Traditional','']:
            accuracy = []
            for interation in range(3):
                model = '{}_{}_{}'.format(thresh,noise,interation)
                print(model)
                
                if interation==0:
                    accuracy.append(Model.student_training(augment = noise, percentile = thresh, model_type = 'ResNext' ,save = True, cycle = False, epochs=30))
               
                else:
                    accuracy.append(Model.student_training(augment = noise, percentile = thresh, model_type = 'ResNext' ,save = True, cycle = True, epochs=30))

                df = pd.DataFrame(accuracy)
                df.to_csv('Data/'+model+'.csv')
