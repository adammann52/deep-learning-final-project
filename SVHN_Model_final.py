"""
Copyright (C) Adam Mann - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Adam Mann <adamimann@gmail.com>, June 2020
"""

import torch
import numpy as np
import torch
from torch.utils import data

# Import dataset related API
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Import common neural network API in pytorch
import torch.nn as nn
import torch.nn.functional as F

# Import optimizer related API
import torch.optim as optim

# Import Efficient net model (pytorch implementation)

from randaugment import RandAugment, ImageNetPolicy


#fix seeds for numpy and pytorch
torch.manual_seed(0)
np.random.seed(0) 

# Check device, using gpu 0 if gpu exist else using cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


"""
The following class, SVHN_Model, contains 4 functions that can be used to train
CNNs on the SVHN dataset.

train_base_model: trains a ResNext on a portion of the SVHN training data

train_base_model_total_data: trains ResNext on the entirety of the traing data

student_training: uses the previously saved model to provide pseudo-labals
                  to the remainder of the training data and uses it to train
                  a new model.

sudent_training_extra: similar to the above but provides pseudo labals to the
                       extra data

"""

class SVHN_Model:

# Stage 1: Initial Training on Labeled Data Set with No noise

    def train_base_model(model_type='ResNext'):

        #set transforms for the data
        transform = transforms.Compose([#transforms.ToPILImage()])
                                   transforms.ToTensor(),
                                   transforms.Normalize([.5,.5,.5],[.5,.5,.5])
                                   ])



        train_data = datasets.SVHN('Data/',split='train',download=True,transform=transform)
        test_data = datasets.SVHN('Data/',split='test',download=True,transform=transform)

        #pull data into dataloaders
        train_dataloader =  torch.utils.data.DataLoader(
                train_data, batch_size=1024,
            )
        test_dataloader = torch.utils.data.DataLoader(
                test_data, batch_size=1024, shuffle = False
            )

        #Upload petrained model
        if model_type == 'ResNext':
            model = models.resnext50_32x4d(pretrained=True)
        
        else:
            model = EfficientNet.from_pretrained('efficientnet-b0')

        #Add final layer for transefer learning
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 10),
                                 nn.Softmax())
        model = model.to(device)


        #Define training criteria
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=.05,momentum=.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) #obtained by trial and error
        model.to(device)

        train_index = (np.random.random(len(train_data))>=.75)

        # Training time
        print('Epoch  | Train Loss  |   Train Accuracy |  Val Loss  |  Val Accuracy   |    LR   ')
        print('_________________________________________________________________________________')
        q=0
        accuracy = []
        for epoch in range(20):
            tmp = train_index
            running_loss = 0.0
            running_corrects = 0
            model.train()

            # learning loop
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]

                indexes = tmp[:data[0].shape[0]]
                tmp = tmp[data[0].shape[0]:]
                

                inputs = data[0][indexes]
                labels = data[1][indexes]
                
                #inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    labels = labels.to(device)


                # zero the parameter gradients: Clean the gradient caclulated in the previous iteration
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                #print(outputs.shape)
                #print(labels.shape)

                loss = criterion(outputs, labels.long())

                # Calculate gradient of matrix with requires_grad = True
                loss.backward()

                # Apply the gradient calculate from last step to the matrix
                optimizer.step()
                # Add 1 more iteration count to learning rate scheduler


                # print statistics
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)


            for el in optimizer.param_groups:
              lr = el['lr']

            scheduler.step()
            val_corrects = 0
            val_loss = 0
            model.eval()

            #validation loop
            with torch.no_grad():
                for data in test_dataloader:
                    images, labels = data
                    if torch.cuda.is_available():
                        images = images.to(device)
                        labels = labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_loss += criterion(outputs, labels.long())
                    num = torch.sum(preds == labels.data)
                    val_corrects += num
            #print('Epoch  | Train Loss  |   Train Accuracy |  Val Loss  |  Val Accuracy   |   LR  ')


            print("  %2d   |    %.4f   |     %.4f       |   %.4f   |      %.4f     | %.5f "
                   %(q,running_loss/sum(train_index),int(running_corrects)/sum(train_index),val_loss/len(test_data),int(val_corrects)/len(test_data),lr))
                   
            accuracy.append(int(val_corrects)/len(test_data))

            q+=1


        print('Finished Training')

        torch.save(model, 'Data/'+model_type+'_model.pt')
        
        return(accuracy,train_index)


    def train_base_model_total_data(model_type='ResNext'):

        transform = transforms.Compose([#transforms.ToPILImage()])
                                   transforms.ToTensor(),
                                   transforms.Normalize([.5,.5,.5],[.5,.5,.5])
                                   ])



        train_data = datasets.SVHN('Data/',split='train',download=True,transform=transform)
        test_data = datasets.SVHN('Data/',split='test',download=True,transform=transform)

        train_dataloader =  torch.utils.data.DataLoader(
                train_data, batch_size=1024
            )
        test_dataloader = torch.utils.data.DataLoader(
                test_data, batch_size=1024
            )

        #Upload petrained model
        if model_type == 'ResNext':
            model = models.resnext50_32x4d(pretrained=True)
        
        else:
            model = EfficientNet.from_pretrained('efficientnet-b0')

        #Add final layer for transefer learning
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 10),
                                 nn.Softmax())
        model = model.to(device)


        #Define training criteria
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05,momentum=.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) #obtained by trial and error
        model.to(device)



        # Training time
        print('Epoch  | Train Loss  |   Train Accuracy |  Val Loss  |  Val Accuracy   |    LR   ')
        print('_________________________________________________________________________________')
        q=0
        accuracy = []
        for epoch in range(25):
            running_loss = 0.0
            running_corrects = 0
            model.train()
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    labels = labels.to(device)


                # zero the parameter gradients: Clean the gradient caclulated in the previous iteration
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                #print(outputs.shape)
                #print(labels.shape)

                loss = criterion(outputs, labels.long())

                # Calculate gradient of matrix with requires_grad = True
                loss.backward()

                # Apply the gradient calculate from last step to the matrix
                optimizer.step()
                # Add 1 more iteration count to learning rate scheduler


                # print statistics
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            for el in optimizer.param_groups:
              lr = el['lr']

            scheduler.step()
            val_corrects = 0
            val_loss = 0
            model.eval()

            with torch.no_grad():
                for data in test_dataloader:
                    images, labels = data
                    if torch.cuda.is_available():
                        images = images.to(device)
                        labels = labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_loss += criterion(outputs, labels.long())
                    num = torch.sum(preds == labels.data)
                    val_corrects += num
            #print('Epoch  | Train Loss  |   Train Accuracy |  Val Loss  |  Val Accuracy   |   LR  ')


            print("  %2d   |    %.4f   |     %.4f       |   %.4f   |      %.4f     | %.5f "
                   %(q,running_loss/len(train_data),int(running_corrects)/len(train_data),val_loss/len(test_data),int(val_corrects)/len(test_data),lr))
                   
            accuracy.append(int(val_corrects)/len(test_data))

            q+=1


        print('Finished Training')

        torch.save(model, 'Data/'+model_type+'_model.pt')
        
        return(accuracy)

    """# Stage 2: Predictions on unlabeled data"""
    def student_training(trained,augment = 'Rand', percentile = 0, model_type = 'ResNext' ,save = False, cycle = False, epochs=25):
        extra = trained==False

        if cycle:
            model = torch.load('Data/'+model_type+'_model_cycled.pt')
        else:
            model = torch.load('Data/'+model_type+'_model.pt')

        transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([.5,.5,.5],[.5,.5,.5])
                                   ])


        extra_data = datasets.SVHN('Data/',split='train',download=True,transform=transform)

        extra_loader = torch.utils.data.DataLoader(
                extra_data, batch_size=1024,shuffle=False)



        # Switch some layers (e.g., batch norm, dropout) to evaluation mode
        model.eval()
        # Turn off the autograd to save memory usage and speed up

        prediction_list =[]
        label_list = []
        tmp = extra
        with torch.no_grad():

            for data in extra_loader:
                images = data[0]
                labels = data[1]
                if torch.cuda.is_available():
                    images = images.to(device)
                    labels = labels.to(device)
                outputs = model(images)
                predicted = outputs.data
                prediction_list.append(predicted.cpu())
                label_list.append(labels.cpu())
        l_list = torch.cat(label_list).numpy()
        p_list = torch.cat(prediction_list)
        p_list = p_list.numpy()


        p_list[trained] = np.array([0]*10)

        """### Select threshold"""
        
        model = models.resnext50_32x4d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 10),
                                 nn.Softmax())

        threshold = np.percentile(p_list,percentile,axis=0)
        print(threshold)

        labels = (p_list>=threshold).astype(int)*(np.array([1]*10))



        ratio = np.min(sum(labels))/sum(labels)
        print(np.min(sum(labels)))
        print(sum(labels))
        print(ratio)
        rands = np.random.random(p_list.shape)<=ratio
        print(rands)
        labels = labels*rands

        print(sum(labels))

        indexes = np.array(list(range(np.max(p_list.shape))))[list((np.sum(labels,axis=1)>.1))]
        labels = np.argmax(labels[indexes],axis=1)


        train_data = datasets.SVHN('Data/',split='train',download=True,transform=transform)

        bs = int(sum(trained)/(sum(trained)+sum(extra))*1024)



        train_dataloader = torch.utils.data.DataLoader(
                train_data, batch_size=bs,
            )


        """## Stage 2.1: Reload Unlabeled (Extra) Data and apply *RandAugment*"""

        if augment == 'Rand':
            rand_transform = transforms.Compose([RandAugment(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([.5,.5,.5],[.5,.5,.5])
                                           ])
        elif augment == 'Traditional':
            rand_transform = transforms.Compose([transforms.RandomAffine(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize([.5,.5,.5],[.5,.5,.5])
                                           ])
        else:
            rand_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize([.5,.5,.5],[.5,.5,.5])
                                           ])


        aug_data = datasets.SVHN('Data/',split='train',download=True,transform=rand_transform)

        aug_dataloader = torch.utils.data.DataLoader(
                extra_data, batch_size=1024-bs,shuffle=False)



        indices = np.array([False]*aug_data.data.shape[0])
        indices[indexes] = True

        #only need to run this cell when step 1 was skipped
        test_data = datasets.SVHN('Data/',split='test',download=True,transform=transform)
        test_dataloader = torch.utils.data.DataLoader(
                test_data, batch_size=1024
            )


        #set features for learning parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05,momentum=.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
        model.to(device)

        # Commented out IPython magic to ensure Python compatibility.
        print('Epoch  | Train Loss  |   Train Accuracy |  Val Loss  |  Val Accuracy   |    LR   ')
        print('_________________________________________________________________________________')

        q=0
        run_len = train_data.data.shape[0]+labels.shape[0]
        test_len = test_data.data.shape[0]
        prev  = 0
        accuracy = []
        while q < epochs:
            running_loss = 0.0
            running_corrects = 0
            model.train()
            inds = indices
            labs = torch.Tensor(labels)

            for i in enumerate(zip(train_dataloader,aug_dataloader)):
                

                # get the inputs; data is a list of [inputs, labels]
                b_s = min((1024-bs),i[1][1][0].shape[0])
                aug = i[1][1][0][inds[:b_s]]
                inputs = torch.cat([i[1][0][0],aug])

                l = torch.cat([i[1][0][1].float(),labs[:aug.shape[0]].float()])

                # randomize the batches
                ran = torch.randperm(l.shape[0])
                inputs = inputs[ran]
                l = l[ran]
                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    l = l.to(device)


                # zero the parameter gradients: Clean the gradient caclulated in the previous iteration
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                #print(outputs.shape)
                #print(labels.shape)

                loss = criterion(outputs, l.long())

                # Calculate gradient of matrix with requires_grad = True
                loss.backward()

                # Apply the gradient calculate from last step to the matrix
                optimizer.step()
                # Add 1 more iteration count to learning rate scheduler


                # print statistics
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == l.data)

            for el in optimizer.param_groups:
                lr = el['lr']

            scheduler.step()
            val_corrects = 0
            val_loss = 0
            model.eval()

            inds = inds[b_s:]

            with torch.no_grad():
                for data in test_dataloader:
                    images, l = data
                    if torch.cuda.is_available():
                        images = images.to(device)
                        l = l.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_loss += criterion(outputs, l.long())
                    num = torch.sum(preds == l.data)
                    val_corrects += num

            print("  %2d   |    %.4f   |     %.4f       |   %.4f   |      %.4f     | %.5f "
                   %(q,running_loss/run_len,int(running_corrects)/run_len,val_loss/len(test_data),int(val_corrects)/len(test_data),lr))
            
            accuracy.append(int(val_corrects)/len(test_data))

            #early stopping


            q+=1
        torch.save(model, 'Data/'+model_type+'_model_cycled.pt')

        print('Finished Training')
        return(accuracy)

    ### In the case of SVHN this is the extra data
    def student_training_extra(augment = 'Rand', percentile = 0, model_type = 'ResNext' ,save = False, cycle = False, epochs=5):
        print(percentile)
        if cycle:
            model = torch.load('Data/'+model_type+'_model_cycled.pt')
        else:
            model = torch.load('Data/'+model_type+'_model.pt')

        transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([.5,.5,.5],[.5,.5,.5])
                                   ])

        extra_data = datasets.SVHN('Data/',split='extra',download=True,transform=transform)


        extra_loader = torch.utils.data.DataLoader(
                extra_data, batch_size=1024,shuffle=False)



        # Switch some layers (e.g., batch norm, dropout) to evaluation mode
        model.eval()
        # Turn off the autograd to save memory usage and speed up

        prediction_list =[]
        label_list = []
        with torch.no_grad():
            for data in extra_loader:
                images, labels = data
                if torch.cuda.is_available():
                    images = images.to(device)
                    labels = labels.to(device)
                outputs = model(images)
                predicted = outputs.data
                prediction_list.append(predicted.cpu())
                label_list.append(labels.cpu())

        p_list = torch.cat(prediction_list)
        p_list = p_list.numpy()

        """### Select threshold"""
        
        model = models.resnext50_32x4d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 10),
                         nn.Softmax())
        
        threshold = np.percentile(np.max(p_list,axis=1),percentile)
        print(threshold)

        if threshold < .5:
            indexes = list(range(p_list.shape[0]))
            labels = np.argmax(p_list,axis=1)

        else:
            print('here')
            labels = (p_list>=threshold).astype(int)*np.array([10,1,2,3,4,5,6,7,8,9])
            indexes = np.array(list(range(np.max(p_list.shape))))[list((np.sum(labels,axis=1)>.1))]
            labels = np.argmax(labels[indexes],axis=1)

        train_data = datasets.SVHN(
                root='Data/', split='train',
                download=True, transform=transform,
            )


        bs = int(73257/(73257+labels.shape[0])*1024)



        train_dataloader = torch.utils.data.DataLoader(
                train_data, batch_size=bs,
            )


        """## Stage 2.1: Reload Unlabeled (Extra) Data and apply *RandAugment*"""

        if augment == 'Rand':
            rand_transform = transforms.Compose([RandAugment(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([.5,.5,.5],[.5,.5,.5])
                                           ])
        elif augment == 'Traditional':
            rand_transform = transforms.Compose([transforms.RandomAffine(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize([.5,.5,.5],[.5,.5,.5])
                                           ])
        else:
            rand_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize([.5,.5,.5],[.5,.5,.5])
                                           ])

        aug_data = datasets.SVHN(
                root='Data/', split='extra',
                download=True, transform=rand_transform,
            )
        aug_dataloader = torch.utils.data.DataLoader(
                aug_data, batch_size=(1024-bs),
            )



        indices = np.array([False]*aug_data.data.shape[0])
        indices[indexes] = True

        #only need to run this cell when step 1 was skipped
        test_data = datasets.SVHN('Data/',split='test',download=True,transform=transform)
        test_dataloader = torch.utils.data.DataLoader(
                test_data, batch_size=1024
            )



        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05,momentum=.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        model.to(device)

        # Commented out IPython magic to ensure Python compatibility.
        print('Epoch  | Train Loss  |   Train Accuracy |  Val Loss  |  Val Accuracy   |    LR   ')
        print('_________________________________________________________________________________')

        q=0
        run_len = train_data.data.shape[0]+labels.shape[0]
        test_len = test_data.data.shape[0]
        prev  = 0
        accuracy = []
        while q < epochs:
            running_loss = 0.0
            running_corrects = 0
            model.train()
            inds = indices
            labs = torch.Tensor(labels)

            for i in enumerate(zip(train_dataloader,aug_dataloader)):
                

                # get the inputs; data is a list of [inputs, labels]
                b_s = min((1024-bs),i[1][1][0].shape[0])
                aug = i[1][1][0][inds[:b_s]]
                inputs = torch.cat([i[1][0][0],aug])

                l = torch.cat([i[1][0][1].float(),labs[:aug.shape[0]].float()])

                # randomize the batches
                ran = torch.randperm(l.shape[0])
                inputs = inputs[ran]
                l = l[ran]
                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    l = l.to(device)


                # zero the parameter gradients: Clean the gradient caclulated in the previous iteration
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                #print(outputs.shape)
                #print(labels.shape)

                loss = criterion(outputs, l.long())

                # Calculate gradient of matrix with requires_grad = True
                loss.backward()

                # Apply the gradient calculate from last step to the matrix
                optimizer.step()
                # Add 1 more iteration count to learning rate scheduler


                # print statistics
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == l.data)

            for el in optimizer.param_groups:
                lr = el['lr']

            scheduler.step()
            val_corrects = 0
            val_loss = 0
            model.eval()

            inds = inds[b_s:]

            with torch.no_grad():
                for data in test_dataloader:
                    images, l = data
                    if torch.cuda.is_available():
                        images = images.to(device)
                        l = l.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_loss += criterion(outputs, l.long())
                    num = torch.sum(preds == l.data)
                    val_corrects += num

            print("  %2d   |    %.4f   |     %.4f       |   %.4f   |      %.4f     | %.5f "
                   %(q,running_loss/run_len,int(running_corrects)/run_len,val_loss/len(test_data),int(val_corrects)/len(test_data),lr))
            
            accuracy.append(int(val_corrects)/len(test_data))

            #early stopping
            if accuracy[-1] > prev:
                torch.save(model, 'Data/'+model_type+'_model_cycled.pt')
                prev = accuracy[-1]
            else:
                q = epochs

            q+=1


        print('Finished Training')
        return(accuracy)

