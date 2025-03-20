import numpy as np
import torch


def load_dataset(name, embedding_type, num_reps, file_type='txt'):
    if file_type == 'txt':
        x_train = np.genfromtxt('./experiment_data/{}/x_train.txt'.format(name))
        x_test = np.genfromtxt('./experiment_data/{}/x_test.txt'.format(name))
        y_train = np.genfromtxt('./experiment_data/{}/y_train.txt'.format(name))
        y_test = np.genfromtxt('./experiment_data/{}/y_test.txt'.format(name))
    elif file_type == 'npy':
        x_train = np.load('./experiment_data/{}/x_train.npy'.format(name))
        x_test = np.load('./experiment_data/{}/x_test.npy'.format(name))
        y_train = np.load('./experiment_data/{}/y_train.npy'.format(name))
        y_test = np.load('./experiment_data/{}/y_test.npy'.format(name))        
        
    if name in ['pneumonia_2/pneumonia_emeds_49', 'breast_2/breast_emeds_49', 'oct_4/oct_emeds_49', 'retina_5/retina_emeds_49', 
                'derma_7/derma_emeds_49', 'blood_8/blood_emeds_49', 'path_9/path_emeds_49', 'organs_11/organs_emeds_49']:
        x_train = x_train[:, :49]
        x_test = x_test[:, :49]
    elif name in ['pneumonia_2/pneumonia_emeds_64', 'breast_2/breast_emeds_64', 'oct_4/oct_emeds_64', 'retina_5/retina_emeds_64', 
                  'derma_7/derma_emeds_64', 'blood_8/blood_emeds_64', 'path_9/path_emeds_64', 'organs_11/organs_emeds_64']:
        x_train = x_train[:, :64]
        x_test = x_test[:, :64]
    else:
        print('Dataset not supported!')
        return
    
    if len(x_train.shape) == 1:
        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
    
    if embedding_type == 'angle':
        x_train = np.mod(np.concatenate([x_train for i in range(num_reps)], 1), 2 * np.pi)
        x_test = np.mod(np.concatenate([x_test for i in range(num_reps)], 1), 2 * np.pi)   
    return x_train, y_train, x_test, y_test


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, embed_type, reps, train=True, reshape_labels=False, file_type='txt'):
        x_train, y_train, x_test, y_test = load_dataset(dataset_name, embed_type, reps, file_type)
        
        if reshape_labels and len(y_train.shape) == 1:
            y_train = y_train.reshape(len(y_train), 1)
            y_test = y_test.reshape(len(y_test), 1)
        
        if train:  
            inds = np.random.permutation(len(x_train))
            
            self.x_train = x_train[inds]
            self.y_train = y_train[inds]
            
            self.length = len(x_train)
        else:
            inds = np.random.permutation(len(x_test))
            
            self.x_train = x_test[inds]
            self.y_train = y_test[inds]
            
            self.length = len(x_test)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, ind):
        return self.x_train[ind], self.y_train[ind]
