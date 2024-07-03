#!/usr/bin/env python3

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_location, batch_size=32, unlabelled = False, shuffle=True, specialscale = False):
        'Initialization'
        
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.unlabelled = unlabelled
        self.shuffle = shuffle
        self.data_location = data_location
        self.on_epoch_end()
        self.specialscale = specialscale
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, X1, y1 = self.__data_generation(list_IDs_temp)

        return X, y, X1, y1 

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Generate data
        X = []
        SDF_X = []
        X1 = []
        SDF_X1 = []
        
        for i, ID in enumerate(list_IDs_temp):
            sheet_ID_lab1 = ID.split(",")[0]
            sheet_ID_lab1_sdf = ID.split(",")[1]
            sheet_ID_lab2 = ID.split(",")[2]
            sheet_ID_lab2_sdf = ID.split(",")[3]
            x = np.load(self.data_location + sheet_ID_lab1)
            X.append(x['arr_0'][:,:, 0:3]/255)
            
            sdf_x = np.load(self.data_location + sheet_ID_lab1_sdf)
            SDF_X.append(1 - sdf_x['arr_0']/60)
            x1 = np.load(self.data_location +  sheet_ID_lab2)
            X1.append(x1['arr_0'][:,:, 0:3]/255)
            sdf_x1 = np.load(self.data_location + sheet_ID_lab2_sdf)
            SDF_X1.append(1 - sdf_x1['arr_0']/60)

        X = np.asarray(X)
        SDF_X = np.asarray(SDF_X)
        X1 = np.asarray(X1)
        SDF_X1 = np.asarray(SDF_X1)
        return X, SDF_X, X1, SDF_X1
    