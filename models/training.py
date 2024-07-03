#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from .transformer import spatial_transformer_network as stn
import keras.backend as K
import cv2
import os

def fit_models(model, training_generator_labelled_unlabelled, batch_size, epochs, num_train_images, writer = None):    
    steps_per_epoch  = num_train_images // batch_size
    n_steps = steps_per_epoch * epochs
    
    for i in range(n_steps):
        
        training_s_samples, training_s_samples_sdf, training_t_samples, training_t_samples_sdf = training_generator_labelled_unlabelled[i % steps_per_epoch]

        msg = model.train_on_batch([training_s_samples, training_s_samples_sdf, training_t_samples, training_t_samples_sdf], None)  #training_t_samples, prediction_skel, GT_skel, GT_samples
        summary = tf.Summary(value=[tf.Summary.Value(tag="model_loss", simple_value = msg), ])
        writer.add_summary(summary, i)

        if i % (10*steps_per_epoch) == 0 and i!=0: #epoch end 
            print("epoch:", i//steps_per_epoch, ",""msg :", msg)   
            result1, gradient1 = model.predict_on_batch([training_s_samples, training_s_samples_sdf, training_t_samples, training_t_samples_sdf])
            Imean   = (result1[0] * 0.5 + training_t_samples[0,28:-28,28:-28,:] * 0.5)   # exclude the edge areas
            Imean_ori = (training_s_samples[0,28:-28,28:-28,:] * 0.5 + training_t_samples[0,28:-28,28:-28,:] * 0.5)

            Imean   = np.concatenate([(Imean_ori*255).astype(np.uint8), (Imean*255).astype(np.uint8), (255*result1[0]).astype(np.uint8)], axis=1)

            cv2.imwrite("results.jpg", Imean)
            if os.path.isdir("weights") == False:
                os.mkdir("weights")
            model.save("weights/" + "model_best" + ".hdf")

    return None