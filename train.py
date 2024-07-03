#!/usr/bin/env python3

import sys

from keras.optimizers import Adam

from models.models import deformation
from models.training import fit_models
from data.data import DataGenerator
from datetime import datetime

from keras import backend as K
import tensorflow as tf

K.clear_session()

#for tensorflow 1*
tf.device("/gpu:0") 
config = tf.ConfigProto() # for tf 2 tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

sess = tf.Session(config=config)  # for tf 2 tf.compat.v1.Session
init = tf.global_variables_initializer()
sess.run(init)
    
def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def main():
    batch_size = 1
    deformation_net = deformation(width=256)
   
    try:
        initial_epoch = int(sys.argv[1])
    except (IndexError, ValueError):
        initial_epoch = 0
    
    epochs = 10000
    opt = Adam(lr=0.002, decay = 0.0002)  #0.0005 for baseline
    deformation_net.compile(optimizer = opt, metrics = ['mse']*3)
    
    deformation_net.summary()
   
    data_location ="F://trainingdataset//deformation//" #
    with open(data_location + "/deformation_pair_pre_sdf.txt") as file:
        lines = file.readlines()
        training_samples= [line.rstrip() for line in lines]
    
    training_samples= training_samples[0:100] 
    num_train_images = len(training_samples)
    training_generator = DataGenerator(training_samples, data_location,  batch_size = batch_size, unlabelled = True, shuffle = True, specialscale = False)
    
    writer = tf.summary.FileWriter("logs/"+ datetime.now().strftime("%Y%m%d-%H%M%S"))
    print("start training! the number of training data is, ", num_train_images)
    histories = fit_models(deformation_net, training_generator, batch_size, epochs, num_train_images, writer) 

if __name__ == '__main__':
    main()
