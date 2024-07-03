#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from keras.callbacks import Callback


class DecoderSnapshot(Callback):

    def __init__(self, step_size=200, latent_dim=128, decoder_index=-2):
        super().__init__()
        self._step_size = step_size
        self._steps = 0
        self._epoch = 0
        self._latent_dim = latent_dim
        self._decoder_index = decoder_index
        self._img_rows = 64
        self._img_cols = 64
        self._thread_pool = ThreadPoolExecutor(1)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._steps = 0

    def on_batch_begin(self, batch, logs=None):
        self._steps += 1
        if self._steps % self._step_size == 0:
            self.plot_images()
            
    def on_epoch_end(self, batch, logs=None):    
        self.plot_images()
        
    def plot_images(self, samples=16):
        decoder = self.model.layers[self._decoder_index]
        filename = 'imgs/generated_%d_%d.png' % (self._epoch, self._steps)
        z = np.random.normal(size=(samples, self._latent_dim))
        images = decoder.predict(z)
        #print("outout image shape: ", np.shape(images))
        self._thread_pool.submit(self.save_plot, images, filename)

    @staticmethod
    def save_plot(images, filename):
        images = (images + 1.) * 127.5
        images = np.clip(images, 0., 255.)
        images = images.astype('uint8')
        rows = []
        for i in range(0, len(images), 4):
            rows.append(np.concatenate(images[i:(i + 4), :, :, :], axis=0))
        plot = np.concatenate(rows, axis=1).squeeze()
        Image.fromarray(plot).save(filename)


class ModelsCheckpoint(Callback):

    def __init__(self, epoch_format, *models):
        super().__init__()
        self._epoch_format = epoch_format
        self._models = models

    def on_epoch_end(self, epoch, logs=None):
        suffix = self._epoch_format.format(epoch=epoch + 1, **logs)
        for model in self._models:
            model.save_weights("weights/"+ model.name + suffix)
