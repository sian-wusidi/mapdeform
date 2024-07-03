#!/usr/bin/env python3
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Conv2DTranspose, \
    Lambda,  Activation, MaxPooling2D, Dropout, AveragePooling2D, DepthwiseConv2D, Concatenate, Cropping2D
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils
import tensorflow as tf
from .transformer import waspGridSpatialIntegral

def rgb2gray(rgb):
    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = 1 - gray
    gray = tf.expand_dims(gray, axis = -1)
    return gray

def deformation(width=256, wdecay=1e-5, bn_mom=0.9, bn_eps=1e-6):
    image_shape = (width, width, 3)
    img = Input(shape=image_shape, name='input_image')
    img_sdf = Input(shape=(width, width, 1), name='input_image_sdf')
    img1 = Input(shape=image_shape, name='input_image1')
    img1_sdf = Input(shape=(width, width, 1), name='input1_image_sdf')
    num_filters = 16
    inputs = Concatenate()([img, img_sdf, img1, img1_sdf])
    # inputs = Concatenate()([img, img1])
    OS = 16
    atrous_rates = (3, 6, 9)
    # 128   input - [batchsize,128,128,4] 
    c1 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    

    # 64
    c2 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    
    # 32
    c3 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    
    # 16
    c4 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
   
    
    # 8
    c5 = Conv2D(num_filters * 16, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(num_filters * 16, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    # ASPP
    # Image Feature branch
    #out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = AveragePooling2D(pool_size=(int(np.ceil(image_shape[0] / OS)), int(np.ceil(image_shape[1] / OS))))(c5)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False)(b4)
    b4 = BatchNormalization(epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((int(np.ceil(image_shape[0] / OS)), int(np.ceil(image_shape[1] / OS))))(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False)(c5)
    b0 = BatchNormalization(epsilon=1e-5)(b0)
    b0 = Activation('relu')(b0)
    #rate = 3 (6)
    b1 = SepConv_BN(c5, 256, rate=atrous_rates[0], depth_activation=False, epsilon=1e-5)
    # rate = 6 (12)
    b2 = SepConv_BN(c5, 256, rate=atrous_rates[1], depth_activation=False, epsilon=1e-5)
    # rate = 9 (18)
    b3 = SepConv_BN(c5, 256, rate=atrous_rates[2], depth_activation=False, epsilon=1e-5)

    # concatenate ASPP branches & project
    c5 = Concatenate()([b4, b0, b1, b2, b3])
    
    # simple 1x1 again
    c5 = Conv2D(256, (1, 1), padding='same', use_bias=False)(c5) 


    c5 = BatchNormalization(epsilon=1e-5)(c5)
    c5 = Activation('relu')(c5)
    c5 = Dropout(0.1)(c5)

    # 16
    u6 = Conv2DTranspose(num_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    # 32
    u7 = Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    
    # 64
    u8 = Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    
    # 128
    u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)
    
    
    finalConv = Conv2D(2, (1, 1), activation='linear')(c9) #spatial gradient  but allow scaling in waspGridSpatialIntegral
    
    finalConv = Lambda(lambda x: K.minimum(x,4))(finalConv)  # hard tanh cutter
    output_grid = Lambda(waspGridSpatialIntegral)(finalConv)   # direct grid for resampling, minus 1 to make it start with -1

    deformed_img = Lambda(bilinear_sampler1)([img, output_grid]) 
    deformed_img_sdf = Lambda(bilinear_sampler1)([img_sdf, output_grid]) 
    
    deformed_img = Cropping2D(cropping=(28, 28), data_format=None, name="cropped2")(deformed_img)
    deformed_img_sdf = Cropping2D(cropping=(28, 28), data_format=None, name="cropped3")(deformed_img_sdf)
    img1_sdfc = Cropping2D(cropping=(28, 28), data_format=None, name="cropped4")(img1_sdf)
    
    model = Model(inputs = [img, img_sdf, img1, img1_sdf], outputs = [deformed_img, output_grid]  , name = "deformation")
    
    model.add_loss(60*tf.reduce_mean(K.abs((deformed_img_sdf - img1_sdfc)*img1_sdfc)))

    finalConv = Cropping2D(cropping=(28, 28), data_format=None, name="cropped5")(finalConv)
    # model.add_loss(10*tf.reduce_mean(K.abs(deformed_img - img1)))
    # smoothness loss 
    # model.add_loss(1e-6*K.sum(finalConv)) #1e-6
    # identity loss
    # model.add_loss(K.abs(K.mean(finalConv) - 1))


    return model




def SepConv_BN(x, filters, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
            the code is based on keras implementation of deeplabV3+ https://github.com/bonlime/keras-deeplab-v3-plus
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('elu')(x)
        
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate), padding=depth_padding, use_bias=False)(x) #depthwise
    x = BatchNormalization(epsilon=epsilon)(x)
    
    if depth_activation:
        x = Activation('elu')(x)
        
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x) #pointwise
    x = BatchNormalization(epsilon=epsilon)(x)
    
    if depth_activation:
        x = Activation('elu')(x)

    return x

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.image_data_format()
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0], inputs.shape[2] * self.upsampling[1]), align_corners=True)
            #return K.tensorflow_backend.tf.image.resize(inputs, (inputs.shape[1] * self.upsampling[0], inputs.shape[2] * self.upsampling[1]))
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0], self.output_size[1]), align_corners=True)
            #return K.tensorflow_backend.tfimage.resize(inputs, (self.output_size[0], self.output_size[1]))
    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
def SmoothReg(inputs,GT):
    inputs_scalor = K.expand_dims(K.sqrt(K.sum(inputs**2, axis = 3)), -1)  + 1e-6
    inputs = inputs / inputs_scalor
    
    
    dy_inputs = K.sum(np.abs(inputs[:,1:,:,:] - inputs[:,:-1,:,:])*GT[:,1:,:,:]*GT[:,:-1,:,:], axis = 3)  #.mean(axis = 3)  
    dx_inputs  = K.sum(np.abs(inputs[:,:,1:,:] - inputs[:,:,:-1,:])*GT[:,:,1:,:]*GT[:,:,:-1,:], axis = 3) #[batch,height,width,channel] ->[batch,height,width]
    
    dxdy_inputs = K.sum(np.abs(inputs[:,1:,1:,:] - inputs[:,:-1,:-1,:])*GT[:,1:,1:,:]*GT[:,:-1,:-1,:], axis = 3)
    dydx_inputs = K.sum(np.abs(inputs[:,-1:,1:,:] - inputs[:,:1,:-1,:])*GT[:,-1:,1:,:]*GT[:,:1,:-1,:], axis = 3)
        
    dy_loss = K.mean(dy_inputs)
    dx_loss = K.mean(dx_inputs)
    dxdy_loss = K.mean(dxdy_inputs)
    dydx_loss = K.mean(dydx_inputs)
    
    smooth_loss = 0.5*dy_loss + 0.5*dx_loss + 0.5* dxdy_loss + 0.5*dydx_loss
    
    return smooth_loss
def get_pixel_value(img, x, y):
    """
    https://github.com/kevinzakka/spatial-transformer-network/
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def nearestsample(inputs):
    """
    adapt from:
    https://github.com/kevinzakka/spatial-transformer-network/
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    img = inputs[0]
    grid = inputs[1]
    print("shape of imagge:", tf.shape(img))
    H = tf.shape(img)[1]
    W = tf.shape(img)[2] 
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')
    x = grid[:,:,:,0] #[-1,1]
    y = grid[:,:,:,1]

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32') # B, W, H
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0) #B,H,W,3
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    w_max = K.max([wa, wb, wc, wd])
    
    wa = (wa == w_max)
    wb = (wb == w_max)
    wc = (wc == w_max)
    wd = (wd == w_max)
    
    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out
        
        
def bilinear_sampler1(inputs):
    """
    adapt from:
    https://github.com/kevinzakka/spatial-transformer-network/
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    img = inputs[0]
    grid = inputs[1]
    # print("shape of imagge:", tf.shape(img))
    H = tf.shape(img)[1]
    W = tf.shape(img)[2] 
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')
    x = grid[:,:,:,0] #[-1,1]
    y = grid[:,:,:,1]

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32') # B, W, H
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0) #B,H,W,3
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out
        
def bilinear_sampler(img, grid):
    """
    adapt from:
    https://github.com/kevinzakka/spatial-transformer-network/
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    print("shape of imagge:", tf.shape(img))
    H = tf.shape(img)[1]
    W = tf.shape(img)[2] 
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')
    x = grid[:,:,:,0] #[-1,1]
    y = grid[:,:,:,1]

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32') # B, W, H
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0) #B,H,W,3
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out
# def f(x, x1, classifier, out_shape):
    # v = tf.Variable(K.zeros([64, out_shape[0], out_shape[1], out_shape[2]]))
    # ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    # tb = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    # i = 0
    # for r in range(8): 
        # for c in range(8):
            # v.assign(classifier(x[r*256:(r+1)*256,c*256:(c+1)*256,:])[:, :out_shape[0], :out_shape[1], :out_shape[2]])
            # ta = ta.write(i, v)
            # v.assign
            # (classifier(x1[r*256:(r+1)*256,c*256:(c+1)*256,:])[:, :out_shape[0], :out_shape[1], :out_shape[2]])
            # tb = tb.write(i, v)
            # i = i+1
    # return ta.stack(), tb.stack()
    
# # def predict_ensemble(args):
    # # x, classifier, image_shape = args
    # # ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    # # tb = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    # # for i in range(image_shape[0]):
        # # # v.assign(classifier(x[:,i,:,:,:]))
        # # ta = ta.write(i, classifier(x[:,i,:,:,:]))
        # # # v.assign(classifier(x1[:,i,:,:,:]))
        # # tb = tb.write(i, classifier(x1[:,i,:,:,:]))
    # # x_prediction = ta.stack()
    # # x_prediction = tf.transpose(x_prediction, [1,0,2,3,4])
    # # x1_prediction = tb.stack()
    # # x1_prediction = tf.transpose(x1_prediction, [1,0,2,3,4])
    # # return x_prediction, x1_prediction
# # def all_predictions(args):
    # # predictions = args
    # # x_prediction = Concatenate(axis = -1)(predictions)
    # # x_prediction = Reshape((256, 256, 4, 16))(x_prediction)
    # # x_prediction = Permute((1,2,4,3), input_shape=(256, 256, 4, 16))(x_prediction)
    # # return x_prediction
# def all_predictions(args):
    # predictions = args
    # x_prediction = Concatenate(axis = 0)(predictions)
    # x_prediction = Lambda(lambda x: K.reshape(x, (2,16,256,256,4)))(x_prediction)#Reshape((256, 256, 4, 16))(x_prediction)  the shape to concatanate should be put in the col
    # # x_prediction = Permute((1,2,4,3), input_shape=(256, 256, 4, 16))(x_prediction)
    # return x_prediction
# def build_graph_scale(classifier, discriminator, batch_size):
    # image_shape = (K.int_shape(discriminator.input)[1], K.int_shape(discriminator.input)[2], K.int_shape(discriminator.input)[3],3)
    # # image_shape = (K.int_shape(discriminator.input)[2], K.int_shape(discriminator.input)[3],3)
    # out_shape = (K.int_shape(discriminator.input)[1], K.int_shape(classifier.output)[2], K.int_shape(classifier.output)[3],4)
    # # out_shape = (K.int_shape(classifier.output)[1], K.int_shape(classifier.output)[2],4)
    # # latent_shape = K.int_shape(decoder.input)[1:]
    # print('input_shape:', image_shape, 'out_shape:', out_shape)
    # # sampler = Lambda(_sampling, output_shape=latent_shape, name='sampler')
    
    # # # Inputs
    # x = Input(shape=image_shape, name='input_image_labelled') #16, 256, 256 ,4
    # x1 = Input(shape=image_shape, name='input_image_unlabelled')

    # # slice_lr = Lambda(lambda x: x[0,:,:,:,:])
    # # # slice_lr = Lambda(lambda x: K.reshape(x, (K.int_shape(x)[0]*K.int_shape(x)[1]), K.int_shape(x)[2], K.int_shape(x)[3], K.int_shape(x)[4]))
    # # prediction_x = classifier(slice_lr(x))
    # # prediction_x1 = classifier(slice_lr(x1))
    # # # print("x_prediction:", x_prediction, x1_prediction)
    # # expand_lr = Lambda(lambda x: K.expand_dims(x, axis = 0))
    # # x_prediction = expand_lr(prediction_x)
    # # x1_prediction = expand_lr(prediction_x1)
    # prediction_x = []
    # prediction_x1 = []
    # for i in range(batch_size):
        # slice_lr = Lambda(lambda x: x[i,:,:,:])
        # prediction_lr = classifier(slice_lr(x))
        # prediction_lr1 = classifier(slice_lr(x1))
        # prediction_x.append(prediction_lr)
        # prediction_x1.append(prediction_lr1)
    # # x_prediction = Lambda(all_predictions)(prediction_x)
    # # x1_prediction = Lambda(all_predictions)(prediction_x1)
    # x_prediction = Lambda(lambda x: tf.convert_to_tensor(x))(prediction_x)
    # x1_prediction = Lambda(lambda x: tf.convert_to_tensor(x))(prediction_x1)
    # # slice_lr_0 =  Lambda(lambda x: x[0,:,:,:])
    # # slice_lr_1 =  Lambda(lambda x: x[1,:,:,:])
    # # x_prediction_0 = classifier(slice_lr_0(x))
    # # x_prediction_1 = classifier(slice_lr_1(x))
    # # print("x_prediction_0:", x_prediction_0)
    # # x_prediction = Concatenate(axis = -1)([x_prediction_0,x_prediction_1])
    # # x_prediction = Reshape((K.int_shape(discriminator.input)[1], K.int_shape(discriminator.input)[2], 4 , 2))(x_prediction)
    # # x_prediction = Permute((1,2,4,3), input_shape=(K.int_shape(discriminator.input)[1], K.int_shape(discriminator.input)[2], 2, 4))(x_prediction)
    
    
    # # x_prediction = K.stack(x_prediction_0, x_prediction_1)
    # # x_prediction = tf.transpose(x_prediction, [1,0,2,3,4])
    # #x_prediction = classifier(x)
    # # x_prediction = tf.Variable(K.zeros([1, 16, image_shape[1], image_shape[2], 4]))
    # # x_prediction = x_prediction.assign(ta.stack())
    
    # # x_prediction = tf.einsum("ij...->ji...", x_prediction)
    # print("x_prediction", x_prediction, x1_prediction)
    # # x_prediction_pooling = MaxPooling2D((4, 4))(x_prediction)
    # dis_x_prediction = discriminator(x_prediction)
    # print("dis_x_prediction:", dis_x_prediction, tf.gradients(dis_x_prediction, x))
    # # x1_prediction = classifier(x1)
    
    # # x1_prediction = tf.Variable(K.zeros([1, 16, image_shape[1], image_shape[2], 4]))
    # # x1_prediction.assign(tb.stack())
    
    # # x1_prediction = classifier(x1)
    # # x1_prediction_pooling = MaxPooling2D((4, 4))(x1_prediction)
    # dis_x1_prediction = discriminator(x1_prediction)
    
    # # print("x1_prediction", x1_prediction, tf.gradients(x1_prediction, x1))
    # # # x_prediction_pooling = MaxPooling2D((4, 4))(x_prediction)
    # # print("dis_x1_prediction:", dis_x1_prediction, tf.gradients(dis_x1_prediction, x1))
    # combined = Model([x, x1], [dis_x_prediction, dis_x1_prediction], name='clas')
    # # combined = Model(x, dis_x_prediction, name='clas')
    # return combined
