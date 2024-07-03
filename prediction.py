#!/usr/bin/env python3

# from UNETGAN.models import create_models, build_graph, build_graph_all, deformation
from models.models import deformation
import numpy as np
import cv2
import tensorflow as tf
import keras.backend as K

deform_net = deformation(width=256)
deform_net.load_weights('weights/model_best.hdf')

imagesize = 200
# vae = q[-2]


def waspGridSpatialIntegral(grid, **kwargs):
    """
    Deformation AutoEncoder: https://github.com/zhixinshu/DeformingAutoencoders-pytorch/blob/
    """
    H = 256
    grid = grid* 2/255  # allow scaling = 2
    A = []
    B = []
    for i in range(0,H):
        A.append(K.sum(grid[:,0:i,:,0], axis = 1))
        B.append(K.sum(grid[:,:,0:i,1], axis = 2))
    A = tf.stack(A)
    A = tf.transpose(A, perm = [1,0,2])
    B = tf.stack(B)
    B = tf.transpose(B, perm = [1,2,0])

    integral = tf.stack([B,A], axis = -1)
    integral = integral - 1

    # intergal = K.max([K.min([1,integral]), -1])
    return integral

def predict_sheet(model, sheet0, sheet1, output_sheet_location, output_deform_location, batch_size = 1, resolution = 1.25, padding = 0, img_size=imagesize):
    
    ds = gdal.Open(sheet0)
    ds1 = gdal.Open(sheet1)
        
    transform_in = ds.GetGeoTransform()
    transform_out = (transform_in[0], transform_in[1], transform_in[2], transform_in[3], transform_in[4], transform_in[5])

    
    bands = []
    bands1 = []
    shape = np.shape(ds.ReadAsArray())
    print('shape of ds', shape)
    
    for i in range(shape[0]):
        bands.append(ds.GetRasterBand(i+1).ReadAsArray())
        bands1.append(ds1.GetRasterBand(i+1).ReadAsArray())
    if shape[0] ==3:
        bands.append(255*np.ones([shape[1],shape[2]]))
        bands1.append(255*np.ones([shape[1],shape[2]]))
    sheet0 = np.dstack(tuple(bands))
    sheet1 = np.dstack(tuple(bands1))
    sheet = sheet0
    # make sure that the sheet matches the model extent parameters
    excess_x = sheet.shape[1] % img_size
    excess_y = sheet.shape[0] % img_size
    
    if not excess_x == 0:
        additional_padding_x = img_size - excess_x
    else:
        additional_padding_x = 0
    
    if not excess_y == 0:
        additional_padding_y = img_size - excess_y
    else:
        additional_padding_y = 0

    xa = np.linspace(-1.0, 1.0, 256)  #normalized 2D grid of width and height
    ya = np.linspace(-1.0, 1.0, 256)
    x_t, y_t = np.meshgrid(xa, ya)
    base_grid = np.stack([x_t, y_t])
    base_grid = np.transpose(base_grid, [1,2,0])
    base_grid = np.expand_dims(base_grid, axis=0)
    base_grid = np.tile(base_grid, [batch_size, 1, 1, 1]) # [1,256,256,2]

    sheet_template = np.zeros((sheet.shape[0] + additional_padding_y, sheet.shape[1] + additional_padding_x, 3), np.float32)
    deform_template = np.zeros((sheet.shape[0] + additional_padding_y, sheet.shape[1] + additional_padding_x, 2), np.float32)    
        
    sheet_extended = np.zeros((int(sheet.shape[0] + 2*padding + additional_padding_y), int(sheet.shape[1] + 2*padding + additional_padding_x), 4), np.float32)
    sheet_extended[padding:-padding-additional_padding_y, padding:-padding-additional_padding_x,:] = sheet0

    sheet_extended1 = np.zeros((int(sheet.shape[0] + 2*padding + additional_padding_y), int(sheet.shape[1] + 2*padding + additional_padding_x), 4), np.float32)
    sheet_extended1[padding:-padding-additional_padding_y, padding:-padding-additional_padding_x,:] = sheet1

    x_count = int((sheet_extended.shape[1] - padding * 2) / img_size)
    y_count = int((sheet_extended.shape[0] - padding * 2) / img_size)
    print("x_count:", x_count, "y_count:", y_count, "padding:", padding, np.shape(sheet_extended), img_size)

    for y in range(y_count):
        print(str(y) + str("/") + str(y_count))
        for x in range(x_count):
            y_start = int(y * img_size)
            y_end = int(y * img_size + img_size)
            x_start = int(x * img_size)
            x_end = int(x * img_size + img_size)
            sub_img = sheet_extended[y_start:y_end + 2 * padding, x_start:x_end + 2 * padding] / 255
            sub_img1 = sheet_extended1[y_start:y_end + 2 * padding, x_start:x_end + 2 * padding] / 255
            
            cv2.imwrite('img.jpg', sub_img*255)
            cv2.imwrite('img1.jpg', sub_img1*255)
            sub_img_g = cv2.imread('img.jpg', 0)
            sub_img1_g = cv2.imread('img1.jpg', 0)
            edges = cv2.Canny(sub_img_g, 200, 400)
            sdf = cv2.distanceTransform(255 - edges, distanceType=cv2.DIST_L2, maskSize=0)
            edges1 = cv2.Canny(sub_img1_g, 200, 400)
            sdf1 = cv2.distanceTransform(255 - edges1, distanceType=cv2.DIST_L2, maskSize=0)
            
            sub_img_expanded = np.expand_dims(sub_img, axis=0)
            sub_img_expanded1 = np.expand_dims(sub_img1, axis=0)
            
            """
            change here to make the input only have 3 channels
            """
            sub_img_expanded = sub_img_expanded[:,:,:,0:3]
            sub_img_expanded1 = sub_img_expanded1[:,:,:,0:3]
            sdf_expanded = np.expand_dims(sdf, axis = 0)
            sdf1_expanded = np.expand_dims(sdf1, axis = 0)
            sdf_expanded = 1 - np.expand_dims(sdf_expanded, axis = -1)/60
            sdf1_expanded = 1 - np.expand_dims(sdf1_expanded, axis = -1)/60
            
            
            Y_pred, gradients = model.predict([sub_img_expanded, sdf_expanded, sub_img_expanded, sdf1_expanded], batch_size)
            gradients = tf.cast(gradients, 'float32')
            gradients = K.eval(gradients)

            sheet_template[y_start:y_end, x_start:x_end, :] = Y_pred.copy()#[:,28:-28,28:-28]
            deform_template[y_start:y_end, x_start:x_end, :] = gradients.copy()[:,28:-28,28:-28]

    
    sheet_out = sheet_template[0:sheet.shape[0], 0:sheet.shape[1], :]
    deform_out = deform_template[0:sheet.shape[0], 0:sheet.shape[1], :]
    print('the size of sheet out', np.shape(sheet_out))

    # write raster
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_sheet_location, sheet_out.shape[1], sheet_out.shape[0], 3, 
                            gdal.GDT_Float32, ['COMPRESS=LZW'])
    outdata.SetGeoTransform(transform_out)
    outdata.SetProjection(ds.GetProjection())
    sheet_out[sheet_out<=0.05] = 0

    # outdata.GetRasterBand(1).SetNoDataValue(nodata)
    outdata_d = driver.Create(output_deform_location, sheet_out.shape[1], sheet_out.shape[0], 2, 
                            gdal.GDT_Float32, ['COMPRESS=LZW'])
    outdata_d.SetGeoTransform(transform_out)
    outdata_d.SetProjection(ds.GetProjection())
    sheet_out[sheet_out<=0.05] = 0
    for i in range(3):
        outdata.GetRasterBand(i+1).WriteArray(np.squeeze(sheet_out[:, :, i]))
    for i in range(2):
        outdata_d.GetRasterBand(i+1).WriteArray(np.squeeze(deform_out[:, :, i]))
    outdata.FlushCache()
    outdata = None
    outdata_d.FlushCache()
    outdata_d = None
    ds = None   


sheet1 = 'F://rgb_TA_318_1899.tif'
sheet0 = 'F://rgb_TA_318_1874.tif'
vector = "F://rgb_TA_318_1874_stream.tif"
target_dir = "prediction//"
output_sheet_location = target_dir + "rgb_TA_318_1874_prediction.tif"
output_deform_location = target_dir + "rgb_TA_318_1874_gradient.tif"
predict_sheet(deform_net, sheet0, sheet1, output_sheet_location, output_deform_location, batch_size = 1, resolution = 1.25, padding = 28, img_size=imagesize)
