from keras import backend as K
import tensorflow as tf
import os
import numpy as np
from scipy import misc
from keras.models import Model

'''
Compatible with tensorflow backend
'''
def FocalLoss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def DiceCoef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def DiceCoefLoss(y_true, y_pred):
    return 1. - DiceCoef(y_true, y_pred)

def GetHiddenLayers2D(model, input_data, store_folder):
    if isinstance(input_data, list):
        n_samples, image_row, image_col, image_channel = input_data[0].shape
        if n_samples > 1:
            new_input_data = [input_data[index][[0], ...] for index in range(len(input_data))]
        else:
            new_input_data = input_data
    else:
        n_samples, image_row, image_col, image_channel = input_data.shape
        if n_samples > 1:
            new_input_data = input_data[[0], ...]
        else:
            new_input_data = input_data

    for layer_index in range(len(model.layers)):
        hidden_layer = model.get_layer(index=layer_index)
        file_name = os.path.join(store_folder, str(layer_index) + '_' + hidden_layer.name)

        if not os.path.exists(file_name):
            os.makedirs(file_name)

        new_model = Model(inputs=model.input, outputs=hidden_layer.output)
        result = new_model.predict(new_input_data)
        result = np.squeeze(result)
        print(np.max(result))
        if result.ndim == 2:
            result = result[..., np.newaxis]
        for channel_index in range(np.shape(result)[-1]):
            feature_map = result[:, :, channel_index]
            misc.imsave(file_name + '/' + str(channel_index) + '.jpg', feature_map)

def GetHiddenLayers3D(model, input_data, store_folder):
    if isinstance(input_data, list):
        n_samples, image_row, image_col, image_slice, image_channel = input_data[0].shape
        if n_samples > 1:
            new_input_data = [input_data[index][[0], ...] for index in range(len(input_data))]
        else:
            new_input_data = input_data
    else:
        n_samples, image_row, image_col, image_slice, image_channel = input_data.shape
        if n_samples > 1:
            new_input_data = input_data[[0], ...]
        else:
            new_input_data = input_data

    for layer_index in range(len(model.layers)):
        hidden_layer = model.get_layer(index=layer_index)
        file_name = os.path.join(store_folder, str(layer_index) + '_' + hidden_layer.name)

        if not os.path.exists(file_name):
            os.makedirs(file_name)

        new_model = Model(inputs=model.input, outputs=hidden_layer.output)
        result = new_model.predict(new_input_data)
        result = np.squeeze(result)
        if result.ndim == 3:
            result = result[..., np.newaxis]
        for channel_index in range(np.shape(result)[-1]):
            for slice_index in range(np.shape(result)[-2]):
                feature_map = np.squeeze(result[:, :, slice_index, channel_index])
                misc.imsave(file_name + '/' + str(slice_index) + '-' + str(channel_index) + '.jpg', feature_map)
