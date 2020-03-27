from keras.layers import Conv2D, MaxPool2D, Input, UpSampling2D
from keras.layers import Activation, Concatenate, Add, Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from MeDIT.CNNModel.Model.UsualLayer import Conv2D_BN, Conv2D_BN_Tanh

from copy import deepcopy
import numpy as np

K.set_image_data_format('channels_last')

smooth = 1

def _EncodingPart(inputs, filters=64, blocks=3):
    x = inputs
    encoding_list = []
    for index in range(blocks):
        x = Conv2D_BN(x, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x, filters * np.power(2, index), (1, 1))
        encoding_list.append(x)
        x = MaxPool2D()(x)
    return x, encoding_list

def _BottomPart(x, filters=64, blocks=3):
    x = Conv2D_BN(x, filters * np.power(2, blocks), (3, 3))
    x = Conv2D_BN(x, filters * np.power(2, blocks), (3, 3))
    x = Conv2D_BN(x, filters * np.power(2, blocks), (1, 1))
    x = UpSampling2D()(x)
    return x

def _DecodingPart_Output1(x, encoding_list, filters=64, blocks=3):
    for index in np.arange(blocks - 1, -1, -1):
        x = Concatenate(axis=-1)([x, encoding_list[index]])
        x = Conv2D_BN(x, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x, filters * np.power(2, index), (1, 1))
        if index > 0:
            x = UpSampling2D()(x)
    return x

def UNet2D(input_shape, filters=64, blocks=3):
    inputs = Input(input_shape)
    x, encoding = _EncodingPart(inputs, filters, blocks)
    x = _BottomPart(x, filters, blocks)
    x = _DecodingPart_Output1(x, encoding, filters, blocks)

    x = Conv2D_BN_Tanh(x, filters, (1, 1))
    x = Conv2D(1, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    return Model(inputs=inputs, outputs=x)

def _DecodingPart_hierarchy(x, encoding_list, filters=64, blocks=3, hierarchy_level=2):
    output_list = []
    for index in np.arange(blocks - 1, -1, -1):
        x = Concatenate(axis=-1)([x, encoding_list[index]])
        x = Conv2D_BN(x, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x, filters * np.power(2, index), (1, 1))
        if index <= hierarchy_level:
            output_list.append(x)
        if index > 0:
            x = UpSampling2D()(x)
    return output_list

def UNet2D_hierarchy(input_shape, filters=64, blocks=3, hierarchy_level=2):
    inputs = Input(input_shape)
    x, encoding = _EncodingPart(inputs, filters, blocks)
    x = _BottomPart(x, filters, blocks)

    outputs = []
    hierarchy_list = _DecodingPart_hierarchy(x, encoding, filters, blocks, hierarchy_level)
    for output in hierarchy_list:
        x = Conv2D_BN_Tanh(output, filters, (1, 1))
        x = Conv2D(1, (1, 1), padding='same')(x)
        x = Activation('sigmoid')(x)
        outputs.append(x)

    return Model(inputs=inputs, outputs=outputs)
