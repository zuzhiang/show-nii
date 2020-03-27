from keras.layers import Conv2D, MaxPool2D, Input, UpSampling2D
from keras.layers import Activation, Concatenate, Add
from keras.models import Model
from MeDIT.CNNModel.Model.UsualLayer import Conv2D_BN, Conv2D_BN_Tanh

import numpy as np

def EncodingPart(input_shape, filters=64, blocks=3):
    inputs = Input(input_shape)
    x = inputs
    encoding_list = []
    for index in range(blocks):
        x_add = Conv2D_BN(x, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x_add, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x, filters * np.power(2, index), (1, 1))
        x = Add()([x, x_add])
        encoding_list.append(x)
        x = MaxPool2D()(x)
    return inputs, x, encoding_list

def DecodingPart(x, encoding_list, filters=64, blocks=3):
    for index in np.arange(blocks - 1, -1, -1):
        concat_list = []
        for temp in encoding_list:
            concat_list.append(temp[index])
        concat_list.append(x)

        concat = Concatenate(axis=-1)(concat_list)
        x_add = Conv2D_BN(concat, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x_add, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x, filters * np.power(2, index), (1, 1))
        x = Add()([x, x_add])
        if index > 0:
            x = UpSampling2D()(x)
    return x

def _DecodingPart_hierarchy(x, encoding_list, filters=64, blocks=3, hierarchy_level=2):
    output_list = []
    for index in np.arange(blocks - 1, -1, -1):
        concat_list = []
        for temp in encoding_list:
            concat_list.append(temp[index])
        concat_list.append(x)

        concat = Concatenate(axis=-1)(concat_list)
        x_add = Conv2D_BN(concat, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x_add, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x, filters * np.power(2, index), (1, 1))
        x = Add()([x, x_add])

        if index <= hierarchy_level:
            output_list.append(x)
        if index > 0:
            x = UpSampling2D()(x)
    return output_list

def ConnectionPart(encoding_output_list, filters=64, blocks=3):
    concat = Concatenate(axis=-1)(encoding_output_list)
    x_add = Conv2D_BN(concat, filters * np.power(2, blocks), (3, 3))
    x = Conv2D_BN(x_add, filters * np.power(2, blocks), (3, 3))
    x = Conv2D_BN(x, filters * np.power(2, blocks), (1, 1))
    x = Add()([x, x_add])
    x = UpSampling2D()(x)
    return x

def TrumpetNet(input_shape_list, filters=64, blocks=3):
    inputs_list, encoding_output_list, encoding_list = [], [], []
    for input_shape in input_shape_list:
        inputs, encoding_output, encoding = EncodingPart(input_shape, filters, blocks)
        inputs_list.append(inputs)
        encoding_output_list.append(encoding_output)
        encoding_list.append(encoding)

    connection = ConnectionPart(encoding_output_list, filters, blocks)

    x = DecodingPart(connection, encoding_list, filters, blocks)

    x = Conv2D_BN_Tanh(x, filters, (1, 1))
    x = Conv2D(1, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    return Model(inputs=inputs_list, outputs=x)

def TrumpetNet_hierarchy(input_shape_list, filters=64, blocks=3, hierarchy_level=2):
    inputs_list, encoding_output_list, encoding_list = [], [], []
    for input_shape in input_shape_list:
        inputs, encoding_output, encoding = EncodingPart(input_shape, filters, blocks)
        inputs_list.append(inputs)
        encoding_output_list.append(encoding_output)
        encoding_list.append(encoding)

    connection = ConnectionPart(encoding_output_list, filters, blocks)
    hierarchy_list = _DecodingPart_hierarchy(connection, encoding_list, filters=filters,
                                             blocks=blocks, hierarchy_level=hierarchy_level)
    output_list = []
    for output in hierarchy_list:
        x = Conv2D_BN_Tanh(output, filters, (1, 1))
        x = Conv2D(1, (1, 1), padding='same')(x)
        x = Activation('sigmoid')(x)
        output_list.append(x)

    return Model(inputs=inputs_list, outputs=output_list)