from keras.layers import Conv2D, PReLU
from keras.layers import BatchNormalization, Activation

def Conv2D_BN(x, filters, filter_size, padding='same', strides=(1, 1)):
    x = Conv2D(filters, filter_size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = PReLU()(x)
    return x

def Conv2D_BN_Tanh(x, filters, filter_size, padding='same', strides=(1, 1)):
    x = Conv2D(filters, filter_size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation('tanh')(x)
    return x