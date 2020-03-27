import tensorflow as tf
from keras import backend as K

def FocalLoss(y_true, y_pred, gamma=2., alpha=.25):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    eps = 1e-12
    y_pred = K.clip(y_pred, eps, 1. - eps)

    pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
    pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1. - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def DiceCoef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_pred_f * y_true_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def DiceLoss(y_true, y_pred):
    return 1 - DiceCoef(y_true, y_pred)