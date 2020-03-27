import os
import h5py
from copy import deepcopy
from random import shuffle
import numpy as np
import cv2
from MeDIT.SaveAndLoad import LoadH5InfoForGenerate, LoadH5
from MeDIT.ArrayProcess import ExtractPatch
from MeDIT.DataAugmentor import AugmentParametersGenerator, DataAugmentor2D, random_2d_augment

def GetKeysFromStoreFolder(case_folder):
    key_list = []
    for case in os.listdir(case_folder):
        case_path = os.path.join(case_folder, case)
        if case_path.endswith('.h5'):
            file = h5py.File(case_path, 'r')
            key_list = list(file.keys())
            file.close()
            break

    if len(key_list) == 0:
        print('there is no h5 file, check the folder path: ', case_folder)
        return []

    return key_list

def _GetInputOutputNumber(case_folder):
    key_list = GetKeysFromStoreFolder(case_folder)

    input_number, output_number = 0, 0
    for key in key_list:
        if 'input' in key:
            input_number += 1
        elif 'output' in key:
            output_number += 1
        else:
            print(key)

    if input_number > 0 and output_number > 0:
        return input_number, output_number
    else:
        print('Lack input or output: ', case_folder)
        return 0, 0

def _AugmentDataList2D(data_list, augmentor):
    aug_data_list = []

    for data in data_list:
        aug_data = np.zeros_like(data)
        for channel_index in range(data.shape[-1]):
            aug_data[..., channel_index] = augmentor.Execute(data[..., channel_index])
        aug_data_list.append(aug_data)
    return aug_data_list

def _CropDataList2D(data_list, input_shape):
    crop_data_list = []
    for data in data_list:
        if data.ndim == 2:
            data = data[..., np.newaxis]
        crop_data = np.zeros((input_shape[0], input_shape[1], data.shape[-1]))
        for channel_index in range(data.shape[-1]):
            crop_data[..., channel_index], _ = ExtractPatch(data[..., channel_index], input_shape[:2])
        crop_data_list.append(crop_data)
    return crop_data_list

def _AddOneSample(all_data_list, one_data_list):
    if len(all_data_list) != len(one_data_list):
        print('the number of all samples and the number of one data list is not same: ', len(all_data_list), len(one_data_list))
        return

    for index in range(len(one_data_list)):
        all_data_list[index].append(one_data_list[index])

def _MakeKerasFormat(data_list, dtype=np.float32):
    if len(data_list) == 1:
        return np.asarray(data_list[0], dtype=dtype)
    else:
        format_data_list = []
        for one_input in data_list:
            format_data_list.append(np.asarray(one_input, dtype=dtype))
        return format_data_list

##############################################

def ImageInImageOut2D(root_folder, input_shape, batch_size=8, augment_param={}):
    input_number, output_number = _GetInputOutputNumber(root_folder)
    case_list = os.listdir(root_folder)

    input_list = [[] for index in range(input_number)]
    output_list = [[] for index in range(output_number)]

    param_generator = AugmentParametersGenerator()
    augmentor = DataAugmentor2D()

    while True:
        shuffle(case_list)
        for case in case_list:
            case_path = os.path.join(root_folder, case)
            if not case_path.endswith('.h5'):
                continue

            input_data_list, output_data_list = [], []
            file = h5py.File(case_path, 'r')
            for input_number_index in range(input_number):
                temp_data = np.asarray(file['input_' + str(input_number_index)])
                if temp_data.ndim == 2:
                    temp_data = temp_data[..., np.newaxis]
                input_data_list.append(temp_data)
            for output_number_index in range(output_number):
                temp_data = np.asarray(file['output_' + str(output_number_index)])
                if temp_data.ndim == 2:
                    temp_data = temp_data[..., np.newaxis]
                output_data_list.append(temp_data)

            param_generator.RandomParameters(augment_param)
            augmentor.SetParameter(param_generator.GetRandomParametersDict())

            input_data_list = _AugmentDataList2D(input_data_list, augmentor)
            output_data_list = _AugmentDataList2D(output_data_list, augmentor)

            input_data_list = _CropDataList2D(input_data_list, input_shape)
            output_data_list = _CropDataList2D(output_data_list, input_shape)

            _AddOneSample(input_list, input_data_list)
            _AddOneSample(output_list, output_data_list)

            if len(input_list[0]) >= batch_size:
                inputs = _MakeKerasFormat(input_list)
                outputs = _MakeKerasFormat(output_list)
                yield inputs, outputs
                input_list = [[] for index in range(input_number)]
                output_list = [[] for index in range(output_number)]

def ImageInImageOut2DTest(root_folder, input_shape):
    from MeDIT.Visualization import LoadWaitBar
    input_number, output_number = _GetInputOutputNumber(root_folder)
    case_list = os.listdir(root_folder)
    case_list = [case for case in case_list if case.endswith('.h5')]

    input_list = [[] for index in range(input_number)]
    output_list = [[] for index in range(output_number)]

    for case in case_list:
        LoadWaitBar(len(case_list), case_list.index(case))
        case_path = os.path.join(root_folder, case)

        input_data_list, output_data_list = [], []
        file = h5py.File(case_path, 'r')
        for input_number_index in range(input_number):
            input_data_list.append(np.asarray(file['input_' + str(input_number_index)]))
        for output_number_index in range(output_number):
            output_data_list.append(np.asarray(file['output_' + str(output_number_index)]))

        input_data_list = _CropDataList2D(input_data_list, input_shape)
        output_data_list = _CropDataList2D(output_data_list, input_shape)

        _AddOneSample(input_list, input_data_list)
        _AddOneSample(output_list, output_data_list)

        inputs = _MakeKerasFormat(input_list)
        outputs = _MakeKerasFormat(output_list)

    return inputs, outputs, case_list

def _TestOutput1():
    input_list, output_list = ImageInImageOut2D(r'c:\SharedFolder\ProstateSegment\Input_1_Ouput_1\testing',
                                                input_shape=[160, 160], batch_size=1, augment_param=random_2d_augment)
    from MeDIT.Visualization import Imshow3DArray

    if isinstance(input_list, list):
        input1, input2, input3 = input_list[0], input_list[1], input_list[2]
        input1 = np.concatenate((input1[..., 0], input1[..., 1], input1[..., 2]), axis=2)
        input2 = np.concatenate((input2[..., 0], input2[..., 1], input2[..., 2]), axis=2)
        input3 = np.concatenate((input3[..., 0], input3[..., 1], input3[..., 2]), axis=2)
        input1 = np.transpose(input1, (1, 2, 0))
        input2 = np.transpose(input2, (1, 2, 0))
        input3 = np.transpose(input3, (1, 2, 0))
        input_show = np.concatenate((input1, input2, input3), axis=0)
        roi = np.squeeze(output_list)
        roi = np.concatenate((np.zeros_like(roi), roi, np.zeros_like(roi)), axis=2)
        roi = np.concatenate((roi, roi, roi), axis=1)
        roi = np.transpose(roi, (1, 2, 0))

        Imshow3DArray(input_show, ROI=np.asarray(roi, dtype=np.uint8))
    elif not isinstance(output_list, list):
        show_input = np.transpose(np.squeeze(input_list), (1, 2, 0))
        show_roi = np.transpose(np.squeeze(output_list), (1, 2, 0))
        Imshow3DArray(show_input, ROI=np.asarray(show_roi, dtype=np.uint8))

##############################################

def ImageInMultiROIOut2D(root_folder, input_shape, batch_size=8, hierarchical_level=0, augment_param={}):
    input_number, output_number = _GetInputOutputNumber(root_folder)
    assert(output_number == 1)
    case_list = os.listdir(root_folder)

    input_list = [[] for index in range(input_number)]
    output_list = [[] for index in range(hierarchical_level + 1)]

    param_generator = AugmentParametersGenerator()
    augmentor = DataAugmentor2D()

    while True:
        shuffle(case_list)
        for case in case_list:
            case_path = os.path.join(root_folder, case)
            if not case_path.endswith('.h5'):
                continue

            input_data_list, output_data_list = [], []
            file = h5py.File(case_path, 'r')
            for input_number_index in range(input_number):
                temp_data = np.asarray(file['input_' + str(input_number_index)])
                if temp_data.ndim == 2:
                    temp_data = temp_data[..., np.newaxis]
                input_data_list.append(temp_data)

            one_roi = np.asarray(file['output_0'])

            param_generator.RandomParameters(augment_param)
            augmentor.SetParameter(param_generator.GetRandomParametersDict())

            input_data_list = _AugmentDataList2D(input_data_list, augmentor)
            one_roi = _AugmentDataList2D([one_roi], augmentor)[0]

            input_data_list = _CropDataList2D(input_data_list, input_shape)
            one_roi = _CropDataList2D([one_roi], input_shape)[0]

            one_roi_list = [one_roi]
            for index in np.arange(1, hierarchical_level + 1):
                temp_roi = deepcopy(cv2.resize(one_roi,
                                      (one_roi.shape[0] // np.power(2, index), one_roi.shape[1] // np.power(2, index)),
                                      interpolation=cv2.INTER_LINEAR))
                one_roi_list.insert(0, temp_roi[..., np.newaxis])

            _AddOneSample(input_list, input_data_list)
            _AddOneSample(output_list, one_roi_list)

            if len(input_list[0]) >= batch_size:
                inputs = _MakeKerasFormat(input_list)
                outputs = _MakeKerasFormat(output_list)
                yield inputs, outputs
                input_list = [[] for index in range(input_number)]
                output_list = [[] for index in range(hierarchical_level + 1)]

def ImageInMultiROIOut2DTest(root_folder, input_shape, hierarchical_level=0):
    from MeDIT.Visualization import LoadWaitBar
    input_number, output_number = _GetInputOutputNumber(root_folder)
    assert (output_number == 1)

    input_list = [[] for index in range(input_number)]
    output_list = [[] for index in range(hierarchical_level + 1)]

    case_list = sorted(os.listdir(root_folder))
    case_list = [case for case in case_list if case.endswith('.h5')]

    for case in case_list:
        LoadWaitBar(len(case_list), case_list.index(case))
        case_path = os.path.join(root_folder, case)

        input_data_list, output_data_list = [], []
        file = h5py.File(case_path, 'r')
        for input_number_index in range(input_number):
            temp_data = np.asarray(file['input_' + str(input_number_index)])
            if temp_data.ndim == 2:
                temp_data = temp_data[..., np.newaxis]
            input_data_list.append(temp_data)

        one_roi = np.asarray(file['output_0'])

        input_data_list = _CropDataList2D(input_data_list, input_shape)
        one_roi = _CropDataList2D([one_roi], input_shape)[0]

        one_roi_list = [one_roi]
        for index in np.arange(1, hierarchical_level + 1):
            temp_roi = deepcopy(cv2.resize(one_roi,
                                           (one_roi.shape[0] // np.power(2, index),
                                            one_roi.shape[1] // np.power(2, index)),
                                           interpolation=cv2.INTER_LINEAR))
            one_roi_list.insert(0, temp_roi[..., np.newaxis])

        _AddOneSample(input_list, input_data_list)
        _AddOneSample(output_list, one_roi_list)

        inputs = _MakeKerasFormat(input_list)
        outputs = _MakeKerasFormat(output_list)

    return inputs, outputs, case_list

def _TestOutput3():
    input_list, output_list = ImageInMultiROIOut2D(r'v:\ProstateSegment\data\Input_1_Ouput_1\testing',
                                                input_shape=[160, 160], batch_size=1,
                                                hierarchical_level=2,
                                                augment_param=random_2d_augment)

    show_input = np.squeeze(input_list)
    roi1, roi2, roi3 = np.squeeze(output_list[0]), np.squeeze(output_list[1]), np.squeeze(output_list[2])

    import matplotlib.pyplot as plt
    plt.subplot(221)
    plt.imshow(show_input, cmap='gray')
    plt.subplot(222)
    plt.imshow(roi1, cmap='gray')
    plt.subplot(223)
    plt.imshow(roi2, cmap='gray')
    plt.subplot(224)
    plt.imshow(roi3, cmap='gray')
    plt.show()

##############################################

if __name__ == '__main__':
    _TestOutput3()