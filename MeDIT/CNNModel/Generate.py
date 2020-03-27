import os
from random import shuffle
import numpy as np
import json
from MeDIT.SaveAndLoad import LoadH5InfoForGenerate, LoadH5
from MeDIT.ArrayProcess import ExtractPatch
from MeDIT.DataAugmentor import DataAugmentor2D, DataAugmentor3D, AugmentParametersGenerator

def InitialInputList(input_shape, batch_size, info):
    input_list = []
    for input_number in range(info['input_number']):
        if len(input_shape) == 2:
            one_data = np.zeros((batch_size, input_shape[0], input_shape[1], info['input_channel'][input_number + 1]))
        else:
            one_data = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2], info['input_channel'][input_number + 1]))
        input_list.append(one_data)
    return input_list

def GenerateMultiInputOneOutput_From2DMultiSliceTo2D(root_folder, input_shape, batch_size=8, augment_config='', is_yield=True):
    if augment_config:
        with open(augment_config, 'r') as file:
            random_params = json.load(file)
        param_generator = AugmentParametersGenerator()
        aug_generator = DataAugmentor2D()

    case_list = os.listdir(root_folder)

    shuffle(case_list)
    one_path = os.path.join(root_folder, case_list[0])
    info = LoadH5InfoForGenerate(one_path)

    if info['input_number'] <= 1:
        print('Need Multi Input ', one_path)
        return


    input_list = [[] for temp in range(info['input_number'])]
    one_output = LoadH5(one_path, tag='output_0')
    output_array = np.zeros((batch_size, input_shape[0], input_shape[1], one_output.shape[-1]))
    current_batch = 0

    while True:
        shuffle(case_list)
        for case in case_list:
            case_path = os.path.join(root_folder, case)

            if augment_config:
                param_generator.RandomParameters(random_params)
                aug_generator.SetParameter(param_generator.GetRandomParametersDict())

            for input_number in range(info['input_number']):
                data = LoadH5(case_path, tag='input_{:d}'.format(input_number))

                crop_data = np.zeros((input_shape[0], input_shape[1], data.shape[-1]))
                for slice_index in range(data.shape[-1]):
                    if augment_config:
                        aug_data = aug_generator.Execute(data[..., slice_index], interpolation_method='linear')
                        crop_data[..., slice_index] = ExtractPatch(aug_data, patch_size=input_shape)[0]
                    else:
                        crop_data[..., slice_index] = ExtractPatch(data[..., slice_index], patch_size=input_shape)[0]
                input_list[input_number].append(crop_data)

            output_data = LoadH5(case_path, tag='output_0')
            if augment_config:
                aug_data = aug_generator.Execute(np.squeeze(output_data), interpolation_method='linear')
                output_array[current_batch, ..., 0] = ExtractPatch(aug_data, patch_size=input_shape)[0]
            else:
                output_array[current_batch, ..., 0] = ExtractPatch(np.squeeze(output_data), patch_size=input_shape)[0]

            current_batch += 1
            if current_batch >= batch_size:
                input_list = [np.asarray(temp) for temp in input_list]
                return input_list, output_array
                current_batch = 0
                # yield input_list, output_array
                input_list = [[] for temp in range(info['input_number'])]
                output_array = np.zeros((batch_size, input_shape[0], input_shape[1], one_output.shape[-1]))

# root_folder = r'C:\Users\SY\Desktop\H5_test'
root_folder = r'z:\Data\CS_ProstateCancer_Detect_multicenter\JSPH_NIH_H5\3sliceInput_2017'
from MeDIT.Visualization import Imshow3DArray, DrawBoundaryOfBinaryMask, FlattenAllSlices
from MeDIT.Normalize import NormalizeEachSlice01, Normalize01

while True:
    input_list, output_array = GenerateMultiInputOneOutput_From2DMultiSliceTo2D(root_folder, input_shape=[96, 96], batch_size=1, is_yield=False)
    input_array = np.asarray(input_list)
    input_array = np.transpose(np.squeeze(input_array), (1, 2, 0, 3))
    input_array = np.reshape(input_array, (input_array.shape[0], input_array.shape[1], -1))
    Imshow3DArray(input_array)
    input_array = FlattenAllSlices(NormalizeEachSlice01(input_array), is_show=False)

    show_roi = np.squeeze(output_array)
    show_roi = np.concatenate((show_roi, show_roi, show_roi), axis=0)
    show_roi = np.concatenate((np.zeros_like(show_roi), show_roi, np.zeros_like(show_roi)), axis=1)
    DrawBoundaryOfBinaryMask(input_array, show_roi)

