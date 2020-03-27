# Created on Jun 05, 2018
# author: Yang Song
# All right reserved

import os
from configparser import ConfigParser
import numpy as np

from scipy import ndimage as nd


class ImagePrepare:
    def __init__(self):
        self._config = {}
        self._log = []
        self._center_point = [-1, -1, -1]

        self.__curslice = None

    def LoadModelConfig(self, config_path):
        cf = ConfigParser()
        if not os.path.exists(config_path):
            print('Check the config file path: ', config_path)
            return

        cf.read(config_path)

        self._config = dict()
        self._config['model_dimension'] = int(cf.get("net_structure", "model_dimension"))

        self._config['resolution_x'] = float(cf.get("net_structure", "resolution_x"))
        self._config['input_x'] = int(cf.get("net_structure", "input_x"))

        self._config['resolution_y'] = float(cf.get("net_structure", "resolution_y"))
        self._config['input_y'] = int(cf.get("net_structure", "input_y"))

        if self._config['model_dimension'] == 3:
            self._config['resolution_z'] = float(cf.get("net_structure", "resolution_z"))
            self._config['input_z'] = int(cf.get("net_structure", "input_z"))

        self._config['num_channel'] = int(cf.get("net_structure", "num_channel"))
        for index in range(self._config['num_channel']):
            self._config[('modality_' + str(index))] = cf.get("net_structure", ('modality_' + str(index)))

    def CropDataShape(self, data, raw_resolution, center_point=[-1, -1, -1]):
        if isinstance(raw_resolution, list):
            raw_resolution = np.asarray(raw_resolution)

        new_data = np.copy(data)
        if new_data.ndim == 2:
            new_data = new_data[..., np.newaxis]

        # To log wrap or crop the data, 0 denotes same, 1 denotes the crop and 2 denotes the wrap. For example, if there
        # 3 dimension, the crop_wrap_state will be one vector with size 12. like:
        # [image row, image_col, image_slice, #image after interpolation,
        #  row_state, row_start, row_end, col_state, col_start, col_end, slice_state, slice_start, slice_end]
        if self._config['model_dimension'] == 2:
            target_resolution = [self._config['resolution_x'], self._config['resolution_y'],
                                 raw_resolution[2]]
            final_data_shape = [self._config['input_x'], self._config['input_y'], new_data.shape[2]]
            self._log = np.zeros((9,))

        elif self._config['model_dimension'] == 3:
            target_resolution = [self._config['resolution_x'], self._config['resolution_y'],
                                 self._config['resolution_z']]
            final_data_shape = [self._config['input_x'], self._config['input_y'],
                                self._config['input_z']]
            self._log = np.zeros((12,))
        else:
            return []

        target_resolution = np.asarray(target_resolution)
        final_data_shape = np.asarray(final_data_shape)
        if len(center_point) == 2:
            center_point.extend([-1])

        self._center_point = []
        for index in range(len(target_resolution)):
            if center_point[index] == -1:
                self._center_point.append(-1)
            else:
                self._center_point.append(int(center_point[index] * raw_resolution[index] / target_resolution[index]))

        new_data = nd.interpolation.zoom(new_data, raw_resolution / target_resolution, order=3)
        new_data_shape = np.shape(new_data)
        self._log[0:3] = new_data_shape
        # final_data_shape = [self._config['input_x'], self._config['input_y'], self._config['input_z']]

        if final_data_shape[0] < new_data_shape[0]:  # crop new_data in row direction
            self._log[3] = 1
            if self._center_point[0] == -1:
                row_start = new_data_shape[0] // 2 - final_data_shape[0] // 2
            else:
                row_start = self._center_point[0] - final_data_shape[0] // 2
            row_end = row_start + final_data_shape[0]
            new_data = new_data[row_start:row_end, :, :]

        elif final_data_shape[0] > new_data_shape[0]:  # wrap zeros on new_data in row direction
            self._log[3] = 2
            row_start = final_data_shape[0] // 2 - new_data_shape[0] // 2
            row_end = row_start + new_data_shape[0]
            temp_new_data = np.zeros((final_data_shape[0], new_data_shape[1], new_data_shape[2]))
            temp_new_data[row_start:row_end, :, :] = new_data
            new_data = temp_new_data
            del temp_new_data
        else:
            row_start = 0
            row_end = final_data_shape[0]
        self._log[4:6] = [row_start, row_end]

        if final_data_shape[1] < new_data_shape[1]:  # crop new_data in col direction
            self._log[6] = 1
            if self._center_point[1] == -1:
                col_start = new_data_shape[1] // 2 - final_data_shape[1] // 2
            else:
                col_start = self._center_point[1] - final_data_shape[1] // 2
            col_end = col_start + final_data_shape[1]
            new_data = new_data[:, col_start:col_end, :]

        elif final_data_shape[1] > new_data_shape[1]:  # wrap zeros on new_data in col direction
            self._log[6] = 2
            col_start = final_data_shape[1] // 2 - new_data_shape[1] // 2
            col_end = col_start + new_data_shape[1]
            temp_new_data = np.zeros((final_data_shape[0], final_data_shape[1], new_data_shape[2]))
            temp_new_data[:, col_start:col_end, :] = new_data
            new_data = temp_new_data
            del temp_new_data
        else:
            col_start = 0
            col_end = final_data_shape[1]
        self._log[7:9] = [col_start, col_end]

        if self._config['model_dimension'] == 3:
            self._log[9] = 1
            if final_data_shape[2] < new_data_shape[2]:  # crop new_data in slice direction
                if self.__curslice != None:
                    slice_start = self.__curslice - 8
                    slice_end = self.__curslice + 8
                else:
                    slice_start = new_data_shape[2] // 2 - final_data_shape[2] // 2
                    slice_end = slice_start + final_data_shape[2]

                new_data = new_data[:, :, slice_start:slice_end]
            elif final_data_shape[2] > new_data_shape[2]:  # wrap zeros on new_data in slice direction
                self._log[9] = 2
                slice_start = final_data_shape[2] // 2 - new_data_shape[2] // 2
                slice_end = slice_start + new_data_shape[2]
                temp_new_data = np.zeros((final_data_shape[0], final_data_shape[1], final_data_shape[2]))
                temp_new_data[:, :, slice_start:slice_end] = new_data
                new_data = temp_new_data
                del temp_new_data
            else:
                slice_start = 0
                slice_end = final_data_shape[2]
            self._log[10:12] = [slice_start, slice_end]

        self._log = np.asarray(self._log, dtype=np.int)
        return np.squeeze(new_data)

    def RecoverDataShape(self, data, raw_resolution):
        new_data = np.copy(data)

        if self._config['model_dimension'] == 2:
            target_resolution = [self._config['resolution_x'], self._config['resolution_y'],
                                 raw_resolution[2]]

        elif self._config['model_dimension'] == 3:
            target_resolution = [self._config['resolution_x'], self._config['resolution_y'],
                                 self._config['resolution_slice']]
        else:
            return []

        target_resolution = np.asarray(target_resolution)
        hidden_data_shape = self._log[0:3]
        new_data_shape = np.shape(new_data)
        # new_data = nd.interpolation.zoom(new_data, target_resolution / raw_resolution)

        row_start = self._log[4]
        row_end = self._log[5]
        if self._log[3] == 2:  # crop new_data in row direction
            new_data = new_data[row_start:row_end, :, :]
        elif self._log[3] == 1:  # wrap zeros on new_data in row direction
            temp_new_data = np.zeros((hidden_data_shape[0], new_data_shape[1], new_data_shape[2]))
            temp_new_data[row_start:row_end, :, :] = new_data
            new_data = temp_new_data
            del temp_new_data

        col_start = self._log[7]
        col_end = self._log[8]
        if self._log[6] == 2:  # crop new_data in col direction
            new_data = new_data[:, col_start:col_end, :]
        elif self._log[6] == 1:  # wrap zeros on new_data in col direction
            temp_new_data = np.zeros((hidden_data_shape[0], hidden_data_shape[1], new_data_shape[2]))
            temp_new_data[:, col_start:col_end, :] = new_data
            new_data = temp_new_data
            del temp_new_data

        if self._config['model_dimension'] == 3:
            slice_start = self._log[10]
            slice_end = self._log[11]
            if self._log[9] == 2:  # crop new_data in slice direction
                new_data = new_data[:, :, slice_start:slice_end]
            elif self._log[9] == 1:  # wrap zeros on new_data in slice direction
                temp_new_data = np.zeros((hidden_data_shape[0], hidden_data_shape[1], hidden_data_shape[2]))
                temp_new_data[:, :, slice_start:slice_end] = new_data
                new_data = temp_new_data
                del temp_new_data

        new_data = nd.interpolation.zoom(new_data, target_resolution / raw_resolution, order=2)
        return new_data
    
    def GetDimension(self):
        return self._config['model_dimension']
    
    def GetResolution(self):
        if self.GetDimension() == 2:
            return  [self._config['resolution_x'],  self._config['resolution_y']]
        if self.GetDimension() == 3:
            return  [self._config['resolution_x'],  self._config['resolution_y'], self._config['resolution_z']]
        
    def GetShape(self):
        if self.GetDimension() == 2:
            return [self._config['input_x'], self._config['input_y']]
        if self.GetDimension() == 3:
            return [self._config['input_x'], self._config['input_y'], self._config['input_z']]


if __name__ == '__main__':
    import nibabel as nb
    data = nb.load(r'..\..\ProstateX-0004\005_t2_tse_tra.nii')
    mate_data = np.transpose(np.asarray(data.get_data(), dtype=np.float64), (1, 0, 2))

    image_preparer = ImagePrepare()
    image_preparer.LoadModelConfig(r'..\..\DPmodel\ProstateSegmentation\config.ini')
    resize_image = image_preparer.CropDataShape(mate_data, data.header['pixdim'][1:4])
    recover_image = image_preparer.RecoverDataShape(resize_image, data.header['pixdim'][1:4])

    print(mate_data.shape)
    print(resize_image.shape)
    print(recover_image.shape)

    from Imshow3D import Imshow3D
    from Normalize import Normalize01

    Imshow3D(Normalize01(mate_data))
    Imshow3D(Normalize01(resize_image))
    Imshow3D(Normalize01(recover_image))