import numpy as np
import os
from scipy import ndimage as nd
from scipy import ndimage
from keras.models import model_from_yaml

from MeDIT.CNNModel.ImagePrepare import ImagePrepare
from MeDIT.ImageProcess import GetDataFromSimpleITK, GetImageFromArrayByImage
from MeDIT.Normalize import NormalizeForTensorflow
from MeDIT.SaveAndLoad import SaveNiiImage
from MeDIT.ArrayProcess import Crop3DImage

class ProstateSegmentation2D:
    def __init__(self):
        self._loaded_model = None
        self._selected_slice_index = None
        self._raw_data_shape = None
        self._selected_index = dict()
        self._image_preparer = ImagePrepare()

    def __KeepLargest(self, mask):
        label_im, nb_labels = ndimage.label(mask)
        max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
        index = np.argmax(max_volume)
        new_mask = np.zeros(mask.shape)
        new_mask[label_im == index + 1] = 1
        return new_mask

    def LoadModel(self, fold_path):

        model_path = os.path.join(fold_path, 'model.yaml')

        if not self.CheckFileExistence(model_path): return

        yaml_file = open(model_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self._loaded_model = model_from_yaml(loaded_model_yaml)

        weight_path = os.path.join(fold_path, 'weights.h5')
        if self.CheckFileExistence(weight_path):
            self._loaded_model.load_weights(weight_path)

    def CheckFileExistence(self, filepath):
        if os.path.isfile(filepath):
            return True
        else:
            print('Current file not exist or path is not correct')
            print('current filepath:' + os.path.abspath('.') + '\\' + filepath)
            return False

    def TransDataFor2DModel(self, data):
        data = data.swapaxes(0, 2).swapaxes(1, 2)  # Exchange the order of axis
        data = data[..., np.newaxis]  # Add channel axis
        return data

    def invTransDataFor2DModel(self, preds):
        preds = np.squeeze(preds)
        # Exchange the order of axis back
        preds = preds.swapaxes(1, 2).swapaxes(0, 2)
        return preds

    def Run(self, image, model_folder_path, store_folder='', is_return_all_data=False):
        resolution = image.GetSpacing()
        _, data = GetDataFromSimpleITK(image, dtype=np.float32)

        self._image_preparer.LoadModelConfig(os.path.join(model_folder_path, 'config.ini'))

        ''' 2) Select Data'''
        data = self._image_preparer.CropDataShape(data, resolution)
        data = self.TransDataFor2DModel(data)
        data = NormalizeForTensorflow(data)

        preds = self._loaded_model.predict(data)

        if is_return_all_data:
            data = np.squeeze(data)
            preds = np.squeeze(preds)
            pred1 = preds[:, :np.prod(data.shape[1:3])//16]
            pred2 = preds[:, np.prod(data.shape[1:3])//16 : -np.prod(data.shape[1:3])]
            pred3 = preds[:, -np.prod(data.shape[1:3]):]

            pred1 = np.reshape(pred1, (data.shape[0], data.shape[1] // 4, data.shape[2] // 4))
            pred2 = np.reshape(pred2, (data.shape[0], data.shape[1] // 2, data.shape[2] // 2))
            pred3 = np.reshape(pred3, (data.shape[0], data.shape[1], data.shape[2]))

            import cv2
            data1 = np.zeros((data.shape[0], data.shape[1] // 4, data.shape[2] // 4))
            data2 = np.zeros((data.shape[0], data.shape[1] // 2, data.shape[2] // 2))

            for slice_index in range(data.shape[0]):
                data1[slice_index, ...] = cv2.resize(data[slice_index, ...], (data.shape[1] // 4, data.shape[2] // 4))
                data2[slice_index, ...] = cv2.resize(data[slice_index, ...], (data.shape[1] // 2, data.shape[2] // 2))

            data1 = np.transpose(data1, (1, 2, 0))
            data2 = np.transpose(data2, (1, 2, 0))
            data3 = np.transpose(data, (1, 2, 0))

            pred1 = np.transpose(pred1, (1, 2, 0))
            pred2 = np.transpose(pred2, (1, 2, 0))
            pred3 = np.transpose(pred3, (1, 2, 0))

            return [data1, data2, data3], [pred1, pred2, pred3]

        preds = preds[:, -np.prod(self._image_preparer.GetShape()):, :]
        preds = np.reshape(preds, (
        data.shape[0], self._image_preparer.GetShape()[0], self._image_preparer.GetShape()[1]))

        # ct the ROI
        preds = self.invTransDataFor2DModel(preds)
        preds = self._image_preparer.RecoverDataShape(preds, resolution)

        mask = np.asarray(preds > 0.5, dtype=np.uint8)
        mask = self.__KeepLargest(mask)

        # To process the extremely cases
        final_shape = image.GetSize()
        final_shape = [final_shape[1], final_shape[0], final_shape[2]]
        mask = Crop3DImage(mask, final_shape)

        mask_image =  GetImageFromArrayByImage(mask, image)
        if store_folder:
            if os.path.isdir(store_folder):
                store_folder = os.path.join(store_folder, 'prostate_roi.nii.gz')
            SaveNiiImage(store_folder, mask_image)

        return preds, mask, mask_image

class ProstateSegmentation2_5D:
    def __init__(self):
        self._loaded_model = None
        self._selected_slice_index = None
        self._raw_data_shape = None
        self._selected_index = dict()
        self._image_preparer = ImagePrepare()

    def __KeepLargest(self, mask):
        label_im, nb_labels = ndimage.label(mask)
        max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
        index = np.argmax(max_volume)
        new_mask = np.zeros(mask.shape)
        new_mask[label_im == index + 1] = 1
        return new_mask

    def LoadModel(self, fold_path):
        model_path = os.path.join(fold_path, 'model.yaml')

        if not self.CheckFileExistence(model_path): return

        yaml_file = open(model_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self._loaded_model = model_from_yaml(loaded_model_yaml)

        weight_path = os.path.join(fold_path, 'weights.h5')
        if self.CheckFileExistence(weight_path):
            self._loaded_model.load_weights(weight_path)

    def CheckFileExistence(self, filepath):
        if os.path.isfile(filepath):
            return True
        else:
            print('Current file not exist or path is not correct')
            print('current filepath:' + os.path.abspath('.') + '\\' + filepath)
            return False

    def TransOneDataFor2_5DModel(self, data):
        # Here needs to be set according to config
        data_list = [data[..., :-2], data[..., 1:-1], data[..., 2:]]
        for input_data_index in range(len(data_list)):
            temp = data_list[input_data_index]
            temp = np.transpose(temp, (2, 0, 1))
            temp = temp[..., np.newaxis]
            temp = NormalizeForTensorflow(temp)
            data_list[input_data_index] = temp

        return data_list

    def invTransDataFor2_5DModel(self, preds):
        preds = np.squeeze(preds)
        preds = np.transpose(preds, (1, 2, 0))
        preds = np.concatenate((np.zeros((self._image_preparer.GetShape()[0], self._image_preparer.GetShape()[1], 1)),
                                preds,
                                np.zeros((self._image_preparer.GetShape()[0], self._image_preparer.GetShape()[1], 1))),
                               axis=-1)

        return preds

    def Run(self, image, model_folder_path, store_folder='', is_return_all_data=False, password=''):
        resolution = image.GetSpacing()
        _, data = GetDataFromSimpleITK(image, dtype=np.float32)

        # Load Model
        self._image_preparer.LoadModelConfig(os.path.join(model_folder_path, 'config.ini'))

        # Preprocess Data
        data = self._image_preparer.CropDataShape(data, resolution)
        input_list = self.TransOneDataFor2_5DModel(data)

        preds = self._loaded_model.predict(input_list, batch_size=1)

        if is_return_all_data:
            data = np.squeeze(data)
            preds = np.squeeze(preds)
            pred1 = preds[:, :np.prod(data.shape[1:3])//16]
            pred2 = preds[:, np.prod(data.shape[1:3])//16 : -np.prod(data.shape[1:3])]
            pred3 = preds[:, -np.prod(data.shape[1:3]):]

            pred1 = np.reshape(pred1, (data.shape[0], data.shape[1] // 4, data.shape[2] // 4))
            pred2 = np.reshape(pred2, (data.shape[0], data.shape[1] // 2, data.shape[2] // 2))
            pred3 = np.reshape(pred3, (data.shape[0], data.shape[1], data.shape[2]))

            import cv2
            data1 = np.zeros((data.shape[0], data.shape[1] // 4, data.shape[2] // 4))
            data2 = np.zeros((data.shape[0], data.shape[1] // 2, data.shape[2] // 2))

            for slice_index in range(data.shape[0]):
                data1[slice_index, ...] = cv2.resize(data[slice_index, ...], (data.shape[1] // 4, data.shape[2] // 4))
                data2[slice_index, ...] = cv2.resize(data[slice_index, ...], (data.shape[1] // 2, data.shape[2] // 2))

            data1 = np.transpose(data1, (1, 2, 0))
            data2 = np.transpose(data2, (1, 2, 0))
            data3 = np.transpose(data, (1, 2, 0))

            pred1 = np.transpose(pred1, (1, 2, 0))
            pred2 = np.transpose(pred2, (1, 2, 0))
            pred3 = np.transpose(pred3, (1, 2, 0))

            return [data1, data2, data3], [pred1, pred2, pred3]

        preds = preds[:, :np.prod(self._image_preparer.GetShape())]
        preds = np.reshape(preds,
                           (data.shape[-1] - 2, self._image_preparer.GetShape()[0], self._image_preparer.GetShape()[1]))

        # ct the ROI
        preds = self.invTransDataFor2_5DModel(preds)

        preds = self._image_preparer.RecoverDataShape(preds, resolution)

        mask = np.asarray(preds > 0.5, dtype=np.uint8)
        mask = self.__KeepLargest(mask)

        # To process the extremely cases
        final_shape = image.GetSize()
        final_shape = [final_shape[1], final_shape[0], final_shape[2]]
        mask = Crop3DImage(mask, final_shape)

        mask_image =  GetImageFromArrayByImage(mask, image)
        if store_folder:
            if os.path.isdir(store_folder):
                store_folder = os.path.join(store_folder, 'prostate_roi_25D.nii.gz')
            SaveNiiImage(store_folder, mask_image)

        return preds, mask, mask_image

def testSeg():
    model_folder_path = r'z:\SuccessfulModel\ProstateSegmentation'
    prostate_segmentor = ProstateSegmentation2D()
    prostate_segmentor.LoadModel(model_folder_path)

    import os
    import glob
    from MeDIT.SaveAndLoad import LoadNiiData
    root_folder = r'z:\Data\CS_ProstateCancer_Detect_multicenter\ToTest\Cui hai pei'
    t2_candidate = glob.glob(os.path.join(root_folder, '*t2*tra*Resize.nii'))
    if len(t2_candidate) != 1:
        print('Check T2 Path')
        return
    t2_path = r'z:\temp_data\test_data_prostate\301_case2_1.2.840.113619.2.25.4.1415787.1457070159.316\004_AX T2 PROPELLER.nii'
    image, _, show_data = LoadNiiData(t2_path, dtype=np.float32, is_show_info=True)

    predict_data, mask, mask_image = prostate_segmentor.Run(image, model_folder_path)

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01
    Imshow3DArray(predict_data)
    Imshow3DArray(Normalize01(show_data), ROI=np.asarray(predict_data > 0.5, dtype=np.uint8))


if __name__ == '__main__':
    testSeg()
