import numpy as np
import os
from configparser import ConfigParser
import SimpleITK as sitk
from keras.models import model_from_yaml

from MeDIT.ArrayProcess  import ExtractPatch, XY2Index, Crop2DImage, DetectRegionBlurry, GetIndexRangeInROI, ExtractBlock
from MeDIT.CNNModel.ImagePrepare import ImagePrepare
from MeDIT.CNNModel.ProstateSegment import ProstateSegmentation2D
from MeDIT.Normalize import NormalizeForModality, NormalizeByROI
from MeDIT.SaveAndLoad import GetImageFromArrayByImage, SaveNiiImage
from MeDIT.ImageProcess import GetDataFromSimpleITK
from scipy import ndimage
from scipy.ndimage import filters
from scipy.ndimage.morphology import binary_closing


class CST2AdcDwiDetect():
    def __init__(self):
        self.__model = None
        self.__image_preparer = ImagePrepare()

    def __RemoveSmallRegion(self, mask, size_thres=50):
        # seperate each connected ROI
        label_im, nb_labels = ndimage.label(mask)

        # remove small ROI
        for i in range(1, nb_labels + 1):
            if (label_im == i).sum() < size_thres:
                # remove the small ROI in mask
                mask[label_im == i] = 0
        return mask

    def LoadModel(self, fold_path):
        from keras.models import model_from_yaml
        model_path = os.path.join(fold_path, 'model.yaml')

        if not self.CheckFileExistence(model_path): return

        yaml_file = open(model_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.__model = model_from_yaml(loaded_model_yaml)

        weight_path = os.path.join(fold_path, 'weights.h5')
        if self.CheckFileExistence(weight_path):
            self.__model.load_weights(weight_path)

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

    # Do Detect
    def Run(self, t2_image, adc_image, dwi_image, detect_model_path, seg_model_path='', prostate_roi_image=None, store_folder=''):
        if len(seg_model_path) != 0 :
            prostate_segmentor = ProstateSegmentation2D()
            prostate_segmentor.LoadModel(seg_model_path)
            prostate_roi_image, prostate_roi = prostate_segmentor.Run(t2_image, seg_model_path)
        else:
            _, prostate_roi = GetDataFromSimpleITK(prostate_roi_image, dtype=np.uint8)

        self.__image_preparer.LoadModelConfig(os.path.join(detect_model_path, 'config.ini'))
        if self.__model == None:
            self.LoadModel(fold_path=detect_model_path)
            
        if not ((t2_image.GetSpacing() == adc_image.GetSpacing()) and (dwi_image.GetSpacing() == prostate_roi_image.GetSpacing())):
            print("The spacing is not consistant among mp-mr images")
            return
        if not ((t2_image.GetSize() == adc_image.GetSize()) and (dwi_image.GetSize() == prostate_roi_image.GetSize())):
            print("The size is not consistant among mp-mr images")
            return

        _, t2 = GetDataFromSimpleITK(t2_image)
        _, adc = GetDataFromSimpleITK(adc_image)
        _, dwi = GetDataFromSimpleITK(dwi_image)

        resolution = t2_image.GetSpacing()

        pred = np.zeros(t2.shape)
        for slice_index in range(t2.shape[-1]):
            t2_slice = np.asarray(t2[..., slice_index], dtype=np.float32)
            adc_slice = np.asarray(adc[..., slice_index], dtype=np.float32)
            dwi_slice = np.asarray(dwi[..., slice_index], dtype=np.float32)
            prostate_roi_slice = prostate_roi[..., slice_index]

            roi_index_x, roi_index_y = np.where(prostate_roi_slice > 0.5)
            if roi_index_x.size < 10 or roi_index_y.size < 10:
                continue
            else:
                center_x = (np.max(roi_index_x) + np.min(roi_index_x)) // 2
                center_y = (np.max(roi_index_y) + np.min(roi_index_y)) // 2

            if np.std(adc_slice) < 1e-4 or np.std(dwi_slice) < 1e-4:
                continue

            t2_slice = self.__image_preparer.CropDataShape(t2_slice, resolution, [center_x, center_y])
            adc_slice = self.__image_preparer.CropDataShape(adc_slice, resolution, [center_x, center_y])
            dwi_slice = self.__image_preparer.CropDataShape(dwi_slice, resolution, [center_x, center_y])
            prostate_roi_slice = self.__image_preparer.CropDataShape(prostate_roi_slice, resolution,
                                                                     [center_x, center_y])

            t2_slice = NormalizeByROI(t2_slice, np.asarray(prostate_roi_slice > 0.5, dtype=np.uint8))
            adc_slice = NormalizeByROI(adc_slice, np.asarray(prostate_roi_slice > 0.5, dtype=np.uint8))
            dwi_slice = NormalizeByROI(dwi_slice, np.asarray(prostate_roi_slice > 0.5, dtype=np.uint8))

            t2_slice = t2_slice[np.newaxis, ..., np.newaxis]
            adc_slice = adc_slice[np.newaxis, ..., np.newaxis]
            dwi_slice = dwi_slice[np.newaxis, ..., np.newaxis]

            input_list = [t2_slice, adc_slice, dwi_slice]

            pred_slice = self.__model.predict(input_list)
            pred_slice = np.squeeze(pred_slice)
            pred_slice = pred_slice[..., np.newaxis]

            pred_slice = np.squeeze(self.__image_preparer.RecoverDataShape(pred_slice, resolution))
            # To process the extremely cases
            final_shape = t2_image.GetSize()
            final_shape = [final_shape[1], final_shape[0], final_shape[2]]
            pred_slice = Crop2DImage(pred_slice, final_shape)

            pred[..., slice_index] = np.squeeze(pred_slice)

        mask = np.asarray(pred > 0.5, dtype=np.uint8)
        mask = self.__RemoveSmallRegion(mask, 20)

        mask_image = GetImageFromArrayByImage(mask, t2_image)
        pred_image = GetImageFromArrayByImage(pred, t2_image)
        if store_folder:
            if os.path.isdir(store_folder):
                roi_output = os.path.join(store_folder, 'CS_PCa_ROI.nii.gz')
                SaveNiiImage(roi_output, mask_image)
                pred_output = os.path.join(store_folder, 'CS_PCa_Pred.nii.gz')
                SaveNiiImage(pred_output, pred_image)

        return pred, pred_image, mask, mask_image

class CST2AdcDwiProstateRoiDetect():
    def __init__(self):
        self.__model = None
        self.__image_preparer = ImagePrepare()

    def __RemoveSmallRegion(self, mask, size_thres=50):
        # seperate each connected ROI
        mask = binary_closing(mask, np.ones((3, 3, 3)))
        label_im, nb_labels = ndimage.label(mask)

        # remove small ROI
        for i in range(1, nb_labels + 1):
            if (label_im == i).sum() < size_thres:
                # remove the small ROI in mask
                mask[label_im == i] = 0
        return mask

    def LoadModel(self, fold_path):
        from keras.models import model_from_yaml
        model_path = os.path.join(fold_path, 'model.yaml')

        if not self.CheckFileExistence(model_path): return

        yaml_file = open(model_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.__model = model_from_yaml(loaded_model_yaml)

        weight_path = os.path.join(fold_path, 'weights.h5')
        if self.CheckFileExistence(weight_path):
            self.__model.load_weights(weight_path)

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

    # Do Detect
    def Run(self, t2_image, adc_image, dwi_image, detect_model_path, seg_model_path='', prostate_roi_image=None,
            store_folder='', threshold=0.5, skip_size=20):
        if len(seg_model_path) != 0:
            prostate_segmentor = ProstateSegmentation2D()
            prostate_segmentor.LoadModel(seg_model_path)
            prostate_roi_image, prostate_roi = prostate_segmentor.Run(t2_image, seg_model_path)
        else:
            _, prostate_roi = GetDataFromSimpleITK(prostate_roi_image, dtype=np.uint8)

        self.__image_preparer.LoadModelConfig(os.path.join(detect_model_path, 'config.ini'))
        if self.__model == None:
            self.LoadModel(fold_path=detect_model_path)

        if not ((t2_image.GetSpacing() == adc_image.GetSpacing()) and (
                dwi_image.GetSpacing() == prostate_roi_image.GetSpacing())):
            print("The spacing is not consistant among mp-mr images")
            return
        if not ((t2_image.GetSize() == adc_image.GetSize()) and (dwi_image.GetSize() == prostate_roi_image.GetSize())):
            print("The size is not consistant among mp-mr images")
            return

        _, t2 = GetDataFromSimpleITK(t2_image)
        _, adc = GetDataFromSimpleITK(adc_image)
        _, dwi = GetDataFromSimpleITK(dwi_image)

        resolution = t2_image.GetSpacing()

        pred = np.zeros(t2.shape)
        for slice_index in range(t2.shape[-1]):
            t2_slice = np.asarray(t2[..., slice_index], dtype=np.float32)
            adc_slice = np.asarray(adc[..., slice_index], dtype=np.float32)
            dwi_slice = np.asarray(dwi[..., slice_index], dtype=np.float32)
            prostate_roi_slice = prostate_roi[..., slice_index]

            roi_index_x, roi_index_y = np.where(prostate_roi_slice > 0.5)
            if roi_index_x.size < 10 or roi_index_y.size < 10:
                continue
            else:
                center_x = (np.max(roi_index_x) + np.min(roi_index_x)) // 2
                center_y = (np.max(roi_index_y) + np.min(roi_index_y)) // 2

            if np.std(adc_slice) < 1e-4 or np.std(dwi_slice) < 1e-4:
                continue

            t2_slice = self.__image_preparer.CropDataShape(t2_slice, resolution, [center_x, center_y])
            adc_slice = self.__image_preparer.CropDataShape(adc_slice, resolution, [center_x, center_y])
            dwi_slice = self.__image_preparer.CropDataShape(dwi_slice, resolution, [center_x, center_y])
            prostate_roi_slice = self.__image_preparer.CropDataShape(prostate_roi_slice, resolution,
                                                                     [center_x, center_y])

            t2_slice = NormalizeByROI(t2_slice, np.asarray(prostate_roi_slice > 0.5, dtype=np.uint8))
            adc_slice = NormalizeByROI(adc_slice, np.asarray(prostate_roi_slice > 0.5, dtype=np.uint8))
            dwi_slice = NormalizeByROI(dwi_slice, np.asarray(prostate_roi_slice > 0.5, dtype=np.uint8))

            t2_slice = t2_slice[np.newaxis, ..., np.newaxis]
            adc_slice = adc_slice[np.newaxis, ..., np.newaxis]
            dwi_slice = dwi_slice[np.newaxis, ..., np.newaxis]

            prostate_roi_slice = DetectRegionBlurry(prostate_roi_slice, hard_dist=5, soft_dist=5)
            prostate_slice = prostate_roi_slice[np.newaxis, ..., np.newaxis]

            input_list = [t2_slice, adc_slice, dwi_slice, prostate_slice]

            pred_slice = self.__model.predict(input_list)
            pred_slice = np.squeeze(pred_slice)
            pred_slice = pred_slice[..., np.newaxis]

            pred_slice = np.squeeze(self.__image_preparer.RecoverDataShape(pred_slice, resolution))
            # To process the extremely cases
            final_shape = t2_image.GetSize()
            final_shape = [final_shape[1], final_shape[0], final_shape[2]]
            pred_slice = Crop2DImage(pred_slice, final_shape)

            pred[..., slice_index] = np.squeeze(pred_slice)

        mask = np.asarray(pred > threshold, dtype=np.uint8)
        mask = self.__RemoveSmallRegion(mask, skip_size)
        mask = np.asarray(mask, dtype=np.uint8)

        mask_image = GetImageFromArrayByImage(mask, t2_image)
        pred_image = GetImageFromArrayByImage(pred, t2_image)
        if store_folder:
            if os.path.isdir(store_folder):
                roi_output = os.path.join(store_folder, 'CS_PCa_ROI_Prsotate_ROI_Constrain.nii.gz')
                SaveNiiImage(roi_output, mask_image)
                pred_output = os.path.join(store_folder, 'CS_PCa_Pred_ROI_Constrain.nii.gz')
                SaveNiiImage(pred_output, pred_image)

        return pred, pred_image, mask, mask_image

class CST2AdcDwiDetect2_5D():
    def __init__(self):
        self.__model = None
        self.__image_preparer = ImagePrepare()

    def __RemoveSmallRegion(self, mask, size_thres=50):
        # seperate each connected ROI
        label_im, nb_labels = ndimage.label(mask)

        # remove small ROI
        for i in range(1, nb_labels + 1):
            if (label_im == i).sum() < size_thres:
                # remove the small ROI in mask
                mask[label_im == i] = 0
        return mask

    def LoadModel(self, fold_path):
        from keras.models import model_from_yaml
        model_path = os.path.join(fold_path, 'model.yaml')

        if not self.CheckFileExistence(model_path): return

        yaml_file = open(model_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.__model = model_from_yaml(loaded_model_yaml)

        weight_path = os.path.join(fold_path, 'weights.h5')
        if self.CheckFileExistence(weight_path):
            self.__model.load_weights(weight_path)

    def CheckFileExistence(self, filepath):
        if os.path.isfile(filepath):
            return True
        else:
            print('Current file not exist or path is not correct')
            print('current filepath:' + os.path.abspath('.') + '\\' + filepath)
            return False

    # Do Detect
    def Run(self, t2_image, adc_image, dwi_image, detect_model_path, seg_model_path='', prostate_roi_image=None,
            store_folder=''):
        if len(seg_model_path) != 0:
            prostate_segmentor = ProstateSegmentation2D()
            prostate_segmentor.LoadModel(seg_model_path)
            prostate_roi_image, prostate_roi = prostate_segmentor.Run(t2_image, seg_model_path)
        else:
            _, prostate_roi = GetDataFromSimpleITK(prostate_roi_image, dtype=np.uint8)

        self.__image_preparer.LoadModelConfig(os.path.join(detect_model_path, 'config.ini'))
        if self.__model == None:
            self.LoadModel(fold_path=detect_model_path)

        if not ((t2_image.GetSize() == adc_image.GetSize()) and (dwi_image.GetSize() == prostate_roi_image.GetSize())):
            print("The size is not consistant among mp-mr images")
            return

        _, t2 = GetDataFromSimpleITK(t2_image)
        _, adc = GetDataFromSimpleITK(adc_image)
        _, dwi = GetDataFromSimpleITK(dwi_image)

        resolution = t2_image.GetSpacing()

        pred = np.zeros(t2.shape)
        roi_sum_slice = np.sum(prostate_roi, axis=(0, 1))
        candidate_slice = np.where(np.asarray(roi_sum_slice > 10, dtype=np.uint8))[0]

        for slice_index in candidate_slice:
            if slice_index == 0 or slice_index == t2.shape[-1] - 1:
                continue

            prosate_block = np.asarray(prostate_roi[..., slice_index - 1: slice_index + 2], dtype=np.uint8)
            x, y, z = GetIndexRangeInROI(prosate_block)
            center_x = (max(x) + min(x)) // 2
            center_y = (max(y) + min(y)) // 2

            t2_block = self.__image_preparer.CropDataShape(t2[..., slice_index - 1 : slice_index + 2], resolution, [center_x, center_y, -1])
            adc_block = self.__image_preparer.CropDataShape(adc[..., slice_index - 1 : slice_index + 2], resolution, [center_x, center_y, -1])
            dwi_block = self.__image_preparer.CropDataShape(dwi[..., slice_index - 1 : slice_index + 2], resolution, [center_x, center_y, -1])
            prostate_roi_block = self.__image_preparer.CropDataShape(prostate_roi[..., slice_index - 1 : slice_index + 2], resolution, [center_x, center_y, -1])

            if np.std(adc_block) < 1e-4 or np.std(dwi_block) < 1e-4:
                continue

            t2_block = NormalizeByROI(t2_block, prostate_roi_block)
            adc_block = NormalizeByROI(adc_block, prostate_roi_block)
            dwi_block = NormalizeByROI(dwi_block, prostate_roi_block)

            t2_block = t2_block[np.newaxis, ...]
            adc_block = adc_block[np.newaxis, ...]
            dwi_block = dwi_block[np.newaxis, ...]

            input_list = [t2_block, adc_block, dwi_block]

            pred_slice = self.__model.predict(input_list)
            pred_slice = np.squeeze(pred_slice)
            pred_slice = pred_slice[..., np.newaxis]

            pred_slice = np.squeeze(self.__image_preparer.RecoverDataShape(pred_slice, resolution))

            # To process the extremely cases
            final_shape = t2_image.GetSize()
            final_shape = [final_shape[1], final_shape[0]]
            pred_slice = Crop2DImage(pred_slice, final_shape)

            pred[..., slice_index] = np.squeeze(pred_slice)

        mask = np.asarray(pred > 0.5, dtype=np.uint8)
        mask = self.__RemoveSmallRegion(mask, 20)

        mask_image = GetImageFromArrayByImage(mask, t2_image)
        pred_image = GetImageFromArrayByImage(pred, t2_image)
        if store_folder:
            if os.path.isdir(store_folder):
                roi_output = os.path.join(store_folder, 'CS_PCa_25_ROI_25.nii.gz')
                SaveNiiImage(roi_output, mask_image)
                pred_output = os.path.join(store_folder, 'CS_PCa_Pred_25.nii.gz')
                SaveNiiImage(pred_output, pred_image)

        return pred, pred_image, mask, mask_image

def testDetect():
    # model_folder_path = r'C:\MyCode\MPApp\DPmodel\CSPCaDetection'
    model_folder_path = r'Z:\SuccessfulModel\PCaDetection25_2'
    # pca_detect = CST2AdcDwiProstateRoiDetect()
    pca_detect = CST2AdcDwiDetect2_5D()
    pca_detect.LoadModel(model_folder_path)

    from MeDIT.SaveAndLoad import LoadNiiData
    t2_image, _, t2_data = LoadNiiData(r'z:\Data\CS_ProstateCancer_Detect_multicenter\ToTest\CAO YONG CHENG\004_t2_fse_tra_Resize.nii', dtype=np.float32, is_show_info=True)
    dwi_image, _, dwi_data = LoadNiiData(r'z:\Data\CS_ProstateCancer_Detect_multicenter\ToTest\CAO YONG CHENG\007_epi_dwi_tra_CBSF5_NL3_b750_Reg_Resize.nii', dtype=np.float32, is_show_info=True)
    adc_image, _, adc_data = LoadNiiData(r'z:\Data\CS_ProstateCancer_Detect_multicenter\ToTest\CAO YONG CHENG\010_epi_dwi_tra_CBSF5_NL3_ADC_Reg_Resize.nii', dtype=np.float32, is_show_info=True)
    prostate_roi_image, _, prostate_roi_data = LoadNiiData(r'z:\Data\CS_ProstateCancer_Detect_multicenter\ToTest\CAO YONG CHENG\prostate_roi.nii.gz', dtype=np.uint8)

    # dwi_data = dwi_data[..., -1]
    # dwi_image = GetImageFromArrayByImage(dwi_data, adc_image)

    print(t2_data.shape, adc_data.shape, dwi_data.shape)

    pred, _, mask, _ = pca_detect.Run(t2_image, adc_image, dwi_image,
                                                 model_folder_path, prostate_roi_image=prostate_roi_image)

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01
    show_data = np.concatenate((Normalize01(t2_data), Normalize01(dwi_data), Normalize01(adc_data), pred), axis=1)
    show_roi = np.concatenate((mask, mask, mask, mask), axis=1)
    show_prostate_roi = np.concatenate((prostate_roi_data, prostate_roi_data, prostate_roi_data), axis=1)
    Imshow3DArray(show_data, ROI=[show_roi, show_prostate_roi])


if __name__ == '__main__':
    testDetect()