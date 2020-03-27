# import dicom2nifti
import pydicom
import os
import shutil
import SimpleITK as sitk
import numpy as np
import nibabel as nb
from copy import deepcopy
from scipy.ndimage.morphology import binary_dilation, binary_erosion

def ProcessROIImage(roi_image, process, store_path='', is_2d=True):
    # Dilate or erode the roi image.
    # If the type of process is int, it denotes the voxel unit.
    # If the type of process is float, it denotes the percentage unit.

    _, roi = GetDataFromSimpleITK(roi_image, dtype=np.uint8)
    if roi.ndim != 3:
        print('Only process on 3D data.')
        return
    if np.max(roi) == 0:
        print('Not valid ROI!')
        return

    if isinstance(process, int):
        if is_2d:
            kernel = np.ones((3, 3))
            processed_roi = np.zeros_like(roi)
            for slice_index in range(roi.shape[2]):
                slice = roi[..., slice_index]
                if np.max(slice) == 0:
                    continue
                if process > 0:
                    processed_roi[..., slice_index] = binary_dilation(slice, kernel, iterations=np.abs(process)).astype(roi.dtype)
                else:
                    processed_roi[..., slice_index] = binary_erosion(slice, kernel, iterations=np.abs(process)).astype(roi.dtype)
        else:
            kernel = np.ones((3, 3, 3))
            if process > 0:
                processed_roi = binary_dilation(roi, kernel, iterations=np.abs(process)).astype(roi.dtype)
            else:
                processed_roi = binary_erosion(roi, kernel, iterations=np.abs(process)).astype(roi.dtype)
    elif isinstance(process, float):
        if is_2d:
            kernel = np.ones((3, 3))
            processed_roi = deepcopy(roi)
            for slice_index in range(roi.shape[2]):
                slice = deepcopy(roi[..., slice_index])
                if np.max(slice) == 0:
                    continue

                if np.abs(process) < 1e-6:
                    processed_roi[..., slice_index] = deepcopy(roi[..., slice_index])
                elif process > 1e-6:
                    while np.sum(processed_roi[..., slice_index]) / np.sum(slice) < 1 + process:
                        processed_roi[..., slice_index] = binary_dilation(slice, kernel, iterations=1).astype(roi.dtype)
                else:
                    while np.sum(processed_roi[..., slice_index]) / np.sum(slice) > 1 + process:
                        processed_roi[..., slice_index] = binary_erosion(processed_roi[..., slice_index], kernel, iterations=1).astype(roi.dtype)
        else:
            kernel = np.ones((3, 3, 3))
            processed_roi = deepcopy(roi)
            if np.abs(process) < 1e-6:
                processed_roi = deepcopy(roi)
            elif process > 1e-6:
                while np.sum(processed_roi) / np.sum(roi) < 1 + process:
                    processed_roi = binary_dilation(roi, kernel, iterations=1).astype(roi.dtype)
            else:
                while np.sum(processed_roi) / np.sum(roi) > 1 + process:
                    processed_roi = binary_erosion(processed_roi, kernel, iterations=1).astype(roi.dtype)
    else:
        processed_roi = roi
        print('The type of the process is not in-valid.')
        return sitk.Image()


    processed_roi_image = GetImageFromArrayByImage(processed_roi, roi_image)

    if store_path:
        sitk.WriteImage(processed_roi_image, store_path)
    return processed_roi_image

def GetImageFromArrayByImage(show_data, refer_image, is_transfer_axis=True):
    if is_transfer_axis:
        show_data = np.transpose(show_data, (2, 0, 1))

    new_image = sitk.GetImageFromArray(show_data)
    new_image.CopyInformation(refer_image)
    return new_image

def GetDataFromSimpleITK(image, dtype=np.float32):
    data = np.asarray(sitk.GetArrayFromImage(image), dtype=dtype)
    show_data = np.transpose(data)
    show_data = np.swapaxes(show_data , 0, 1)

    return data, show_data

def GenerateFileName(file_path, name):
    store_path = ''
    if os.path.splitext(file_path)[1] == '.nii':
        store_path = file_path[:-4] + '_' + name + '.nii'
    elif os.path.splitext(file_path)[1] == '.gz':
        store_path = file_path[:-7] + '_' + name + '.nii.gz'
    else:
        print('the input file should be suffix .nii or .nii.gz')

    return store_path

def DecompressSiemensDicom(data_folder, store_folder, gdcm_path=r"D:\MyCode\Lib\gdcm\GDCMGITBin\bin\Release\gdcmconv.exe"):
    file_list = os.listdir(data_folder)
    file_list.sort()
    for file in file_list:
        file_path = os.path.join(data_folder, file)
        # store_file = os.path.join(store_folder, file+'.IMA')
        store_file = os.path.join(store_folder, file)

        cmd = gdcm_path + " --raw {:s} {:s}".format(file_path, store_file)
        os.system(cmd)

def GetPatientIDfromDicomFolder(dicom_folder):
    file_list = os.listdir(dicom_folder)
    if len(file_list) == 0:
        print('No dicom file')
        return None

    for file in file_list:
        if not os.path.isfile(os.path.join(dicom_folder, file)):
            print('There is other file!')
            return None

    one_file = os.path.join(dicom_folder, file_list[0])
    dcm = pydicom.dcmread(one_file)
    return dcm.PatientID

################################################################################
def ResizeSipmleITKImage(image, expected_resolution=[], expected_shape=[], method=sitk.sitkBSpline, dtype=sitk.sitkFloat32):
    '''
    Resize the SimpleITK image. One of the expected resolution/spacing and final shape should be given.

    :param image: The SimpleITK image.
    :param expected_resolution: The expected resolution.
    :param excepted_shape: The expected final shape.
    :return: The resized image.

    Apr-27-2018, Yang SONG [yang.song.91@foxmail.com]
    '''
    if (expected_resolution == []) and (expected_shape == []):
        print('Give at least one parameters. ')
        return image

    shape = image.GetSize()
    resolution = image.GetSpacing()

    if expected_resolution == []:
        dim_0, dim_1, dim_2 = False, False, False
        if expected_shape[0] == 0: 
            expected_shape[0] = shape[0]
            dim_0 = True
        if expected_shape[1] == 0: 
            expected_shape[1] = shape[1]
            dim_1 = True
        if expected_shape[2] == 0: 
            expected_shape[2] = shape[2]
            dim_2 = True
        expected_resolution = [raw_resolution * raw_size / dest_size for dest_size, raw_size, raw_resolution in
                               zip(expected_shape, shape, resolution)]
        if dim_0: expected_resolution[0] = resolution[0]
        if dim_1: expected_resolution[1] = resolution[1]
        if dim_2: expected_resolution[2] = resolution[2]
        
    elif expected_shape == []:
        dim_0, dim_1, dim_2 = False, False, False
        if expected_resolution[0] < 1e-6: 
            expected_resolution[0] = resolution[0]
            dim_0 = True
        if expected_resolution[1] < 1e-6: 
            expected_resolution[1] = resolution[1]
            dim_1 = True
        if expected_resolution[2] < 1e-6: 
            expected_resolution[2] = resolution[2]
            dim_2 = True
        expected_shape = [int(raw_resolution * raw_size / dest_resolution) for
                       dest_resolution, raw_size, raw_resolution in zip(expected_resolution, shape, resolution)]
        if dim_0: expected_shape[0] = shape[0]
        if dim_1: expected_shape[1] = shape[1]
        if dim_2: expected_shape[2] = shape[2]

    # output = sitk.Resample(image, expected_shape, sitk.AffineTransform(len(shape)), method, image.GetOrigin(),
    #                        expected_resolution, image.GetDirection(), dtype)
    resample_filter = sitk.ResampleImageFilter()
    output = resample_filter.Execute(image, expected_shape, sitk.AffineTransform(len(shape)), method, image.GetOrigin(),
                           expected_resolution, image.GetDirection(), 0.0, dtype)
    return output

def ResizeNiiFile(file_path, store_path='', expected_resolution=[], expected_shape=[], method=sitk.sitkBSpline, dtype=sitk.sitkFloat32):
    expected_resolution = deepcopy(expected_resolution)
    expected_shape = deepcopy(expected_shape)
    if not store_path:
        store_path = GenerateFileName(file_path, 'Resize')

    image = sitk.ReadImage(file_path)
    resized_image = ResizeSipmleITKImage(image, expected_resolution, expected_shape, method=method, dtype=dtype)
    sitk.WriteImage(resized_image, store_path)
    return resized_image

def ResizeROINiiFile(file_path, ref_image, store_path=''):
    if isinstance(ref_image, str):
        ref_image = sitk.ReadImage(ref_image)
    expected_shape = ref_image.GetSize()
    if not store_path:
        store_path = GenerateFileName(file_path, 'Resize')
    image = sitk.ReadImage(file_path)
    resized_image = ResizeSipmleITKImage(image, expected_shape=expected_shape, method=sitk.sitkLinear, dtype=sitk.sitkFloat32)
    data = sitk.GetArrayFromImage(resized_image)

    new_data = np.zeros(data.shape, dtype=np.uint8)
    new_data[data > 0.5] = 1
    new_image = sitk.GetImageFromArray(new_data)
    new_image.CopyInformation(resized_image)
    sitk.WriteImage(new_image, store_path)
    return new_image

################################################################################
def RegistrateImage(fixed_image, moving_image, interpolation_method=sitk.sitkBSpline):
    '''
    Registrate SimpleITK Imageby default parametes.

    :param fixed_image: The reference
    :param moving_image: The moving image.
    :param interpolation_method: The method for interpolation. default is sitkBSpline
    :return: The output image

    Apr-27-2018, Jing ZHANG [798582238@qq.com],
                 Yang SONG [yang.song.91@foxmail.com]
    '''
    if isinstance(fixed_image, str):
        fixed_image = sitk.ReadImage(fixed_image)
    if isinstance(moving_image, str):
        moving_image = sitk.ReadImage(moving_image)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    output_image = sitk.Resample(moving_image, fixed_image, final_transform, interpolation_method, 0.0,
                                     moving_image.GetPixelID())
    return output_image

def RegistrateNiiFile(fixed_image_path, moving_image_path, interpolation_method=sitk.sitkBSpline):
    output_image = RegistrateImage(fixed_image_path, moving_image_path, interpolation_method)
    store_path = GenerateFileName(moving_image_path, 'Reg')
    sitk.WriteImage(output_image, store_path)

def GetTransformByElastix(fix_image_path, moving_image_path, output_folder,
                          elastix_folder=r'c:\MyCode\MPApp\Elastix',
                          parameter_folder=r'c:\MyCode\MPApp\Elastix\RegParam\3ProstateBspline16'):
    '''
    Get registed transform by Elastix. This is depended on the Elastix.

    :param elastix_folder: The folder path of the built elastix.
    :param fix_image_path: The path of the fixed image.
    :param moving_image_path: The path of the moving image.
    :param output_folder: The folder of the output
    :param parameter_folder: The folder path that store the parameter files.
    :return:
    '''
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    cmd = os.path.join(elastix_folder, 'elastix') + r' -f "' + fix_image_path + r'" -m "' + moving_image_path + r'" -out "' + output_folder + '"'
    file_path_list = os.listdir(parameter_folder)
    file_path_list.sort()
    for file_path in file_path_list:
        abs_file_path = os.path.join(parameter_folder, file_path)
        cmd += r' -p "' + abs_file_path + '"'
    os.system(cmd)

def RegisteByElastix(moving_image_path, transform_folder, elastix_folder=r'c:\MyCode\MPApp\Elastix'):
    '''
    Registed Image by Elastix. This is depended on the Elastix.

    :param elastix_folder: The folder path of the built Elastix
    :param moving_image_path: The path of the moving image
    :param transform_folder: The folder path of the generated by the elastix fucntion.
    :return:
    '''
    file_name, suffex = os.path.splitext(moving_image_path)

    temp_folder = os.path.join(transform_folder, 'temp')
    try:
        os.mkdir(temp_folder)
    except:
        pass
    try:
        cmd = os.path.join(elastix_folder, 'transformix') + r' -in "' + moving_image_path + r'" -out "' + temp_folder + '"'
        candidate_transform_file_list = os.listdir(transform_folder)
        candidate_transform_file_list.sort()
        for file_path in candidate_transform_file_list:
            if len(file_path) > len('Transform'):
                if 'Transform' in file_path:
                    abs_transform_path = os.path.join(transform_folder, file_path)
                    cmd += r' -tp "' + abs_transform_path + '"'

        os.system(cmd)
    except:
        shutil.rmtree(temp_folder)

    try:
        shutil.move(os.path.join(temp_folder, 'result.nii'), file_name + '_Reg' + suffex)
        shutil.rmtree(temp_folder)
    except:
        pass

    try:
        shutil.move(os.path.join(temp_folder, 'result.hdr'), file_name + '_Reg' + '.hdr')
        shutil.move(os.path.join(temp_folder, 'result.img'), file_name + '_Reg' + '.img')
        shutil.rmtree(temp_folder)
    except:
        pass

    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

#########################################################################

def FindNfitiDWIConfigFile(file_path, is_allow_vec_missing=True):
    file_name = os.path.splitext(file_path)[0]

    dwi_file = file_name + '.nii'
    dwi_bval_file = file_name + '.bval'
    dwi_vec_file = file_name + '.bvec'

    if os.path.exists(dwi_file) and os.path.exists(dwi_bval_file):
        if os.path.exists(dwi_vec_file):
            return dwi_file, dwi_bval_file, dwi_vec_file
        else:
            if is_allow_vec_missing:
                return dwi_file, dwi_bval_file, ''
            else:
                print('Check these files')
                return '', '', ''
    else:
        print('Check these files')
        return '', '', ''

def SeparateNfitiDWIFile(dwi_file_path):
    dwi_file, dwi_bval_file, _ = FindNfitiDWIConfigFile(dwi_file_path)
    if dwi_file and dwi_bval_file:
        dwi_4d = nb.load(dwi_file)

        with open(dwi_bval_file, 'r') as b_file:
            bvalue_list = b_file.read().split(' ')
        bvalue_list[-1] = bvalue_list[-1][:-1]


        dwi_list = nb.funcs.four_to_three(dwi_4d)
        if len(dwi_list) != len(bvalue_list):
            print('The list of the b values is not consistant to the dwi list')
            return

        for one_dwi, one_b in zip(dwi_list, bvalue_list):
            store_path = os.path.splitext(dwi_file)[0] + '_b' + one_b + '.nii'
            nb.save(one_dwi, store_path)

def ExtractBvalues(candidate_list):
    b_value = []
    for file in candidate_list:
        b_str = ''
        index = -5
        while True:
            if file[index].isdigit():
                b_str = file[index] + b_str
            else:
                b_value.append(int(b_str))
                break
            index -= 1

    return b_value

def FindDWIFile(candidate_list, is_separate=False):
    dwi_list = []
    if is_separate:
        for dwi in candidate_list:
            if (('dwi' in dwi) or ('diff' in dwi)) and ('_b' in dwi) and (('.nii' in dwi) or ('.nii.gz' in dwi)) and \
                    ('Reg' not in dwi) and ('Resize' not in dwi):
                dwi_list.append(dwi)
    else:
        for dwi in candidate_list:
            if (('dwi' in dwi) or ('diff' in dwi)) and ('_b' not in dwi) and (('.nii' in dwi) or ('.nii.gz' in dwi)) and\
                    ('Reg' not in dwi) and ('Resize' not in dwi):
                dwi_list.append(dwi)
    return dwi_list


################################################################################
# def SimulateDWI(adc_image, low_b_value_image, low_b_value, target_b_value, target_file_path, ref=''):
#     if isinstance(adc_image, str):
#         adc_image = sitk.ReadImage(adc_image)
#     if isinstance(low_b_value_image, str):
#         low_b_value_image = sitk.ReadImage(low_b_value_image)
#
#     ref_image = sitk.ReadImage(ref)
#
#     adc_array = GetDataFromSimpleITK(adc_image, dtype=np.float32)[1]
#     low_b_value_array = GetDataFromSimpleITK(low_b_value_image, dtype=np.float32)[1]
#     ref_array = GetDataFromSimpleITK(ref_image, dtype=np.float32)[1]
#     target_b_value_array = low_b_value_array * np.exp(-1 * adc_array / 4097. / 256. * float((target_b_value - low_b_value)))
#
#     from MeDIT.Visualization import Imshow3D
#     from MeDIT.Normalize import Normalize01
#     Imshow3D(np.concatenate((Normalize01(adc_array), Normalize01(low_b_value_array),
#                              Normalize01(target_b_value_array), Normalize01(ref_array)), axis=1))

