from copy import deepcopy
import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy import ndimage


def DetectRegionBlurry(prostate_roi, hard_dist=5, soft_dist=5):
    prostate_roi = binary_dilation(prostate_roi, np.ones((3, 3)), iterations=hard_dist)

    prob = np.zeros(prostate_roi.shape, dtype=np.float32)
    weights = 1. / soft_dist

    mask = prostate_roi
    for index in range(soft_dist):
        prob += weights * mask
        mask = binary_dilation(mask, np.ones((3, 3)))

    return prob

def BluryEdgeOfROI(initial_ROI):
    '''
    This function blurry the ROI. This function can be used when the ROI was drawn not definitely.
    :param initial_ROI: The binary ROI image, support 2D and 3D
    :return:
    '''
    if len(np.shape(initial_ROI)) == 2:
        kernel = np.ones((3, 3))
    elif len(np.shape(initial_ROI)) == 3:
        kernel = np.ones((3, 3, 3))
    else:
        print('Only could process 2D or 3D data')
        return []

    initial_ROI = initial_ROI == 1

    ROI_dilate = binary_dilation(input=initial_ROI, structure=kernel, iterations=1)
    ROI_erode = binary_erosion(input=ROI_dilate, structure=kernel, iterations=1)
    ROI_erode1 = binary_erosion(input=ROI_erode, structure=kernel, iterations=1)
    ROI_erode2 = binary_erosion(input=ROI_erode1, structure=kernel, iterations=1)

    dif_dilate = (ROI_dilate - ROI_erode) * 0.25
    dif_ori = (ROI_erode - ROI_erode1) * 0.5
    dif_in = (ROI_erode1 - ROI_erode2) * 0.75

    blurred_ROI = ROI_erode2 + dif_dilate + dif_ori + dif_in

    return blurred_ROI

def FindBoundaryOfBinaryMask(mask):
    '''
    Find the Boundary of the binary mask. Which was used based on the dilation process.
    :param mask: the binary mask
    :return:
    '''
    kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    boundary = binary_dilation(input=mask, structure=kernel, iterations=1) - mask
    return boundary

def RemoveSmallRegion(mask, size_thres=50):
    # seperate each connected ROI
    label_im, nb_labels = ndimage.label(mask)

    # remove small ROI
    for i in range(1, nb_labels + 1):
        if (label_im == i).sum() < size_thres:
            # remove the small ROI in mask
            mask[label_im == i] = 0
    return mask

def Remove2DSmallPhysicalRegion(mask, image, physical_region=50):
    # seperate each connected ROI
    size_thres = physical_region / (image.GetSpacing()[0] * image.GetSpacing()[1])
    return RemoveSmallRegion(mask, size_thres=size_thres)

### Transfer index to position #######################################################################################
def Index2XY(index, data_shape):
    '''
    Transfer the index to the x, y index based on the 2D image shape.
    :param index: The index list
    :param data_shape: The shape of the image.
    :return: The list of the x, y index.
    '''

    if np.max(index) >= data_shape[0] * data_shape[1]:
        print('The index is out of the range.')
        return []

    y = np.mod(index, data_shape[1])
    x = np.floor_divide(index, data_shape[1])
    return [x, y]

def XY2Index(position, data_shape):
    '''
    Transfer the x, y position to the index if flatten the 2D image.
    :param position: the point index with x and y
    :param data_shape: The shape of the image
    :return: the index of the flatted 1D vector.
    '''
    return position[0] * data_shape[1] + position[1]

def Index2XYZ(index, data_shape):
    '''
    Transfer the index to the x, y, z index based on the 3D image shape.
    :param index: The index index
    :param data_shape: The shape of the image.
    :return: The list of the x, y, z index.
    '''
    if np.max(index) >= data_shape[0] * data_shape[1] * data_shape[2]:
        print('The index is out of the range.')
        return []

    z = np.mod(index, data_shape[2])
    y = np.mod(np.floor_divide((index - z), data_shape[2]), data_shape[1])
    x = np.floor_divide(index, data_shape[2] * data_shape[1])
    return [x, y, z]

def XYZ2Index(position, data_shape):
    '''
    Transfer the x, y, z position to the index if flatten the 3D image.
    :param position: the point index with x and y
    :param data_shape: The shape of the image
    :return: the index of the flatted 1D vector.
    '''
    return position[0] * (data_shape[1] * data_shape[2]) + position[1] * data_shape[2] + position[2]

### Extract Patch from the image #######################################################################################
def ExtractPatch(image, patch_size, center_point=[-1, -1], is_shift=True):
    '''
    Extract patch from a 2D image.
    :param image: the 2D numpy array
    :param patch_size: the size of the 2D patch
    :param center_point: the center position of the patch
    :param is_shift: If the patch is too close to the edge of the image, is it allowed to shift the patch in order to
    ensure that extracting the patch close to the edge. Default is True.
    :return: the extracted patch.
    '''
    patch_size = np.asarray(patch_size)
    if patch_size.shape == () or patch_size.shape == (1,):
        patch_size = np.array([patch_size[0], patch_size[0]])

    image_row, image_col = np.shape(image)
    catch_x_index = np.arange(patch_size[0] // 2, image_row - (patch_size[0] // 2))
    catch_y_index = np.arange(patch_size[1] // 2, image_col - (patch_size[1] // 2))

    if center_point == [-1, -1]:
        center_point[0] = image_row // 2
        center_point[1] = image_col // 2

    if patch_size[0] > image_row or patch_size[1] > image_col:
        print('The patch_size is larger than image shape')
        return np.array([])

    if center_point[0] < catch_x_index[0]:
        if is_shift:
            center_point[0] = catch_x_index[0]
        else:
            print('The center point is too close to the negative x-axis')
            return []
    if center_point[0] > catch_x_index[-1]:
        if is_shift:
            center_point[0] = catch_x_index[-1]
        else:
            print('The center point is too close to the positive x-axis')
            return []
    if center_point[1] < catch_y_index[0]:
        if is_shift:
            center_point[1] = catch_y_index[0]
        else:
            print('The center point is too close to the negative y-axis')
            return []
    if center_point[1] > catch_y_index[-1]:
        if is_shift:
            center_point[1] = catch_y_index[-1]
        else:
            print('The center point is too close to the positive y-axis')
            return []

    patch_row_index = [center_point[0] - patch_size[0] // 2, center_point[0] + patch_size[0] - patch_size[0] // 2]
    patch_col_index = [center_point[1] - patch_size[1] // 2, center_point[1] + patch_size[1] - patch_size[1] // 2]

    patch = deepcopy(image[patch_row_index[0]:patch_row_index[1], patch_col_index[0]:patch_col_index[1]])
    return patch, [patch_row_index, patch_col_index]

def ExtractBlock(image, patch_size, center_point=[-1, -1, -1], is_shift=False):
    '''
    Extract patch from a 3D image.
    :param image: the 3D numpy array
    :param patch_size: the size of the 3D patch
    :param center_point: the center position of the patch
    :param is_shift: If the patch is too close to the edge of the image, is it allowed to shift the patch in order to
    ensure that extracting the patch close to the edge. Default is True.
    :return: the extracted patch.
    '''
    if not isinstance(center_point, list):
        center_point = list(center_point)
    patch_size = np.asarray(patch_size)
    if patch_size.shape == () or patch_size.shape == (1,):
        patch_size = np.array([patch_size[0], patch_size[0], patch_size[0]])

    image_row, image_col, image_slice = np.shape(image)
    catch_x_index = np.arange(patch_size[0] // 2, image_row - (patch_size[0] // 2))
    catch_y_index = np.arange(patch_size[1] // 2, image_col - (patch_size[1] // 2))
    if patch_size[2] == image_slice:
        catch_z_index = [patch_size[2] // 2]
    else:
        catch_z_index = np.arange(patch_size[2] // 2, image_slice - (patch_size[2] // 2))

    if center_point == [-1, -1, -1]:
        center_point[0] = image_row // 2
        center_point[1] = image_col // 2
        center_point[2] = image_slice // 2

    if patch_size[0] > image_row or patch_size[1] > image_col or patch_size[2] > image_slice:
        print('The patch_size is larger than image shape')
        return np.array()

    if center_point[0] < catch_x_index[0]:
        if is_shift:
            center_point[0] = catch_x_index[0]
        else:
            print('The center point is too close to the negative x-axis')
            return np.array()
    if center_point[0] > catch_x_index[-1]:
        if is_shift:
            center_point[0] = catch_x_index[-1]
        else:
            print('The center point is too close to the positive x-axis')
            return np.array()
    if center_point[1] < catch_y_index[0]:
        if is_shift:
            center_point[1] = catch_y_index[0]
        else:
            print('The center point is too close to the negative y-axis')
            return np.array()
    if center_point[1] > catch_y_index[-1]:
        if is_shift:
            center_point[1] = catch_y_index[-1]
        else:
            print('The center point is too close to the positive y-axis')
            return np.array()
    if center_point[2] < catch_z_index[0]:
        if is_shift:
            center_point[2] = catch_z_index[0]
        else:
            print('The center point is too close to the negative z-axis')
            return np.array()
    if center_point[2] > catch_z_index[-1]:
        if is_shift:
            center_point[2] = catch_z_index[-1]
        else:
            print('The center point is too close to the positive z-axis')
            return np.array()
    #
    # if np.shape(np.where(catch_x_index == center_point[0]))[1] == 0 or \
    #     np.shape(np.where(catch_y_index == center_point[1]))[1] == 0 or \
    #     np.shape(np.where(catch_z_index == center_point[2]))[1] == 0:
    #     print('The center point is too close to the edge of the image')
    #     return []

    block_row_index = [center_point[0] - patch_size[0] // 2, center_point[0] + patch_size[0] - patch_size[0] // 2]
    block_col_index = [center_point[1] - patch_size[1] // 2, center_point[1] + patch_size[1] - patch_size[1] // 2]
    block_slice_index = [center_point[2] - patch_size[2] // 2, center_point[2] + patch_size[2] - patch_size[2] // 2]

    block = deepcopy(image[block_row_index[0]:block_row_index[1], block_col_index[0]:block_col_index[1], block_slice_index[0]:block_slice_index[1]])
    return block, [block_row_index, block_col_index, block_slice_index]

def Crop2DImage(image, shape):
    '''
    Crop the size of the image. If the shape of the result is smaller than the image, the edges would be cut. If the size
    of the result is larger than the image, the edge would be filled in 0.
    :param image: the 2D numpy array
    :param shape: the list of the shape.
    :return: the cropped image.
    '''
    if image.shape[0] >= shape[0]:
        center = image.shape[0] // 2
        if shape[0] % 2 == 0:
            new_image = image[center - shape[0] // 2: center + shape[0] // 2, :]
        else:
            new_image = image[center - shape[0] // 2: center + shape[0] // 2 + 1, :]
    else:
        new_image = np.zeros((shape[0], image.shape[1]))
        center = shape[0] // 2
        if image.shape[0] % 2 ==0:
            new_image[center - image.shape[0] // 2: center + image.shape[0] // 2, :] = image
        else:
            new_image[center - image.shape[0] // 2 - 1: center + image.shape[0] // 2, :] = image


    image = new_image
    if image.shape[1] >= shape[1]:
        center = image.shape[1] // 2
        if shape[1] % 2 == 0:
            new_image = image[:, center - shape[1] // 2: center + shape[1] // 2]
        else:
            new_image = image[:, center - shape[1] // 2: center + shape[1] // 2 + 1]
    else:
        new_image = np.zeros((image.shape[0], shape[1]))
        center = shape[1] // 2
        if image.shape[1] % 2 ==0:
            new_image[:, center - image.shape[1] // 2: center + image.shape[1] // 2] = image
        else:
            new_image[:, center - image.shape[1] // 2 - 1: center + image.shape[1] // 2] = image

    return new_image

def Crop3DImage(image, shape):
    '''
    Crop the size of the image. If the shape of the result is smaller than the image, the edges would be cut. If the size
    of the result is larger than the image, the edge would be filled in 0.
    :param image: the 3D numpy array
    :param shape: the list of the shape.
    :return: the cropped image.
    '''
    if image.shape[0] >= shape[0]:
        center = image.shape[0] // 2
        if shape[0] % 2 == 0:
            new_image = image[center - shape[0] // 2: center + shape[0] // 2, :, :]
        else:
            new_image = image[center - shape[0] // 2: center + shape[0] // 2 + 1, :, :]
    else:
        new_image = np.zeros((shape[0], image.shape[1], image.shape[2]))
        center = shape[0] // 2
        if image.shape[0] % 2 == 0:
            new_image[center - image.shape[0] // 2: center + image.shape[0] // 2, :, :] = image
        else:
            new_image[center - image.shape[0] // 2 - 1: center + image.shape[0] // 2, :, :] = image

    image = new_image
    if image.shape[1] >= shape[1]:
        center = image.shape[1] // 2
        if image.shape[1] % 2 == 0:
            new_image = image[:, center - shape[1] // 2: center + shape[1] // 2, :]
        else:
            new_image = image[:, center - shape[1] // 2: center + shape[1] // 2 + 1, :]

    else:
        new_image = np.zeros((image.shape[0], shape[1], image.shape[2]))
        center = shape[1] // 2
        if image.shape[1] % 2 == 0:
            new_image[:, center - image.shape[1] // 2: center + image.shape[1] // 2, :] = image
        else:
            new_image[:, center - image.shape[1] // 2 - 1: center + image.shape[1] // 2, :] = image

    image = new_image
    if image.shape[2] >= shape[2]:
        center = image.shape[2] // 2
        if shape[2] % 2 == 0:
            new_image = image[:, :, center - shape[2] // 2: center + shape[2] // 2]
        else:
            new_image = image[:, :, center - shape[2] // 2: center + shape[2] // 2 + 1]
    else:
        new_image = np.zeros((image.shape[0], image.shape[1], shape[2]))
        center = shape[2] // 2
        if image.shape[2] % 2 == 0:
            new_image[:, :, center - image.shape[2] // 2: center + image.shape[2] // 2] = image
        else:
            new_image[:, :, center - image.shape[2] // 2 - 1: center + image.shape[2] // 2] = image

    return new_image

def GetIndexRangeInROI(roi_mask, target_value=1):
    if np.ndim(roi_mask) == 2:
        x, y = np.where(roi_mask == target_value)
        x = np.unique(x)
        y = np.unique(y)
        return x, y
    elif np.ndim(roi_mask) == 3:
        x, y, z = np.where(roi_mask == target_value)
        x = np.unique(x)
        y = np.unique(y)
        z = np.unique(z)
        return x, y, z
