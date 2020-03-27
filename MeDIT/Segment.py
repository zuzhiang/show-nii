from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_closing
from scipy import ndimage as ndi
from skimage.filters import roberts
import numpy as np
import scipy.ndimage.interpolation as ImgResize
from skimage import morphology

from MeDIT.Normalize import Normalize01
import matplotlib.pyplot as plt

def SegmentLungAxialSlice(image, threshold_value=-374):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    # Convert into a binary image.
    image = Normalize01(image)


    binary = image < threshold_value
    # plt.imshow(binary, cmap=plt.cm.gray)

    # Remove the blobs connected to the border of the image
    cleared = clear_border(binary)

    # Label the image
    label_image = label(cleared)

    # Keep the labels with 2 largest areas
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    # Closure operation with disk of radius 12
    selem = disk(2)
    binary = binary_erosion(binary, selem)

    selem = disk(10)
    binary = binary_closing(binary, selem)

    # Fill in the small holes inside the lungs
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    # Superimpose the mask on the input image
    get_high_vals = binary == 0
    mask = np.ones(np.shape(image))
    mask[get_high_vals] = 0

    return mask

def SegmentLungAxialSlice1(image, threshold_value=-374):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    mask = np.zeros(image.shape)
    mask[image < threshold_value] = 1
    # 形态学操作
    final_mask = np.zeros(image.shape)

    temp = clear_border(mask)
    temp = temp.astype(np.bool)
    temp = morphology.remove_small_holes(temp, 512)
    temp = morphology.remove_small_objects(temp, 100)
    temp = temp.astype(np.float64)
    final_mask = temp

    return final_mask


    return mask

def SegmentLung(data, threshold_value=-374, shrink=0, method=1):
    '''

    :param data: the 3D data, slice was in the last axis
    :param threshold_value: To thershold the lung
    :param shrink: To crop the edges.
    :param method: 1 denotes the 'Kaggle' method, 2 denotes the Yang Zhi-wei script.
    :return: The 3D mask.
    '''
    data = np.asarray(data, dtype=np.float32)
    mask = np.zeros(np.shape(data))

    for slice in range(np.shape(data)[2]):
        if shrink == 0:
            if method == 1:
                mask[:, :, slice] = SegmentLungAxialSlice(data[:, :, slice], threshold_value)
            else:
                mask[:, :, slice] = SegmentLungAxialSliceYZW(data[:, :, slice], threshold_value)
        else:
            slice_data = np.squeeze(data[:, :, slice])
            row, col = slice_data.shape
            crop_row = int(row * (1 - shrink))
            crop_col = int(col * (1 - shrink))
            slice_data = slice_data[row // 2 - crop_row // 2 : row // 2 + crop_row // 2,
                         col // 2 - crop_col // 2: col // 2 + crop_col // 2]
            if method == 1:
                crop_mask = SegmentLungAxialSlice(slice_data, threshold_value)
            else:
                crop_mask = SegmentLungAxialSliceYZW(slice_data, threshold_value)
            mask[row // 2 - crop_row // 2 : row // 2 + crop_row // 2,
                         col // 2 - crop_col // 2: col // 2 + crop_col // 2, slice] = crop_mask
    return mask

def OtsuSegment(data):
    max_value = np.max(data)
    min_value = np.min(data)
    step = (max_value - min_value) / 100

    max_g = 0
    threshold_value = 0
    for threshold in np.arange(min_value, max_value, step):
        bin_img = data > threshold
        bin_img_inv = data <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue

        w0 = float(fore_pix) / data.size
        u0 = float(np.sum(data * bin_img)) / fore_pix
        w1 = float(back_pix) / data.size
        u1 = float(np.sum(data * bin_img_inv)) / back_pix
        # intra-class variance
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            threshold_value = threshold

    mask = np.zeros(data.shape, dtype=np.uint8)
    mask[data > threshold_value] = 1

    return threshold_value, mask

if __name__ == '__main__':
    image = np.load(r'C:\MyCode\PythonScript\NoduleDetection\slice_demo.npy')

    mask = np.zeros(image.shape)
    mask[image < -374] = 1
    # 形态学操作
    final_mask = np.zeros(image.shape)

    # temp = morphology.binary_opening(mask, morphology.disk(11))
    # plt.imshow(temp, cmap='gray'), plt.show()
    temp = clear_border(mask)
    plt.imshow(temp, cmap='gray'), plt.show()
    temp = temp.astype(np.bool)
    temp = morphology.remove_small_holes(temp, 512)
    plt.imshow(temp, cmap='gray'), plt.show()
    temp = morphology.remove_small_objects(temp, 100)
    plt.imshow(temp, cmap='gray'), plt.show()
    temp = temp.astype(np.float64)
    plt.imshow(temp, cmap='gray'), plt.show()
    final_mask = temp

    from Visualization import DrawBoundaryOfBinaryMask
    DrawBoundaryOfBinaryMask(image, final_mask)
