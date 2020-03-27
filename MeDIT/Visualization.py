from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import imread
import os
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg


from MeDIT.Normalize import Normalize01
from MeDIT.ArrayProcess import Index2XY
def HSI2RGBDATA(hsi):
    rgb = np.copy(hsi)
    if(hsi[0]>=0 and hsi[0]< 2/3*np.pi):
        H = hsi[0]
        S = hsi[1]
        I = hsi[2]
        item0 = I * (1 - S)
        item1 = I * (1 + (S * np.cos(H) / (np.cos(1 / 3 * np.pi - H))))
        item2 = 3 * I - (item0 + item1)
        rgb[2] = item0
        rgb[0] = item1
        rgb[1] = item2
    if (hsi[0] >= 2 / 3 * np.pi and hsi[0] < 4 / 3 * np.pi):
        H = hsi[0]-2 / 3 * np.pi
        S = hsi[1]
        I = hsi[2]
        item0 = I * (1 - S)
        item1 = I * (1 + (S * np.cos(H) / (np.cos(1 / 3 * np.pi - H))))
        item2 = 3 * I - (item0 + item1)
        rgb[0] = item0
        rgb[1] = item1
        rgb[2] = item2
    if (hsi[0] >= 4 / 3 * np.pi and hsi[0] <= 2 * np.pi):
        H = hsi[0] - 4 / 3 * np.pi
        S = hsi[1]
        I = hsi[2]
        item0 = I * (1 - S)
        item1 = I * (1 + (S * np.cos(H) / (np.cos(1 / 3 * np.pi - H))))
        item2 = 3 * I - (item0 + item1)
        rgb[1] = item0
        rgb[2] = item1
        rgb[0] = item2
    if(np.max(rgb)>1):
        rgb = 1/np.max(rgb)*rgb
    return rgb
def HSI2RGBIMG(hsi_image):
    rgb_image = np.copy(hsi_image)
    rg_mask = (hsi_image[...,0]>=0 and hsi_image[...,0]<2/3*np.pi)
    rg_idx = np.where(rg_mask)
    H = hsi_image[rg_idx,0]
    S = hsi_image[rg_idx,1]
    I = hsi_image[rg_idx, 2]
    item0 = I*(1-S)
    item1 = I * (1 + (S*np.cos(H)/(np.cos(1/3*np.pi-H))))
    item2 = 3*I - (item0+item1)
    rgb_image[rg_idx,2] =item0
    rgb_image[rg_idx, 0] = item1
    rgb_image[rg_idx, 1] = item2

    gb_mask = (hsi_image[..., 0] >= 2 / 3 * np.pi and hsi_image[..., 0] < 4 / 3 * np.pi)
    gb_idx = np.where(gb_mask)
    H = hsi_image[gb_idx, 0]-2 / 3 * np.pi
    S = hsi_image[gb_idx, 1]
    I = hsi_image[gb_idx, 2]
    item0 = I * (1 - S)
    item1 = I * (1 + (S * np.cos(H) / (np.cos(1 / 3 * np.pi - H))))
    item2 = 3 * I - (item0 + item1)

    rgb_image[gb_idx, 0] = item0
    rgb_image[gb_idx, 1] = item1
    rgb_image[gb_idx, 2] = item2

    br_mask = (hsi_image[..., 0] >= 4 / 3 * np.pi and hsi_image[..., 0] <= 2 * np.pi)
    br_idx = np.where(br_mask)
    H = hsi_image[br_idx, 0] - 4 / 3 * np.pi
    S = hsi_image[br_idx, 1]
    I = hsi_image[br_idx, 2]
    item0 = I * (1 - S)
    item1 = I * (1 + (S * np.cos(H) / (np.cos(1 / 3 * np.pi - H))))
    item2 = 3 * I - (item0 + item1)

    rgb_image[br_idx, 1] = item0
    rgb_image[br_idx, 2] = item1
    rgb_image[br_idx, 0] = item2

    return rgb_image


def DrawBoundaryOfBinaryMask(image, ROI):
    '''
    show the image with ROIs
    :param image: the 2D image
    :param ROI: the binary ROI with same size of the image
    :return:
    '''
    plt.imshow(image, cmap='Greys_r')
    plt.contour(ROI, colors='r')
    plt.axis('off')
    plt.show()

def LoadWaitBar(total, progress):
    '''
    To show the waitbar for visulization
    :param total: the number of the total step
    :param progress: the number of the current processing
    :return:
    '''
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()

class IndexTracker(object):
    def __init__(self, ax, X, vmin, vmax, ROI):
        self.ax = ax

        self.X = X
        self.ROI = ROI
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', vmin=vmin, vmax=vmax)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = np.clip(self.ind + 1, 0, self.slices - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.slices - 1)
        self.update()

    def FindBoundaryOfROI(self):
        kernel = np.ones((3, 3))
        if isinstance(self.ROI, list):
            boundary_list = []
            for roi in self.ROI:
                boundary_list.append(binary_dilation(input=roi[:, :, self.ind], structure=kernel, iterations=1) - roi[:, :, self.ind])
            return boundary_list
        else:
            ROI_dilate = binary_dilation(input=self.ROI[:, :, self.ind], structure=kernel, iterations=1)
            return ROI_dilate - self.ROI[:, :, self.ind]


    def MergeDataWithROI(self, boundary):
        if isinstance(boundary, list):
            imshow_data = np.stack((self.X[:, :, self.ind], self.X[:, :, self.ind], self.X[:, :, self.ind]), axis=2)
            index_x, index_y = np.where(boundary[0] == 1)
            imshow_data[index_x, index_y, :] = 0
            imshow_data[index_x, index_y, 0] = np.max(self.X[:, :, self.ind])
            if len(boundary) > 1:
                index_x, index_y = np.where(boundary[1] == 1)
                imshow_data[index_x, index_y, :] = 0
                imshow_data[index_x, index_y, 1] = np.max(self.X[:, :, self.ind])
            if len(boundary) > 2:
                index_x, index_y = np.where(boundary[2] == 1)
                imshow_data[index_x, index_y, :] = 0
                imshow_data[index_x, index_y, 2] = np.max(self.X[:, :, self.ind])
        else:
            imshow_data = np.stack((self.X[:, :, self.ind], self.X[:, :, self.ind], self.X[:, :, self.ind]), axis=2)
            index_x, index_y = np.where(boundary == 1)
            imshow_data[index_x, index_y, :] = 0
            imshow_data[index_x, index_y, 0] = np.max(self.X[:, :, self.ind])
        return imshow_data

    def update(self):
        if isinstance(self.ROI, list):
            imshow_data = self.MergeDataWithROI(self.FindBoundaryOfROI())
            self.im.set_data(imshow_data)
        elif np.max(self.ROI) == 0:
            self.im.set_data(self.X[:, :, self.ind])
        else:
            imshow_data = self.MergeDataWithROI(self.FindBoundaryOfROI())
            self.im.set_data(imshow_data)

        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        # if self.ROI.any() == None:
        #     print('There is no ROI')
        # else:
        #     self.im = ax.contour(self.ROI[:, :, self.ind], colors='y')

def Imshow3D(data, vmin=None, vmax=None, ROI=0, name=' '):
    fig, ax = plt.subplots(1, 1, )
    tracker = IndexTracker(ax, data, vmin, vmax, ROI)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.set_window_title(name)
    plt.show()

################################################

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    ax.set_title('slice %d' % ax.index)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[..., ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[..., ax.index])

def Imshow3DConsole(volume, vmin=None, vmax=None):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[2] // 2
    ax.imshow(volume[..., ax.index], cmap='Greys_r', vmin=vmin, vmax=vmax)
    fig.canvas.mpl_connect('key_press_event', process_key)

##############################################################
def FlattenImages(data_list, is_show=True):
    if len(data_list) == 1:
        return data_list[0]
    width = 1

    if data_list[0].ndim == 2:
        row, col = data_list[0].shape
        for one_data in data_list:
            temp_row, temp_col = one_data.shape
            assert(temp_row == row and temp_col == col)

        while True:
            if width * width >= len(data_list):
                break
            else:
                width += 1
        imshow_data = np.zeros((row * width, col * width))
        case_index = range(0, len(data_list))
        x, y = Index2XY(case_index, (width, width))

        for x_index, y_index, index in zip(x, y, case_index):
            imshow_data[x_index * row: (x_index + 1) * row, y_index * col: (y_index + 1) * col] = data_list[index]

        if is_show:
            plt.imshow(Normalize01(imshow_data), cmap='gray')
            plt.show()
        return imshow_data

    elif data_list[0].ndim == 3:
        row, col, slice = data_list[0].shape
        for one_data in data_list:
            temp_row, temp_col, temp_slice = one_data.shape
            assert (temp_row == row and temp_col == col and temp_slice == slice)

        while True:
            if width * width >= len(data_list):
                break
            else:
                width += 1
        imshow_data = np.zeros((row * width, col * width, slice))
        case_index = range(0, len(data_list))
        x, y = Index2XY(case_index, (width, width))

        for x_index, y_index, index in zip(x, y, case_index):
            imshow_data[x_index * row: (x_index + 1) * row, y_index * col: (y_index + 1) * col, :] = data_list[index]

        if is_show:
            Imshow3DArray(Normalize01(imshow_data), cmap='gray')

        return imshow_data

def FlattenAllSlices(data, is_show=True):
    assert(data.ndim == 3)
    row, col, slice = data.shape
    width = 1
    while True:
        if width * width >= slice:
            break
        else:
            width += 1
    imshow_data = np.zeros((row * width, col * width))
    slice_indexs = range(0, slice)
    x, y = Index2XY(slice_indexs, (width, width))

    for x_index, y_index, slice_index in zip(x, y, slice_indexs):
        imshow_data[x_index * row : (x_index + 1) * row, y_index * col : (y_index + 1) * col] = data[..., slice_index]

    if is_show:
        plt.imshow(Normalize01(imshow_data), cmap='gray')
        plt.show()
    return imshow_data

################################################################
# 该函数将每个2d图像进行变换。
def MergeImageWithROI(data, roi, overlap=False,max_value=1,min_value=0):
    if data.ndim >= 3:
        print("Should input 2d image")
        return data

    if not isinstance(roi, list):
        roi = [roi]
    num_of_roi = len(roi)
    delta_H = 2*np.pi/num_of_roi

    # if len(roi) > 3:
    #     print('Only show 3 ROIs')
    #     return data
    #intensity = max_value
    #data = np.asarray(data * intensity, dtype=np.float)

    if overlap:
        new_data = np.stack([data, data, data], axis=2)
        for i in range(num_of_roi):
            hsi = [delta_H*i,1,1]
            rgb = HSI2RGBDATA(hsi)*(max_value-min_value)+min_value
            index_x, index_y = np.where(roi[i] == 1)
            new_data[index_x, index_y, 0] = rgb[0]
            new_data[index_x, index_y, 1] = rgb[1]
            new_data[index_x, index_y, 2] = rgb[2]

    else:
        kernel = np.ones((3, 3))
        new_data = np.stack([data, data, data], axis=2)
        for i in range(num_of_roi):
            hsi = [delta_H*i,1,1]
            rgb = HSI2RGBDATA(hsi)*(max_value-min_value)+min_value
            boundary = binary_dilation(input=roi[i], structure=kernel, iterations=1)
            boundary = boundary-roi[i]
            index_x, index_y = np.where(boundary == 1)
            new_data[index_x, index_y, 0] = rgb[0]
            new_data[index_x, index_y, 1] = rgb[1]
            new_data[index_x, index_y, 2] = rgb[2]
    return new_data

def Merge3DImageWithROI(data, roi, overlap=False):
    if not isinstance(roi, list):
        roi = [roi]

    # if len(roi) > 3:
    #     print('Only show 3 ROIs')
    #     return data

    new_data = np.zeros((data.shape[2], data.shape[0], data.shape[1], 3))
    #data = Normalize01(data)
    max_value = np.max(data)
    min_value = np.min(data)
    for slice_index in range(data.shape[2]):
        slice = data[..., slice_index]
        one_roi_list = []
        for one_roi in roi:
            one_roi_list.append(one_roi[..., slice_index])

        new_data[slice_index, ...] = MergeImageWithROI(slice, one_roi_list, overlap=overlap,max_value=max_value,min_value=min_value)

    return new_data

def FusionImage(data, mask, is_show=False):
    '''
    To Fusion two 2D images.
    :param data: The background
    :param mask: The fore-ground
    :param is_show: Boolen. If set to True, to show the result; else to return the fusion image. (RGB).
    :return:
    '''
    if data.ndim >= 3:
        print("Should input 2d image")
        return data

    dpi = 96
    x, y = data.shape
    w = y / dpi
    h = x / dpi

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(data, cmap='gray')
    plt.imshow(mask, cmap='rainbow', alpha=0.3)

    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)

    if is_show:
        plt.show()
    else:
        plt.axis('off')
        plt.savefig('temp.jpg', format='jpeg', aspect='normal', bbox_inches='tight', pad_inches=0.0)
        array = imread('temp.jpg')
        os.remove('temp.jpg')
        return array

def ShowColorByROI(background_array, fore_array, roi, threshold_value = 1e-6, color_map='rainbow', store_path='', is_show=True):
    if background_array.shape != roi.shape:
        print('Array and ROI must have same shape')
        return

    background_array = Normalize01(background_array)
    fore_array = Normalize01(fore_array)
    cmap = plt.get_cmap(color_map)
    rgba_array = cmap(fore_array)
    rgb_array = np.delete(rgba_array, 3, 2)

    print(background_array.shape)
    print(rgb_array.shape)

    index_roi_x, index_roi_y = np.where(roi < threshold_value)
    for index_x, index_y in zip(index_roi_x, index_roi_y):
        rgb_array[index_x, index_y, :] = background_array[index_x, index_y]

    plt.imshow(rgb_array)
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if store_path:
        plt.savefig(store_path, format='tif', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    if is_show:
        plt.show()
    return rgb_array

def Imshow3DArray(data, ROI=None, window_size=[800, 800], window_name='Imshow3D', overlap=False):
    '''
    Imshow 3D Array, the dimension is row x col x slice. If the ROI was combined in the data, the dimension is:
    slice x row x col x color
    :param data: 3D Array [row x col x slice] or 4D array [slice x row x col x RGB]
    '''

    if isinstance(ROI, list) or isinstance(ROI, type(data)):
        data = Merge3DImageWithROI(data, ROI, overlap=overlap)

    if np.ndim(data) == 3:
        data = np.swapaxes(data, 0, 1)
        data = np.transpose(data)



    pg.setConfigOptions(imageAxisOrder='row-major')
    app = QtGui.QApplication([])

    win = QtGui.QMainWindow()
    win.resize(window_size[0], window_size[1])

    imv = pg.ImageView()
    min_value = np.min(data)
    max_value = np.max(data)
    win.setCentralWidget(imv)
    win.show()
    win.setWindowTitle(window_name)
    imv.setImage(data,levels=(min_value,max_value),autoRange=False)
    app.exec()
    imv.clear()


def CheckROIForSeries(root_folder, store_folder, key, roi_key):
    import glob
    from scipy.misc import imsave
    from MeDIT.SaveAndLoad import LoadNiiData, SaveArrayAsGreyImage

    for case in os.listdir(root_folder):
        case_folder = os.path.join(root_folder, case)
        if not os.path.isdir(case_folder):
            continue

        print(case)
        key_path = glob.glob(os.path.join(case_folder, key))
        if len(key_path) != 1:
            print('More Key Image: ', case)
            continue
        image, _, data = LoadNiiData(key_path[0])
        show_data = FlattenAllSlices(data, is_show=False)

        show_roi_list = []
        roi_path_list = glob.glob(os.path.join(case_folder, roi_key))
        if len(roi_path_list) == 0:
            print('No ROI Image: ', case)
            continue
        for roi_path in roi_path_list:
            _, _, roi = LoadNiiData(roi_path, dtype=np.uint8)
            show_roi = FlattenAllSlices(roi, is_show=False)
            if show_data.shape != show_roi.shape:
                print('Data and ROI are not consistent: ', case)
            show_roi_list.append(show_roi)

        merge_data = MergeImageWithROI(show_data, show_roi_list)

        store_path = os.path.join(store_folder, case + '.jpg')
        imsave(store_path, merge_data)



