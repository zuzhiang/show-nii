import numpy as np
import h5py
import nibabel as nb
import os
import SimpleITK as sitk
import pydicom
import imageio
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt

from MeDIT.ImageProcess import GetImageFromArrayByImage
from MeDIT.Normalize import Normalize01

def FilePathSplitext(file_path):
    if file_path[-7:] == '.nii.gz':
        return file_path[:-7], '.nii.gz'
    else:
        return os.path.splitext(file_path)

def LoadCSVwithChineseInPandas(file_path, **kwargs):
    if 'encoding' not in kwargs.keys():
        return pd.read_csv(file_path, encoding="gbk", **kwargs)
    else:
        return pd.read_csv(file_path, **kwargs)

def SaveArrayAsGreyImage(array, store_path, roi=0, dip=300):
    # image = Normalize01(image)
    plt.imshow(array, cmap='Greys_r')
    if np.max(roi) != 0:
        plt.contour(roi, colors='g')
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(store_path, format='tif', dpi=dip, bbox_inches='tight', pad_inches=0)
    plt.close()

def SaveH5(store_path, data, tag, data_type=np.float32):
    if not isinstance(data, list):
        data = [data]
    if not isinstance(tag, list):
        tag = [tag]
    if not isinstance(data_type, list):
        data_type = [data_type for index in range(len(data))]

    file = h5py.File(store_path, 'w')
    for index in range(len(data)):
        file.create_dataset(tag[index], data=data[index], dtype=data_type[index])
    file.close()

def LoadH5(data_path, tag, data_type=np.float32):
    file = h5py.File(data_path, 'r')
    data = np.asarray(file[tag], dtype=data_type)
    file.close()
    return data

def SaveArrayAsTIFF(array, store_path, dpi=(300, 300)):
    import PIL.Image
    arr2im = PIL.Image.fromarray(array)
    arr2im.save(store_path, format='TIFF', dpi=dpi)

def LoadH5InfoForGenerate(data_path):
    file = h5py.File(data_path, 'r')
    info = {'input_number': 0, 'output_number': 0}
    key_list = []
    for key in file.keys():
        key_list.append(key)
        input_output, current_number = key.split('_')
        if input_output == 'input':
            info['input_number'] = np.max([int(current_number) + 1, info['input_number']])
        elif input_output == 'output':
            info['output_number'] = np.max([int(current_number) + 1, info['output_number']])
        else:
            print('Error:', data_path)

    return info

def LoadKerasWeightH5Info(data_path):
    '''
    Load the h5file and print all the weights.
    :param data_path: the path of the h5 file.
    :return:
    '''
    data_dict = h5py.File(data_path, 'r')
    for top_group_name in data_dict.keys():
        print(top_group_name)
        for group_name in data_dict[top_group_name].keys():
            print('    ' + group_name)
            for data_name in data_dict[top_group_name][group_name].keys():
                print('        ' + data_name + ': ' + str(np.shape(data_dict[top_group_name][group_name][data_name].value)))

def LoadNiiData(file_path, dtype=np.float32, is_show_info=False, is_flip=True, flip_log=[0, 0, 0]):
    image = sitk.ReadImage(file_path)
    data = np.asarray(sitk.GetArrayFromImage(image), dtype=dtype)

    show_data = np.transpose(data)
    show_data = np.swapaxes(show_data, 0, 1)

    # To process the flip cases
    if is_flip:
        direction = image.GetDirection()
        for direct_index in range(3):
            direct_vector = [direction[0 + direct_index * 3], direction[1 + direct_index * 3], direction[2 + direct_index * 3]]
            abs_max_point = np.argmax(np.abs(direct_vector))
            if direct_vector[abs_max_point] < 0:
                show_data = np.flip(show_data, axis=direct_index)
                flip_log[direct_index] = 1

    if is_show_info:
        print('Image size is: ', image.GetSize())
        print('Image resolution is: ', image.GetSpacing())
        print('Image direction is: ', image.GetDirection())
        print('Image Origion is: ', image.GetOrigin())

    return image, data, show_data

def LoadNiiHeader(file_path, is_show_info=True):
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    info = OrderedDict()
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        info[k] = v

    dimension = int(info['dim[0]'])
    spaceing = []
    center = [float(info['qoffset_x']), float(info['qoffset_y']), float(info['qoffset_z'])]
    shape = []
    for d in range(dimension):
        spaceing.append(float(info['pixdim[' + str(d+1) + ']']))
        shape.append(int(info['dim[' + str(d + 1) + ']']))

    info['dimension'] = dimension
    info['spacing'] = spaceing
    info['center'] = center
    info['shape'] = shape

    if is_show_info:
        print(info)

    return info

def SaveNiiImage(store_path, image):
    sitk.WriteImage(image, store_path)

def SaveNumpyToImageByRef(store_path, data, ref_image):
    image = GetImageFromArrayByImage(data, ref_image)
    image.CopyInformation(ref_image)
    sitk.WriteImage(image, store_path)

def SaveDicomByRefer(data, dicom_data, store_path):
    if isinstance(dicom_data, str) and dicom_data[-3:] == 'dcm':
        dicom_data = pydicom.dcmread(dicom_data)

    try:
        assert(data.shape == dicom_data.pixel_array.shape)
    except:
        print("The data shape must be same with the shape of the dicom data. ")

    ds = deepcopy(dicom_data)
    ds.PixelData = data.tostring()
    ds.save_as(store_path)

def SaveSiemens2DDicomSeries(array, dicom_folder, store_folder):
    # Sort the array according the SliceLocation
    dicom_file_list = os.listdir(dicom_folder)
    dicom_file_list.sort()

    slice_location_list = []
    for dicom_file in dicom_file_list:
        dicom_data = pydicom.dcmread(os.path.join(dicom_folder, dicom_file))
        slice_location_list.append(float(dicom_data.SliceLocation))

    sort_index_list = sorted(range(len(slice_location_list)), key=lambda k: slice_location_list[k])
    sort_index_list = sorted(range(len(sort_index_list)), key=lambda k: sort_index_list[k])

    for dicom_file, store_index in zip(dicom_file_list, range(len(dicom_file_list))):
        dicom_data = pydicom.dcmread(os.path.join(dicom_folder, dicom_file))

        ds = deepcopy(dicom_data)
        ds.PixelData = array[..., sort_index_list[store_index]].tostring()
        ds.save_as(os.path.join(store_folder, str(store_index) + '.dcm'))

def LoadDicomData(data_path):
    ds = pydicom.dcmread(data_path)
    data = ds.pixel_array

    return ds, data

def SaveAsGif(image_list, store_path, duration=1):
    gif = []
    for image in image_list:
        gif.append(np.asarray(deepcopy(image), dtype=np.uint8))

    imageio.mimsave(store_path, gif, duration=duration)

def LoadFreeSurferLabel(ref_image_path, label_file, store_folder=''):
    if store_folder:
        if not os.path.exists(store_folder):
            os.mkdir(store_folder)

    def _GetLabel(label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()
            position_list = []
            for line in lines[2:]:
                one_line = line.split(' ')
                one_line = [index for index in one_line if index != '']
                position_list.append([float(one_line[1]), float(one_line[2]), float(one_line[3])])

        return position_list

    if not (isinstance(ref_image_path, str) and isinstance(label_file, str)):
        print('The input must be file path (string)')
    flip_log = [0, 0, 0]
    if ref_image_path.endswith('.mgz'):
        temp_image = nb.load(ref_image_path)
        if store_folder:
            nb.save(temp_image, os.path.join(store_folder, 'transfer_image.nii.gz'))
            image, data, show_data = LoadNiiData(os.path.join(store_folder, 'transfer_image.nii.gz'), is_flip=flip_log)
        else:
            temp_image_path = 'temp_image.nii.gz'
            nb.save(temp_image, temp_image_path)
            image, data, show_data = LoadNiiData(temp_image_path, is_flip=flip_log)
            os.remove(temp_image_path)

    if ref_image_path.endswith('.nii') or ref_image_path.endswith('.nii.gz'):
        image, data, show_data = LoadNiiData(ref_image_path, is_flip=flip_log)

    origion = image.GetOrigin()
    image.SetOrigin((128, 128, 128))
    position_list = _GetLabel(label_file)
    index_list = list(set([image.TransformPhysicalPointToIndex(position) for position in position_list]))
    mask = np.zeros((image.GetSize()), dtype=np.uint8)
    for index in index_list:
        mask[index[1], index[0], index[2]] = 1

    if store_folder:
        from MeDIT.ImageProcess import GetImageFromArrayByImage
        image.SetOrigin(origion)
        store_image = GetImageFromArrayByImage(mask, image, is_transfer_axis=True)
        SaveNiiImage(os.path.join(store_folder, 'label.nii.gz'), store_image)

    return show_data, mask

def LoadImageJROI(file_path):
    # Load ImageJ saved format
    suffix = os.path.splitext(file_path)[1]
    if suffix == 'roi':
        return read_roi_file(file_path)
    elif suffix == 'zip':
        return read_roi_zip(file_path)
    else:
        print('Only read .roi and .zip file')
        return None

####################################################################
'''
The following code was copyed from  https://github.com/hadim/read-roi. If I need to fork the raw source of the code 
form GitHub, please let me know. 
'''
import struct
import zipfile
import logging

__all__ = ['read_roi_file', 'read_roi_zip']
OFFSET = dict(VERSION_OFFSET=4,
              TYPE=6,
              TOP=8,
              LEFT=10,
              BOTTOM=12,
              RIGHT=14,
              N_COORDINATES=16,
              X1=18,
              Y1=22,
              X2=26,
              Y2=30,
              XD=18,
              YD=22,
              WIDTHD=26,
              HEIGHTD=30,
              STROKE_WIDTH=34,
              SHAPE_ROI_SIZE=36,
              STROKE_COLOR=40,
              FILL_COLOR=44,
              SUBTYPE=48,
              OPTIONS=50,
              ARROW_STYLE=52,
              ELLIPSE_ASPECT_RATIO=52,
              ARROW_HEAD_SIZE=53,
              ROUNDED_RECT_ARC_SIZE=54,
              POSITION=56,
              HEADER2_OFFSET=60,
              COORDINATES=64)

ROI_TYPE = dict(polygon=0,
                rect=1,
                oval=2,
                line=3,
                freeline=4,
                polyline=5,
                noRoi=6,
                freehand=7,
                traced=8,
                angle=9,
                point=10)

OPTIONS = dict(SPLINE_FIT=1,
               DOUBLE_HEADED=2,
               OUTLINE=4,
               OVERLAY_LABELS=8,
               OVERLAY_NAMES=16,
               OVERLAY_BACKGROUNDS=32,
               OVERLAY_BOLD=64,
               SUB_PIXEL_RESOLUTION=128,
               DRAW_OFFSET=256)

HEADER_OFFSET = dict(C_POSITION=4,
                     Z_POSITION=8,
                     T_POSITION=12,
                     NAME_OFFSET=16,
                     NAME_LENGTH=20,
                     OVERLAY_LABEL_COLOR=24,
                     OVERLAY_FONT_SIZE=28,
                     AVAILABLE_BYTE1=30,
                     IMAGE_OPACITY=31,
                     IMAGE_SIZE=32,
                     FLOAT_STROKE_WIDTH=36)

SUBTYPES = dict(TEXT=1,
                ARROW=2,
                ELLIPSE=3,
                IMAGE=4)

def get_byte(data, base):
    if isinstance(base, int):
        return data[base]
    elif isinstance(base, list):
        return [data[b] for b in base]

def get_short(data, base):
    b0 = data[base]
    b1 = data[base + 1]
    n = (b0 << 8) + b1
    return n

def get_int(data, base):
    b0 = data[base]
    b1 = data[base + 1]
    b2 = data[base + 2]
    b3 = data[base + 3]
    n = ((b0 << 24) + (b1 << 16) + (b2 << 8) + b3)
    return n

def get_float(data, base):
    s = struct.pack('I', get_int(data, base))
    return struct.unpack('f', s)[0]

def read_roi_file(fpath):
    """
    """

    if isinstance(fpath, zipfile.ZipExtFile):
        data = fpath.read()
        name = os.path.splitext(os.path.basename(fpath.name))[0]
    elif isinstance(fpath, str):
        fp = open(fpath, 'rb')
        data = fp.read()
        fp.close()
        name = os.path.splitext(os.path.basename(fpath))[0]
    else:
        logging.error("Can't read {}".format(fpath))
        return None

    logging.debug("Read ROI for \"{}\"".format(name))

    size = len(data)
    code = '>'

    roi = {}

    magic = get_byte(data, list(range(4)))
    magic = "".join([chr(c) for c in magic])

    # TODO: raise error if magic != 'Iout'

    version = get_short(data, OFFSET['VERSION_OFFSET'])
    roi_type = get_byte(data, OFFSET['TYPE'])
    subtype = get_short(data, OFFSET['SUBTYPE'])
    top = get_short(data, OFFSET['TOP'])
    left = get_short(data, OFFSET['LEFT'])

    if top > 6000:
        top -= 2**16
    if left > 6000:
        left -= 2**16

    bottom = get_short(data, OFFSET['BOTTOM'])
    right = get_short(data, OFFSET['RIGHT'])
    width = right - left
    height = bottom - top
    n_coordinates = get_short(data, OFFSET['N_COORDINATES'])
    options = get_short(data, OFFSET['OPTIONS'])
    position = get_int(data, OFFSET['POSITION'])
    hdr2Offset = get_int(data, OFFSET['HEADER2_OFFSET'])

    logging.debug("n_coordinates: {n_coordinates}")
    logging.debug("position: {position}")
    logging.debug("options: {options}")

    sub_pixel_resolution = (options == OPTIONS['SUB_PIXEL_RESOLUTION']) and version >= 222
    draw_offset = sub_pixel_resolution and (options == OPTIONS['DRAW_OFFSET'])
    sub_pixel_rect = version >= 223 and sub_pixel_resolution and (
        roi_type == ROI_TYPE['rect'] or roi_type == ROI_TYPE['oval'])

    logging.debug("sub_pixel_resolution: {sub_pixel_resolution}")
    logging.debug("draw_offset: {draw_offset}")
    logging.debug("sub_pixel_rect: {sub_pixel_rect}")

    # Untested
    if sub_pixel_rect:
        xd = getFloat(data, OFFSET['XD'])
        yd = getFloat(data, OFFSET['YD'])
        widthd = getFloat(data, OFFSET['WIDTHD'])
        heightd = getFloat(data, OFFSET['HEIGHTD'])
        logging.debug("Entering in sub_pixel_rect")

    # Untested
    if hdr2Offset > 0 and hdr2Offset + HEADER_OFFSET['IMAGE_SIZE'] + 4 <= size:
        channel = get_int(data, hdr2Offset + HEADER_OFFSET['C_POSITION'])
        slice = get_int(data, hdr2Offset + HEADER_OFFSET['Z_POSITION'])
        frame = get_int(data, hdr2Offset + HEADER_OFFSET['T_POSITION'])
        overlayLabelColor = get_int(data, hdr2Offset + HEADER_OFFSET['OVERLAY_LABEL_COLOR'])
        overlayFontSize = get_short(data, hdr2Offset + HEADER_OFFSET['OVERLAY_FONT_SIZE'])
        imageOpacity = get_byte(data, hdr2Offset + HEADER_OFFSET['IMAGE_OPACITY'])
        imageSize = get_int(data, hdr2Offset + HEADER_OFFSET['IMAGE_SIZE'])
        logging.debug("Entering in hdr2Offset")

    is_composite = get_int(data, OFFSET['SHAPE_ROI_SIZE']) > 0

    # Not implemented
    if is_composite:
        if version >= 218:
            pass
        if channel > 0 or slice > 0 or frame > 0:
            pass

    if roi_type == ROI_TYPE['rect']:
        roi = {'type': 'rectangle'}

        if sub_pixel_rect:
            roi.update(dict(left=xd, top=yd, width=widthd, height=heightd))
        else:
            roi.update(dict(left=left, top=top, width=width, height=height))

        roi['arc_size'] = get_short(data, OFFSET['ROUNDED_RECT_ARC_SIZE'])

    elif roi_type == ROI_TYPE['oval']:
        roi = {'type': 'oval'}

        if sub_pixel_rect:
            roi.update(dict(left=xd, top=yd, width=widthd, height=heightd))
        else:
            roi.update(dict(left=left, top=top, width=width, height=height))

    elif roi_type == ROI_TYPE['line']:
        roi = {'type': 'line'}

        x1 = get_float(data, OFFSET['X1'])
        y1 = get_float(data, OFFSET['Y1'])
        x2 = get_float(data, OFFSET['X2'])
        y2 = get_float(data, OFFSET['Y2'])

        if subtype == SUBTYPES['ARROW']:
            # Not implemented
            pass
        else:
            roi.update(dict(x1=x1, x2=x2, y1=y1, y2=y2))
            roi['draw_offset'] = draw_offset

    elif roi_type in [ROI_TYPE[t] for t in ["polygon", "freehand", "traced", "polyline", "freeline", "angle", "point"]]:
        x = []
        y = []
        base1 = OFFSET['COORDINATES']
        base2 = base1 + 2 * n_coordinates
        for i in range(n_coordinates):
            xtmp = get_short(data, base1 + i * 2)
            ytmp = get_short(data, base2 + i * 2)
            x.append(left + xtmp)
            y.append(top + ytmp)

        if sub_pixel_resolution:
            xf = []
            yf = []
            base1 = OFFSET['COORDINATES'] + 4 * n_coordinates
            base2 = base1 + 4 * n_coordinates
            for i in range(n_coordinates):
                xf.append(get_float(data, base1 + i * 4))
                yf.append(get_float(data, base2 + i * 4))

        if roi_type == ROI_TYPE['point']:
            roi = {'type': 'point'}

            if sub_pixel_resolution:
                roi.update(dict(x=xf, y=yf, n=n_coordinates))
            else:
                roi.update(dict(x=x, y=y, n=n_coordinates))

        if roi_type == ROI_TYPE['polygon']:
            roi = {'type': 'polygon'}

        elif roi_type == ROI_TYPE['freehand']:
            roi = {'type': 'freehand'}
            if subtype == SUBTYPES['ELLIPSE']:
                ex1 = get_float(data, OFFSET['X1'])
                ey1 = get_float(data, OFFSET['Y1'])
                ex2 = get_float(data, OFFSET['X2'])
                ey2 = get_float(data, OFFSET['Y2'])
                roi['aspect_ratio'] = get_float(
                    data, OFFSET['ELLIPSE_ASPECT_RATIO'])
                roi.update(dict(ex1=ex1, ey1=ey1, ex2=ex2, ey2=ey2))

        elif roi_type == ROI_TYPE['traced']:
            roi = {'type': 'traced'}

        elif roi_type == ROI_TYPE['polyline']:
            roi = {'type': 'polyline'}

        elif roi_type == ROI_TYPE['freeline']:
            roi = {'type': 'freeline'}

        elif roi_type == ROI_TYPE['angle']:
            roi = {'type': 'angle'}

        else:
            roi = {'type': 'freeroi'}

        if sub_pixel_resolution:
            roi.update(dict(x=xf, y=yf, n=n_coordinates))
            #roi.update(dict(x=x, y=y, n=n_coordinates))
        else:
            roi.update(dict(x=x, y=y, n=n_coordinates))
    else:
        # TODO: raise an error for 'Unrecognized ROI type'
        pass

    roi['name'] = name

    if version >= 218:
        # Not implemented
        # Read stroke width, stroke color and fill color
        pass

    if version >= 218 and subtype == SUBTYPES['TEXT']:
        # Not implemented
        # Read test ROI
        pass

    if version >= 218 and subtype == SUBTYPES['IMAGE']:
        # Not implemented
        # Get image ROI
        pass

    roi['position'] = position
    if channel > 0 or slice > 0 or frame > 0:
        roi['position'] = dict(channel=channel, slice=slice, frame=frame)

    return {name: roi}

def read_roi_zip(zip_path):
    """
    """
    from collections import OrderedDict
    rois = OrderedDict()
    zf = zipfile.ZipFile(zip_path)
    for n in zf.namelist():
        rois.update(read_roi_file(zf.open(n)))
    return rois

def GenerateROIFromSPIN(tob_file_path, ref_image):
    '''
    This function was to read .tob file, which was designed by SPIN, MRInnovation, (by Dr. Haacke, Ying Wang.)
    :param tob_file_path: The file with '.tob'
    :param ref_image: The reference image, which provided the shape of the image
    :return: The generated ROI. Different cores was assigned to different values. The first 2 dimensions were swapped for visualization.
     By Jie Wu, Nov-11-18
    '''
    if isinstance(ref_image, str):
        ref_image = sitk.ReadImage(ref_image)

    image_shape = ref_image.GetSize()[0:3]
    recon_data = np.zeros(image_shape)

    index_list = np.fromfile(tob_file_path, dtype=np.uint32)

    total_roi = index_list[1]
    roi_number_index = 2
    for roi_index in range(total_roi):
        roi_point_number = index_list[roi_number_index] // 4
        print('The number of array: ', roi_point_number)
        for roi_point_index in range(1, roi_point_number):
            recon_data[index_list[roi_number_index + roi_point_index * 4 + 1],
                       index_list[roi_number_index + roi_point_index * 4 + 2],
                       index_list[roi_number_index + roi_point_index * 4 + 3]] = roi_index + 1
        roi_number_index += roi_point_number * 4 + 1

    recon_data = np.swapaxes(recon_data, 0, 1)
    return recon_data

def fid_read(file_path):
    with open(file_path, 'rb') as f:
        #1 double
        f.seek(0,0)
        bytes = f.read(8)
        file_version, = struct.unpack('d', bytes)
        #print(file_version)
        
        #4 int
        bytes = f.read(4*4)
        section_size = struct.unpack('4i', bytes)
        #print(section_size)    
        f.seek(8 + 4*4 + section_size[0] + section_size[1] + section_size[2], 0)
        
        #5 int
        bytes = f.read(5*4)
        dim = struct.unpack('5i', bytes)
        #print(dim)
        length = dim[4]*dim[3]*dim[2]*dim[1]*dim[0]
                
        fid = np.fromfile(f, dtype=np.complex64, count=length)
        shape = (dim[4], dim[3], dim[2], dim[1], dim[0])
        fid = fid.reshape(shape)
        
        return fid, file_version
    
 
def fid_data_cine(data, fft=True, transpose=False):
    plt.ion()
    dim = data.shape
    data = data.reshape((-1, dim[-2],  dim[-1]))
    for step in range(data.shape[0]):  
        plt.cla()
        image = data[step, :, :]
        if fft:
            image = np.fft.fftshift(np.fft.ifftn(image))
            
        if transpose:
            plt.imshow(np.transpose(abs(image)), cmap = 'gray')
        else:
            plt.imshow(abs(image), cmap = 'gray')
        
        plt.pause(0.1)  
    plt.ioff()  
    plt.show()
