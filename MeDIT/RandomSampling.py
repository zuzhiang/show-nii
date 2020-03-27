import numpy as np
import imageio
from copy import deepcopy

# import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
import pics.operators_class as opts
import utilities.utilities_func as ut

# from espirit.espirit_func import espirit_2d, espirit_3d
import scipy.io as sio


def Generate1DGaussianSamplingStrategy(phase_encoding_number, center_sampling_rate):
    temp_image = np.zeros((phase_encoding_number, phase_encoding_number))
    temp_image = np.asarray(temp_image, dtype=np.uint8)

    sample_order = []

    sample = np.zeros((phase_encoding_number,))
    center_point = phase_encoding_number // 2
    center_width = round(phase_encoding_number * center_sampling_rate / 2)
    center_index = np.arange(center_point - center_width, center_point + center_width)
    sample[center_index] = 1

    for index in center_index:
        sample_order.append(index)

    random_sample_index = np.where(sample == 0)[0].tolist()

    # Generate Gaussian Distribution
    random_sample_number = phase_encoding_number - len(center_index)
    sigma = random_sample_number / 6
    x = np.arange(-random_sample_number // 2, random_sample_number // 2)
    pdf_array = 1 / np.sqrt(2 * np.pi * sigma * sigma) * np.exp(-1 * x * x / (2 * sigma * sigma))
    sample_value = [np.sum(pdf_array[:index]) for index in range(pdf_array.shape[0])]

    pdf = deepcopy(pdf_array)
    pdf_list = pdf_array.tolist()

    for index in range(random_sample_number):
        random_prob = np.random.rand((1))[0]

        # print(random_prob)
        # print(sample_value)
        # find the index
        for removed_index in range(random_sample_number - 1, -1, -1):
            if sample_value[removed_index] < random_prob:
                if removed_index == random_sample_number - 1:
                    prob = 1 - sample_value[removed_index]
                else:
                    prob = sample_value[removed_index + 1] - sample_value[removed_index]

                # print(random_sample_number)
                # print(random_sample_index)
                sample_order.append(random_sample_index[removed_index])

                # print(len(pdf_list))
                random_sample_index.pop(removed_index)
                pdf_list.pop(removed_index)
                pdf_array = np.asarray(pdf_list, dtype=np.float32)
                pdf_array = pdf_array / (1 - prob)
                pdf_list = pdf_array.tolist()
                sample_value = [np.sum(pdf_array[:index]) for index in range(pdf_array.shape[0])]

                random_sample_number -= 1
                break

        sample_index = np.asarray(sample_order, dtype=np.uint16)
        temp_image[sample_index, :] = 255

    return pdf, sample_order


def Save2DSamplingStategyAsGIF(sampling_order, store_path, image_shape=[], sample_axis=0):
    if image_shape == []:
        image_shape = [len(sampling_order), len(sampling_order)]

    assert(len(sampling_order) == image_shape[sample_axis])

    gif = []
    image = np.zeros(image_shape)
    image = np.asarray(image, dtype=np.uint8)

    for index in sampling_order:
        if sample_axis == 0:
            image[index, :] = 255
        elif sample_axis == 1:
            image[:, index] = 255

        gif.append(deepcopy(image))

    imageio.mimsave(store_path, gif)

def GenSamplingMask(image_shape, sampling_percentage, center_sampling_rate, sample_axis=0):
    if isinstance(image_shape, int):
        image_shape = [image_shape, image_shape]

    mask = np.zeros(image_shape)
    if sample_axis == 0:
        phase_encoding_number = image_shape[0]
    elif sample_axis == 1:
        phase_encoding_number = image_shape[1]
    else:
        print('Give correct sample_axis')
        return mask

    _, order = Generate1DGaussianSamplingStrategy(phase_encoding_number, center_sampling_rate)
    sample_order = np.asarray(order[:round(len(order) * sampling_percentage)], dtype=np.uint16)

    if sample_axis == 0:
        mask[sample_order, :] = 1
    elif sample_axis == 1:
        mask[:, sample_order] = 1

    return mask


def ReconstructADMM_2D(fullysampled_kdata, mask, iterations=10, step=0.05, tv_r=0.005, rho=1.0, is_show=True):
    fullysampled_kdata = fullysampled_kdata[..., np.newaxis]
    FTm = opts.FFTW2d_kmask(mask)

    esp = opts.espirit(sensitivity=np.ones_like(fullysampled_kdata))
    Aopt = opts.joint2operators(esp, FTm)

    im = FTm.backward(fullysampled_kdata)
    dwt = opts.DWT2d(wavelet='haar', level=4)

    # undersampling in k-space
    b = FTm.forward(im)
    scaling = ut.optscaling(FTm, b)
    b = b / scaling

    # do cs mri recon
    Nite = iterations  # number of iterations
    step = step  # step size
    tv_r = tv_r  # regularization term for tv term
    rho = rho
    # th = 1  # threshold

    # xopt = solvers.IST_2(FTm.forward,FTm.backward,b, Nite, step,th) #soft thresholding
    xopt = solvers.ADMM_l2Afxnb_tvx(Aopt.forward, Aopt.backward, b, Nite, step, tv_r, rho, is_show=is_show)
    # xopt = solvers.ADMM_l2Afxnb_l1x_2( FTm.forward, FTm.backward, b, Nite, step, 100, 1 )

    # ut.plotim3(np.absolute(xopt))
    return xopt

def test_recon():
    from MeDIT.SaveAndLoad import LoadH5
    image = np.load(r'z:\Data\MIDAS-recon\resultfolder\fullysampled_image.npy')
    mask = LoadH5(r'z:\Data\MIDAS-recon\CNN\mask.h5', 'mask', data_type=np.uint8)

    import matplotlib.pyplot as plt
    plt.imshow(np.real(np.squeeze(image)), cmap='gray')
    plt.show()

    kdata = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
    recon = ReconstructADMM_2D(kdata, mask)

    plt.imshow(np.real(np.squeeze(recon)), cmap='gray')
    plt.show()

def GetUnderSampledImage(image, mask):
    kdata = np.fft.fftshift(np.fft.fft2(image))
    under_sampled_kdata = kdata * mask
    under_dampled_image = np.fft.ifft2(np.fft.ifftshift(under_sampled_kdata))
    return under_dampled_image


