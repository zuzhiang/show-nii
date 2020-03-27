import numpy as np
from sklearn.linear_model import LinearRegression
from MeDIT.Visualization import Imshow3DArray
from skimage.measure import compare_ssim

from MeDIT.Normalize import IntensityTransfer

def MergeKspace(recon_kdata, sampled_kdata, mask, is_fit=True, alpha = 0.5):
    if recon_kdata.shape != mask.shape or sampled_kdata.shape != mask.shape:
        print('Check the shape of these datas')
        return []

    if is_fit:
        fit_x = recon_kdata[mask == 1].flatten()
        fit_y = sampled_kdata[mask == 1].flatten()

        fit_x = fit_x[..., np.newaxis]
        fit_y = fit_y[..., np.newaxis]

        linear_regression = LinearRegression()
        linear_regression.fit(fit_x, fit_y)
        k, b = linear_regression.coef_, linear_regression.intercept_

        recon_kdata = recon_kdata * k + b

    recon_kdata[mask == 1] = alpha * sampled_kdata[mask == 1] + (1 - alpha) * recon_kdata[mask == 1]
    return recon_kdata

def GetMSE(image1, image2):
    return np.mean(np.square(image1 - image2))

def GetSSIM(image1, image2):
    temp_image1 = np.asarray(IntensityTransfer(image1, 255, 0), dtype=np.uint8)
    temp_image2 = np.asarray(IntensityTransfer(image2, 255, 0), dtype=np.uint8)

    return compare_ssim(temp_image1, temp_image2, data_range=255)