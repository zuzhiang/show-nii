import numpy as np
from scipy.stats import sem, ttest_ind
from sklearn.metrics import roc_auc_score
from scipy import ndimage
from MeDIT.ArrayProcess import RemoveSmallRegion, XY2Index, XYZ2Index

import os
import glob
from MeDIT.SaveAndLoad import LoadNiiData

def ResampleBoost(array, times=1000):
    n_bootstraps = times
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.random_integers(0, array.size - 1, array.size)
        bootstrapped_scores.append(np.mean(array[indices]))

    return np.sort(np.array(bootstrapped_scores))

def AUC_Confidence_Interval(y_true, y_pred, CI_index=0.95):
    AUC = roc_auc_score(y_true, y_pred)

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int((1.0 - CI_index)/2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1.0 - (1.0 - CI_index)/2 * len(sorted_scores))]
    CI = [confidence_lower, confidence_upper]

    print('AUC is {:.3f}, Confidence interval : [{:0.3f} - {:0.3}]'.format(AUC, confidence_lower, confidence_upper))
    return AUC, CI, sorted_scores

def TTest(pred, label, is_show=True):
    if np.unique(label).size != 2:
        print('Only works on the binary classification.')
        return

    sample1 = pred[label == np.unique(label)[0]]
    sample1_label = label[label == np.unique(label)[0]]
    sample2 = pred[label == np.unique(label)[1]]
    sample2_label = label[label == np.unique(label)[1]]

    sample1_score = ResampleBoost(sample1)
    sample2_score = ResampleBoost(sample2)

    t, p = ttest_ind(sample1_score, sample2_score)
    if is_show:
        print("The t-statitics: ", t)
        print("The p-value: ", p)

    return t, p

def StatisticDetection(prediction_map, label_map, threshold_value_on_overlap=0.5):
    '''
    To Calculate the staitstic value of the detection results.
    For each region of the label map, if the prediction overlaps above threshold value, true positive value + 1.
    For each region of the label map, if the prediction overlaps under threshold value, false negative value + 1.
    For each region of the prediction map, if the label overlaps under threshold value, false positive value + 1.

    :param prediction_map: the detection result (binary image)
    :param label_map: the ground truth (binary image)
    :param threshold_value_on_overlap: the threshold value to estiamte the reuslt.
    :return: the true positive value, false positive value, and the false negative value.
    '''
    image_shape = prediction_map.shape

    prediction_im, prediction_nb = ndimage.label(prediction_map)
    label_im, label_nb = ndimage.label(label_map)

    true_positive = 0
    false_negative = 0
    false_positive = 0

    for index in range(1, label_nb+1):
        x_label, y_label = np.where(label_im == index)
        index_label = XY2Index([x_label, y_label], image_shape)

        x_pred, y_pred = np.where(prediction_im > 0)
        index_pred = XY2Index([x_pred, y_pred], image_shape)
        inter_index = np.intersect1d(index_label, index_pred)

        if inter_index.size / index_label.size >= threshold_value_on_overlap:
            true_positive += 1
        else:
            false_negative += 1

    for index in range(1, prediction_nb+1):
        x_pred, y_pred = np.where(prediction_im == index)
        index_pred = XY2Index([x_pred, y_pred], image_shape)

        x_label, y_label = np.where(label_im > 0)
        index_label = XY2Index([x_label, y_label], image_shape)
        inter_index = np.intersect1d(index_label, index_pred)

        if inter_index.size / index_pred.size < threshold_value_on_overlap:
            false_positive += 1

    return true_positive, false_positive, false_negative

def StatsticOverlap(prediction_map, label_map):
    image_shape = prediction_map.shape

    if prediction_map.ndim == 2:
        x_label, y_label = np.where(label_map > 0)
        index_label = XY2Index([x_label, y_label], image_shape)

        x_pred, y_pred = np.where(prediction_map > 0)
        index_pred = XY2Index([x_pred, y_pred], image_shape)
    elif prediction_map.ndim == 3:
        x_label, y_label, z_label = np.where(label_map > 0)
        index_label = XYZ2Index([x_label, y_label, z_label], image_shape)

        x_pred, y_pred, z_pred = np.where(prediction_map > 0)
        index_pred = XYZ2Index([x_pred, y_pred, z_pred], image_shape)

    inter_index = np.intersect1d(index_label, index_pred)

    true_positive_value = len(inter_index)
    false_positive_value = len(index_pred) - len(inter_index)
    false_negative_value = len(index_label) - len(inter_index)

    return true_positive_value, false_positive_value, false_negative_value

def TopNAccuracy(array, label, n):
    assert(array.shape[1] > n)
    correct = []
    for one_label, one_estimation in zip(label, np.argsort(array, axis=1)):
        if one_label in one_estimation[-n:]:
            correct.append(1)
        else:
            correct.append(0)

    correct = np.array(correct)
    return np.mean(correct)

def NiiImageInfoStatistic(root_folder, key_word):
    case_list = os.listdir(root_folder)
    case_list.sort()

    shape_list = []
    spacing_list = []

    for case in case_list:
        case_folder = os.path.join(root_folder, case)
        candidate_file = glob.glob(os.path.join(case_folder, key_word))
        if len(candidate_file) != 1:
            print('Not unique file: ', case)
            continue

        image, _, data = LoadNiiData(candidate_file[0])

        shape_list.append(image.GetSize())
        spacing_list.append(image.GetSpacing())

    shape_info = np.array(shape_list)
    spacing_info = np.array(spacing_list)

    print('The number of cases is :', len(shape_list))
    print('The mean of the size is :', np.mean(shape_info, axis=0))
    print('The max of the size is :', np.max(shape_info, axis=0))
    print('The min of the size is :', np.min(shape_info, axis=0))
    print('The mean of the spacing is :', np.mean(spacing_info, axis=0))
    print('The max of the spacing is :', np.max(spacing_info, axis=0))
    print('The min of the spacing is :', np.min(spacing_info, axis=0))

    import matplotlib.pyplot as plt
    fig = plt.figure(0, [15, 5])
    ax1 = fig.add_subplot(131)
    ax1.hist(spacing_info[:, 0], bins=50)
    ax1.set_xlabel('Resolution')
    ax1.set_ylabel('Spacing/mm')
    ax1.set_title('Histogram of x-axis spacing')
    ax2 = fig.add_subplot(132)
    ax2.hist(spacing_info[:, 1], bins=50)
    ax2.set_xlabel('Resolution')
    ax2.set_ylabel('Spacing/mm')
    ax2.set_title('Histogram of y-axis spacing')
    ax3 = fig.add_subplot(133)
    ax3.hist(spacing_info[:, 2], bins=50)
    ax3.set_xlabel('Resolution')
    ax3.set_ylabel('Spacing/mm')
    ax3.set_title('Histogram of z-axis spacing')
    plt.show()

def ROIImageInfoStatistic(root_folder, key_word):
    case_list = os.listdir(root_folder)
    case_list.sort()

    shape_list = []
    spacing_list = []
    range_list = []

    for case in case_list:
        case_folder = os.path.join(root_folder, case)
        candidate_file = glob.glob(os.path.join(case_folder, key_word))
        if len(candidate_file) != 1:
            print('Not unique file: ', case)
            continue

        image, _, data = LoadNiiData(candidate_file[0])

        x, y, z = np.where(data == 1)
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        z_range = np.max(z) - np.min(z)

        shape_list.append(image.GetSize())
        spacing_list.append(image.GetSpacing())
        range_list.append([x_range, y_range, z_range])

    shape_info = np.array(shape_list)
    spacing_info = np.array(spacing_list)
    range_info = np.array(range_list)

    print('The number of cases is :', len(shape_list))
    print('The mean of the size is :', np.mean(shape_info, axis=0))
    print('The max of the size is :', np.max(shape_info, axis=0))
    print('The min of the size is :', np.min(shape_info, axis=0))
    print('The mean of the spacing is :', np.mean(spacing_info, axis=0))
    print('The max of the spacing is :', np.max(spacing_info, axis=0))
    print('The min of the spacing is :', np.min(spacing_info, axis=0))
    print('The mean of the range is :', np.mean(range_info, axis=0))
    print('The max of the range is :', np.max(range_info, axis=0))
    print('The min of the range is :', np.min(range_info, axis=0))


    import matplotlib.pyplot as plt
    fig = plt.figure(0, [15, 5])
    ax1 = fig.add_subplot(131)
    ax1.hist(spacing_info[:, 0], bins=50)
    ax1.set_xlabel('Resolution')
    ax1.set_ylabel('Spacing/mm')
    ax1.set_title('Histogram of x-axis spacing')
    ax2 = fig.add_subplot(132)
    ax2.hist(spacing_info[:, 1], bins=50)
    ax2.set_xlabel('Resolution')
    ax2.set_ylabel('Spacing/mm')
    ax2.set_title('Histogram of y-axis spacing')
    ax3 = fig.add_subplot(133)
    ax3.hist(spacing_info[:, 2], bins=50)
    ax3.set_xlabel('Resolution')
    ax3.set_ylabel('Spacing/mm')
    ax3.set_title('Histogram of z-axis spacing')
    plt.show()

    import matplotlib.pyplot as plt
    fig = plt.figure(0, [15, 5])
    ax1 = fig.add_subplot(131)
    ax1.hist(range_info[:, 0], bins=50)
    ax1.set_xlabel('Resolution')
    ax1.set_ylabel('Spacing/mm')
    ax1.set_title('Histogram of x-axis spacing')
    ax2 = fig.add_subplot(132)
    ax2.hist(range_info[:, 1], bins=50)
    ax2.set_xlabel('Resolution')
    ax2.set_ylabel('Spacing/mm')
    ax2.set_title('Histogram of y-axis spacing')
    ax3 = fig.add_subplot(133)
    ax3.hist(range_info[:, 2], bins=50)
    ax3.set_xlabel('Resolution')
    ax3.set_ylabel('Spacing/mm')
    ax3.set_title('Histogram of z-axis spacing')
    plt.show()

if __name__ == '__main__':
    label = np.squeeze(np.asarray(np.load(r'C:\Users\SY\Desktop\label.npy'), dtype=np.uint8))
    pred = np.squeeze(np.load(r'C:\Users\SY\Desktop\cnn_result.npy'))

    print(TTest(pred, label))