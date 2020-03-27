import SimpleITK as sitk
import numpy as np
import shutil
import math
import sys
import os

def IsNumber(string):
    '''
    To adjust the string belongs to a number or not.
    :param string:
    :return:
    '''
    try:
        float(string)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(string)
        return True
    except (TypeError, ValueError):
        pass

    return False

def IsValidNumber(string):
    if not IsNumber(string):
        return False

    if math.isnan(float(string)):
        return False

    return True

def GetPhysicaladdress():
    '''
    @summary: return the MAC address of the computer
    '''

    mac = None
    if sys.platform == "win32":
        for line in os.popen("ipconfig /all"):
            # print line
            if line.lstrip().startswith("Physical Address") or line.lstrip().startswith("物理地址"):
                mac = line.split(":")[1].strip().replace("-", ":")
                break

    else:
        for line in os.popen("/sbin/ifconfig"):
            if 'Ether' in line:
                mac = line.split()[4]
                break
    return mac

def RemoveKeyPathFromPathList(path_list, key_word):
    new_path_list = []
    for p in path_list:
        if key_word not in p:
            new_path_list.append(p)

    return new_path_list

def CopyFile(source_path, dest_path, is_replace=True):
    if not os.path.exists(source_path):
        print('File does not exist: ', source_path)
        return None
    if (not os.path.exists(dest_path)) or is_replace:
        shutil.copyfile(source_path, dest_path)

def HumanSortFile(file_list):
    import re

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    file_list.sort(key=alphanum_key)
    return file_list

def SplitPathWithSuffex(file_path):
    if file_path.endswith('.nii.gz'):
        return file_path[:-len('.nii.gz')], '.nii.gz'
    else:
        return os.path.splitext(file_path)

def CompareSimilarityOfLists(*input_lists):
    if len(input_lists) < 2:
        print('At least 2 lists')

    max_diff = -1.
    for one in input_lists:
        for second in input_lists[input_lists.index(one) + 1:]:
            one_array = np.asarray(one)
            second_array = np.asarray(second)
            diff = np.sqrt(np.sum(np.square(one_array - second_array)))
            if diff > max_diff:
                max_diff = diff

    return max_diff

if __name__ == '__main__':
    # array = np.array([1, 'z', 2.5, 1e-4, np.nan, '3'])
    # for index in np.arange(array.size):
    #     print(IsValidNumber(array[index]))
    print(GetPhysicaladdress())
    
