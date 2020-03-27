import os
import glob

from MeDIT.Others import SplitPathWithSuffex

def RemoveDuplicateROI(case_folder):
    for root, dirs, files in os.walk(case_folder):
        if len(dirs) > 0 and len(files) > 0:
            one_roi_candidate = glob.glob(os.path.join(root, '*roi.ni*'))
            more_roi_candidate = glob.glob(os.path.join(root, '*roi?.ni*'))
            if len(one_roi_candidate) > 0 and len(more_roi_candidate) > 0:
                print(case_folder, case_folder, case_folder, )

def FindROICorrespondingFile(roi_path, given_suffex=''):
    file_name, suffex = SplitPathWithSuffex(roi_path)
    if not given_suffex:
        given_suffex = suffex
    try:
        index = file_name.index('_roi')
        file_path = file_name[:index] + given_suffex
        if os.path.exists(file_path):
            return file_path
        else:
            print('No file')
            return ''
    except Exception as e:
        print(e)
        return ''


if __name__ == '__main__':
    # a = FindROICorrespondingFile(r'd:\USB Copy_2018-12-07_140325\Chen Bing Lou\MR\20180912\170329\7830\004_t2_fse_tra_roi2.csv')
    # print(a)

    case_folder = r'w:\JSPH\ProstateCancer\2017Year\CheckAndWork\BSZ^bi shu zhong'
    key_word = ['t2']
    suffex = ['.nii']
    exclude = ['roi']

    from MeDIT.PreProcess.ExtractValidNiiPath import ExtractValidPath
    evp = ExtractValidPath()
    evp.case_folder = case_folder
    print(evp.GetManufacturer())
    print(evp.GetPatientID())
    temp = evp.GetPath(key_word, exclude_key=exclude, suffex=suffex)
    print(temp)

    # import os
    #
    # candiate_file = []
    # for root, folder, files in os.walk(case_folder):
    #     if len(files) > 0:
    #         candiate_file.extend([os.path.join(root, temp) for temp in files if temp.endswith('.nii')])
    #
    #
    # print(candiate_file)


