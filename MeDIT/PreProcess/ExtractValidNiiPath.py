import pydicom
from copy import deepcopy
import os
import numpy as np

from MeDIT.Others import SplitPathWithSuffex

class ExtractValidPath:
    def __init__(self, case_folder=''):
        self.__case_folder = case_folder

    def SetCaseFolder(self, case_folder):
        if not os.path.isdir(case_folder):
            print('Should give a folder path')
        else:
            self.__case_folder = case_folder
    def GetCaseFolder(self):
        return self.__case_folder
    case_folder = property(GetCaseFolder, SetCaseFolder)

    def GetPath(self, include_key, exclude_key=(), suffex='.nii'):
        if isinstance(include_key, str):
            include_key = [include_key]
        if isinstance(exclude_key, str):
            exclude_key = [exclude_key]
        if isinstance(suffex, str):
            suffex = [suffex]
        if isinstance(suffex, list):
            suffex = tuple(suffex)

        t2_candidate = []
        for root, dirs, files in os.walk(self.__case_folder):
            if len(files) == 0:
                continue
            for file in files:
                if file.endswith(suffex):
                    if all(key in file for key in include_key):
                        t2_candidate.append(os.path.join(root, file))

        return [temp for temp in t2_candidate if not any(key in temp for key in exclude_key)]

    def ExtendPath(self, existing_path, include_key, exclude_key=(), suffex='.nii'):
        temp_path = self.GetPath(include_key, exclude_key=exclude_key, suffex=suffex)
        existing_path.extend(temp_path)
        return deepcopy(existing_path)

    def GetManufacturer(self):
        for root, dir, files in os.walk(self.__case_folder):
            if len(dir) == 0 and len(files) > 0:
                try:
                    header = pydicom.dcmread(os.path.join(root, files[0]))
                    return header.Manufacturer
                except:
                    pass

        return 'No Manufacturer'

    def GetPatientID(self):
        for root, dir, files in os.walk(self.__case_folder):
            if len(dir) == 0 and len(files) > 0:
                try:
                    header = pydicom.dcmread(os.path.join(root, files[0]))
                    return header.PatientID
                except:
                    pass

        return 'No ID'

class SimensNii(ExtractValidPath):
    def __init__(self, case_folder=''):
        super(SimensNii, self).__init__(case_folder)

    def GetCorrespondingFiles(self, candidate_path):
        file, file_suffex = SplitPathWithSuffex(candidate_path)
        file_dir, file_name = os.path.split(file)
        return self.GetPath(include_key=file_name, suffex='')

    def GetShortestFile(self, candidate_list):
        candidate_list.sort(key=len)
        return candidate_list[0]

    def GetT2AxisPath(self, include_key='t2_tse_tra', exclude_key=('roi', 'Resize'), suffex=['nii', 'nii.gz']):
        return self.GetPath(include_key=include_key, exclude_key=exclude_key, suffex=suffex)

    def GetDwiPath(self, include_key=['diff'], exclude_key=('Reg', 'ADC', 'Resize'), suffex=('nii', 'nii.gz')):
        dwi_candidate = self.GetPath(include_key=include_key, exclude_key=exclude_key, suffex=suffex)
        dwi_candidate = self.ExtendPath(dwi_candidate, include_key=['dwi'], exclude_key=exclude_key, suffex=suffex)
        dwi_candidate = self.ExtendPath(dwi_candidate, include_key=['DWI'], exclude_key=exclude_key, suffex=suffex)
        dwi_candidate = self.ExtendPath(dwi_candidate, include_key=['b_DKI'], exclude_key=exclude_key, suffex=suffex)

        dwi_path = self.GetShortestFile(dwi_candidate)
        dwi_candidate = self.GetCorrespondingFiles(dwi_path)
        return dwi_candidate

    def GetAdcPath(self, include_key=('ADC'), exclude_key=('Reg', 'Resize'), suffex=('nii', 'nii.gz')):
        adc_candidate = self.GetPath(include_key=include_key, exclude_key=exclude_key, suffex=suffex)
        adc_path = self.GetShortestFile(adc_candidate)
        return adc_path

class UIHNii(ExtractValidPath):
    def __init__(self, case_folder=''):
        super(UIHNii, self).__init__(case_folder)

    def __ExtractBvalueFromDWIListUIH(self, candidate_list):
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

    def GetT2AxisPath(self, include_key=('t2', 'tra'), exclude_key=('roi', 'Resize'), suffex=['nii', 'nii.gz']):
        return self.GetPath(include_key=include_key, exclude_key=exclude_key, suffex=suffex)

    def GetDwiPath(self, target_b_value=-1, tol=200, is_find_lowest_b=True,
                   include_key=('dwi', 'b'), exclude_key=('Reg', 'ADC', 'Resize'), suffex=('nii', 'nii.gz')):
        dwi_candidate = self.GetPath(include_key=include_key, exclude_key=exclude_key, suffex=suffex)
        b_value = self.__ExtractBvalueFromDWIListUIH(dwi_candidate)
        if target_b_value < 0:
            return dwi_candidate, b_value
        else:
            diff = abs(np.array(b_value) - target_b_value)
            if np.min(diff) > tol:
                print('No DWI file with adaptive b vlaue')
            index = np.argmin(diff)
            dwi_path_target = dwi_candidate[index]
            b_value_target = b_value[index]

            if is_find_lowest_b:
                index = np.argmin(np.array(b_value))
                dwi_path_lowest = dwi_candidate[index]
                b_value_lowest = b_value[index]

                return [dwi_path_target, dwi_path_lowest], [b_value_target, b_value_lowest]
            else:
                return dwi_path_target, b_value_target

    def GetAdcPath(self, include_key=('ADC'), exclude_key=('Reg', 'Resize'), suffex=('nii', 'nii.gz')):
        return self.GetPath(include_key=include_key, exclude_key=exclude_key, suffex=suffex)


if __name__ == '__main__':
    # evp = SimensNii(r'd:\USB Copy_2018-12-06_103444\BAO ZHENG LI')
    evp = UIHNii(r'd:\USB Copy_2018-12-07_140325\Chen Bing Lou')
    print(evp.case_folder)
    print(evp.GetManufacturer())
    print(evp.GetT2AxisPath())
    print(evp.GetDwiPath(target_b_value=800, is_find_lowest_b=True))
    print(evp.GetAdcPath())