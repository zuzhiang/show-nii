import os
import shutil
from random import shuffle

def SeparateCases(root_folder, store_folder, testing_percentage=0.3):
    case_list = os.listdir(root_folder)
    shuffle(case_list)

    spe_index = int(len(case_list) * testing_percentage)

    training_store_folder = os.path.join(store_folder, 'training')
    testing_store_folder = os.path.join(store_folder, 'testing')
    if not os.path.exists(training_store_folder):
        os.mkdir(training_store_folder)
    if not os.path.exists(testing_store_folder):
        os.mkdir(testing_store_folder)

    for case in case_list[:spe_index]:
        try:
            shutil.copytree(os.path.join(root_folder, case), os.path.join(testing_store_folder, case))
        except:
            shutil.copyfile(os.path.join(root_folder, case), os.path.join(testing_store_folder, case))

    for case in case_list[spe_index:]:
        try:
            shutil.copytree(os.path.join(root_folder, case), os.path.join(training_store_folder, case))
        except:
            shutil.copyfile(os.path.join(root_folder, case), os.path.join(training_store_folder, case))

