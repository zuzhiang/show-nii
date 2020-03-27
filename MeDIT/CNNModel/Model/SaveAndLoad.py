import os
import pickle

def SaveModel(model, store_folder):
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file=os.path.join(store_folder, 'model.png'), show_shapes=True)
    model.summary()
    model_yaml = model.to_yaml()
    with open(os.path.join(store_folder, 'model.yaml'), "w") as yaml_file:
        yaml_file.write(model_yaml)

def LoadModel(store_folder, weight_name='last_weights.h5', is_show_summary=False):
    from keras.models import model_from_yaml
    yaml_file = open(os.path.join(store_folder, 'model.yaml'), 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights(os.path.join(store_folder, weight_name))

    if is_show_summary:
        model.summary()
    return model

def SaveHistory(history, store_folder):
    f = open(os.path.join(model_path, 'history_dict.txt'), 'wb')
    pickle.dump(temp, f)
    f.close()

def LoadHistory(store_folder, is_show=True):
    f = open(os.path.join(store_folder, 'history_dict.txt'), 'rb')
    history = pickle.load(f)
    f.close()

    if is_show:
        import matplotlib.pyplot as plt
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.legend(['train', 'val'])
        plt.show()

    return history

def SaveResult(store_folder, name_list, data_list, tag_list):
    from MeDIT.SaveAndLoad import SaveH5
    from MeDIT.Visualization import LoadWaitBar

    if len(tag_list) != len(data_list):
        print('The tag is not same as the number of inputs')
        return

    if len(name_list) != data_list[0].shape[0]:
        print('The name is not same as the number of samples')
        return

    if not os.path.exists(store_folder):
        os.mkdir(store_folder)

    print('Saving the result in {}\n'.format(store_folder))
    for sample_index in range(len(name_list)):
        LoadWaitBar(len(name_list), sample_index)

        name = name_list[sample_index]
        if not name.endswith('.h5'):
            print('The store name should be end with \'h5\' ')
            return

        store_path = os.path.join(store_folder, name)

        data = [temp[sample_index, ...] for temp in data_list]
        SaveH5(store_path, data, tag_list)

def LoadResult(store_folder, tag_list=[]):
    from MeDIT.SaveAndLoad import LoadH5
    from MeDIT.Visualization import LoadWaitBar

    if len(tag_list) == 0:
        from MeDIT.CNNModel.training.Generate import GetKeysFromStoreFolder
        tag_list = GetKeysFromStoreFolder(store_folder)

    all_data = [[] for temp in range(len(tag_list))]
    all_data = dict(zip(tag_list, all_data))
    print('Loading the result in {}'.format(store_folder))

    case_list = sorted(os.listdir(store_folder))
    for sample_index in range(len(case_list)):
        LoadWaitBar(len(case_list), sample_index)

        name = case_list[sample_index]
        if not name.endswith('.h5'):
            print('The store name should be end with \'h5\' ')
            return

        store_path = os.path.join(store_folder, name)
        for tag in tag_list:
            data = LoadH5(store_path, tag)
            all_data[tag].append(data)

    return all_data