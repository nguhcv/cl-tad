import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


class Dataset_Loader():
    def __init__(self, dataset, data_path,window_size, ts_num):
#        if dataset not in ["'ecg'", 'ucr', 'pd', 'gesture', 'credit', 'nab', 'kpi', 'psm', 'smd']:
#           raise ValueError ('this dataset is not supported')
#        else:
        self.dataset = dataset

        self.data_path = data_path

        if self.dataset =='ecg':
            self.ecg_dataset_name = ['chfdb_chf01_275.pkl', 'chfdb_chf13_45590.pkl','chfdbchf15.pkl', 'ltstdb_20221_43.pkl', 'ltstdb_20321_240.pkl', 'mitdb__100_180.pkl', 'qtdbsel102.pkl', 'stdb_308_0.pkl', 'xmitdb_x108_0.pkl']
            self.ts_num = ts_num

        if self.dataset =='ucr':
            self.ucr_dataset_name=['135_', '136_', '137_', '138_']
            self.ts_num = ts_num

        if self.dataset =='smd':
            self.smd_dataset_name=['machine-1-1','machine-1-2','machine-1-3','machine-1-4',
                             'machine-1-5', 'machine-1-6', 'machine-1-7', 'machine-1-8',
                             'machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4',
                             'machine-2-5', 'machine-2-6', 'machine-2-7', 'machine-2-8',
                             'machine-2-9', 'machine-3-1', 'machine-3-2','machine-3-3',
                             'machine-3-4', 'machine-3-5', 'machine-3-6', 'machine-3-7',
                             'machine-3-8', 'machine-3-9', 'machine-3-10','machine-3-11',]
            self.ts_num = ts_num

        if self.dataset =='kpi':
            self.ts_num = ts_num

        self.__get_dataset()
        self.__sliding_window_generation(window_size=window_size)

    def __get_dataset(self):
        if self.dataset=='ecg':
            self.dataset = get_ECG_dataset(data_path=self.data_path,
                                      dataset_name=self.ecg_dataset_name,
                                      ts_num=self.ts_num, normalized=True)

        elif self.dataset =='pd':
            self.dataset = get_Power_Demand_dataset(data_path=self.data_path)

        elif self.dataset =='gesture':
            self.dataset = get_Gesture_dataset(data_path=self.data_path)

        elif self.dataset =='nab':
            self.dataset = get_NAB_dataset(data_path=self.data_path)

        elif self.dataset =='ucr':
            self.dataset = get_UCR_dataset(data_path=self.data_path, dataset_name=self.ucr_dataset_name,ts_num=self.ts_num, normalized=True)

        elif self.dataset =='kpi':
            self.dataset = get_KPI_dataset(data_path=self.data_path, ts_num=self.ts_num, normalized=True)

        elif self.dataset =='psm':
            self.dataset = get_PSM_dataset(data_path=self.data_path, normalized=True)

        elif self.dataset =='smd':
            self.dataset = get_SMD_dataset(data_path=self.data_path, ts_num=self.ts_num, normalized=True, dataset_name=self.smd_dataset_name)

    def __sliding_window_generation(self,window_size):
            # self.train_set, self.test_set, self.validation_set = sliding_window_generation(dataset=self.dataset, window_size=window_size)

        self.train_set, self.test_set = sliding_window_generation(dataset=self.dataset,
                                                                                   window_size=window_size)


    def train_loader_generation(self, batch_size, shuffle=True):

        batch_train_data = torch.utils.data.TensorDataset(torch.tensor(self.train_set['samples'].astype(np.float32)),
                                                          torch.tensor(self.train_set['labels'].astype(np.float32)))
        train_loader = torch.utils.data.DataLoader(dataset=batch_train_data, batch_size=batch_size,
                                                         shuffle=shuffle)
        return train_loader

    def val_test_loader_generation(self,batch_size, shuffle=True):

        batch_test_data = torch.utils.data.TensorDataset(torch.tensor(self.test_set['samples'].astype(np.float32)),
                                                          torch.tensor(self.test_set['labels'].astype(np.float32)))
        test_loader = torch.utils.data.DataLoader(dataset=batch_test_data, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last = False)

        # batch_val_data = torch.utils.data.TensorDataset(torch.tensor(self.validation_set['samples'].astype(np.float32)),
        #                                                 torch.tensor(self.validation_set['labels'].astype(np.float32)))
        # val_loader = torch.utils.data.DataLoader(dataset=batch_val_data, batch_size=batch_size,
        #                                          shuffle=shuffle)

        # return val_loader, test_loader

        return test_loader

    def plot(self, data_type='train', dataset_name=None):
        data =None
        labels=None
        if data_type=='train':
            data = self.dataset['train_data']
        elif data_type=='val':
            data = self.dataset['validate_data']
            labels = self.dataset['validate_label']
        elif data_type=='test':
            data = self.dataset['test_data']
            labels= self.dataset['test_label']

        # cl=['gray', 'black']

        for i in range (data.shape[0]):
            # plt.plot(data[i], color=cl[i])
            plt.plot(data[i])
            print(len(data[i]))
        if data_type=='test':
            start = 0
            end=0
            for k in range (len(labels)-1):
                if labels[k] ==0. and labels[k+1]==1:
                    start = k+1
                if labels[k] ==1. and labels[k+1]==0:
                    end = k
            plt.axvspan(start, end,0.,1., alpha=0.5, color='lightgreen')

        plt.title(str(data_type)+ ' data of dataset ' +str(dataset_name))
        plt.xlabel('time point')
        plt.ylabel('values')
        plt.show()

        if data_type in ['val', 'test']:
            plt.plot(labels)
            plt.xlabel('time point')
            plt.ylabel('anomaly labels')
            plt.title(str(data_type)+ ' label')
            plt.show()




        #plot val data



def get_SMD_dataset(data_path,ts_num,dataset_name, normalized = True):
    # output_folder = self.path+'load_data/'
    # folder = os.path.join(output_folder, 'SMD')
    # os.makedirs(folder, exist_ok=True)
    #
    # file_list = os.listdir(os.path.join(self.path, "train"))
    # for filename in file_list:
    #     if filename.endswith('.txt'):
    #         load_and_save('train', filename, filename.strip('.txt'), self.path,output_folder)
    #         s = load_and_save('test', filename, filename.strip('.txt'), self.path,output_folder)
    #         load_and_save2('labels', filename, filename.strip('.txt'), self.path, s,output_folder)

    # folder = self.path+'load_data/SMD/'
    # loader =[]
    # for file in ['train', 'test', 'labels']:
    #     file = 'machine-3-2_' + file
    #     loader.append(np.load(os.path.join(folder, f'{file}.npy')))

    dataset = {}
    file_list = os.listdir(data_path)
    for file_name in file_list:
        if file_name.startswith(dataset_name[ts_num]):
            if file_name.endswith('train.npy'):
                train_data = np.load(data_path + file_name)
                train_data = train_data.T
                print(train_data.shape)

                dataset = {**dataset, 'train_data': train_data}

            elif file_name.endswith('test.npy'):
                test_data = np.load(data_path + file_name)
                test_data = test_data.T
                print(test_data.shape)
                dataset = {**dataset, 'test_data': test_data}

            elif file_name.endswith('labels.npy'):
                label = np.load(data_path + file_name)
                print(label.shape)
                labels = []
                for i in range(label.shape[0]):
                    if 1. in label[i]:
                        labels.append(1.)
                    else:
                        labels.append(0.)
                dataset = {**dataset, 'test_label': labels}

    if normalized:
        max_val = np.nanmax(dataset['train_data'])
        min_val = np.nanmin(dataset['train_data'])

        for dim in range(dataset['train_data'].shape[0]):
            for i in range(dataset['train_data'].shape[1]):
                if dataset['train_data'][dim][i] != np.nan:
                    dataset['train_data'][dim][i] = ((dataset['train_data'][dim][i] - min_val) / (max_val - min_val))

            for j in range(dataset['test_data'].shape[1]):
                if dataset['test_data'][dim][j] != np.nan:
                    dataset['test_data'][dim][j] = ((dataset['test_data'][dim][j] - min_val) / (max_val - min_val))
    return dataset



def get_PSM_dataset(data_path, normalized = True):
    trainfile = open(data_path + '/train.csv', 'rb')
    testfile = open(data_path + '/test.csv', 'rb')
    test_labels = open(data_path + '/test_label.csv', 'rb')

    tr_data = pd.DataFrame(pd.read_csv(trainfile))
    te_data = pd.DataFrame(pd.read_csv(testfile))
    te_label = pd.DataFrame(pd.read_csv(test_labels))

    train_data = tr_data.iloc[:, 1:].to_numpy()
    train_data = train_data.T
    # print(np.isnan(train_data).any())
    #
    # for i in range (train_data.shape[0]):
    #     if np.isnan(train_data[i]).any():
    #         print(i)

    # breakpoint()
    # print(train_data.shape)

    test_data = te_data.iloc[:, 1:].to_numpy()
    test_data = test_data.T
    # print(test_data.shape)
    test_label = te_label.iloc[:, -1].to_numpy()
    test_label = np.reshape(test_label, newshape=(test_label.shape[0],))
    # print(test_label.shape)
    # plt.plot(test_label)
    # plt.show()

    if normalized:
        max_val = np.nanmax(train_data)
        min_val = np.nanmin(train_data)

        for dim in range(train_data.shape[0]):
            for i in range(train_data.shape[1]):
                if train_data[dim][i] != np.nan:
                    train_data[dim][i] = ((train_data[dim][i] - min_val) / (max_val - min_val))

            for j in range(test_data.shape[1]):
                if test_data[dim][j] != np.nan:
                    test_data[dim][j] = ((test_data[dim][j] - min_val) / (max_val - min_val))

    dataset = {'train_data': train_data, 'test_data': test_data, 'test_label': test_label,
               }
    return dataset



def get_UCR_dataset(data_path,dataset_name,ts_num, normalized = True):
    dataset = {}
    file_list = os.listdir(data_path)
    for file_name in file_list:
        if file_name.startswith(dataset_name[ts_num]):
            if file_name.endswith('train.npy'):
                train_data = np.load(data_path + file_name)
                train_data = train_data.T
                dataset = {**dataset, 'train_data': train_data}

            elif file_name.endswith('test.npy'):
                test_data = np.load(data_path + file_name)
                test_data = test_data.T
                dataset = {**dataset, 'test_data': test_data}

            elif file_name.endswith('labels.npy'):
                label = np.load(data_path + file_name)
                label = label.flatten()
                dataset = {**dataset, 'test_label': label}

    if normalized:
        max_value = np.max(dataset['train_data'][0])
        min_value = np.min(dataset['train_data'][0])
        # test_data[:] = 2 * ((test_data[:] - min_value) / (max_value - min_value)) - 1
        # train_data[:] = 2 * ((train_data[:] - min_value) / (max_value - min_value)) - 1
        dataset['test_data'][0][:] = ((dataset['test_data'][0][:] - min_value) / (max_value - min_value))
        dataset['train_data'][0][:] = ((dataset['train_data'][0][:] - min_value) / (max_value - min_value))

    return dataset


def get_KPI_dataset(data_path,ts_num, normalized = True):
    trainfile = open(data_path + 'phase2_train/phase2_train.csv', 'rb')
    tr_data = pd.DataFrame(pd.read_csv(trainfile))
    ids1 = tr_data['KPI ID'].unique()
    # print(ids1)

    tr_data = tr_data.loc[tr_data['KPI ID'] == ids1[ts_num]]
    tr_data = tr_data['value'].values
    tr_data = np.reshape(tr_data, newshape=(1, len(tr_data)))

    gt2 = data_path + 'phase2_ground_truth/phase2_ground_truth.hdf'
    te_data = pd.DataFrame(pd.read_hdf(gt2))
    ids2 = te_data['KPI ID'].unique()
    te_data = te_data.loc[te_data['KPI ID'] == ids2[ts_num]]
    te_labels = te_data['label'].values
    te_data = te_data['value'].values
    te_data = np.reshape(te_data, newshape=(1, len(te_data)))
    # breakpoint()

    if normalized:
        max_value = np.max(tr_data)
        min_value = np.min(tr_data)
        # test_data[:] = 2 * ((test_data[:] - min_value) / (max_value - min_value)) - 1
        # train_data[:] = 2 * ((train_data[:] - min_value) / (max_value - min_value)) - 1
        te_data[:] = ((te_data[:] - min_value) / (max_value - min_value))
        tr_data[:] = ((tr_data[:] - min_value) / (max_value - min_value))

    dataset = {'train_data': tr_data, 'test_data': te_data, 'test_label': te_labels}
    return dataset



def get_Gesture_dataset(data_path, normalized = True, validation_ratio=0.2):
    trainfile = open(data_path + '/labeled/train/ann_gun_CentroidA.pkl', 'rb')
    testfile = open(data_path + '/labeled/test/ann_gun_CentroidA.pkl', 'rb')

    tr_data = pd.DataFrame(pd.read_pickle(trainfile))
    train_data = tr_data[[0, 1]].to_numpy()
    train_data = train_data.T

    te_data = pd.DataFrame(pd.read_pickle(testfile))
    test_data = te_data[[0, 1]].to_numpy()
    test_data = test_data.T
    test_label = te_data[[2]].to_numpy()
    test_label = np.reshape(test_label, newshape=(test_label.shape[0],))

    if normalized:
        max_x_train = np.max(train_data[0, :])
        min_x_train = np.min(train_data[0, :])
        max_y_train = np.max(train_data[1, :])
        min_y_train = np.min(train_data[1, :])
        f_max = max(max_x_train, max_y_train)
        f_min = min(min_x_train, min_y_train)
        for i in range(train_data.shape[1]):
            train_data[0][i] = ((train_data[0][i] - f_min) / (f_max - f_min))
            train_data[1][i] = ((train_data[1][i] - f_min) / (f_max - f_min))

        for j in range(test_data.shape[1]):
            test_data[0][j] = ((test_data[0][j] - f_min) / (f_max - f_min))
            test_data[1][j] = ((test_data[1][j] - f_min) / (f_max - f_min))

    n_validate = int(test_data.shape[1] * validation_ratio)

    validate_data = test_data[:, 0: n_validate]
    # test_data = test_data[:, n_validate:]
    validate_label = test_label[0: n_validate]
    # test_label = test_label[n_validate:]

    dataset = {'train_data': train_data, 'test_data': test_data, 'test_label': test_label,
               'validate_data': validate_data, 'validate_label': validate_label}
    return dataset



def get_Power_Demand_dataset(data_path, normalized = True, validation_ratio=0.2):
    trainfile = open(data_path + '/labeled/train/power_data.pkl', 'rb')
    testfile = open(data_path + '/labeled/test/power_data.pkl', 'rb')
    tr_data = pd.DataFrame(pd.read_pickle(trainfile))
    train_data = tr_data[[0]].to_numpy()
    train_data = train_data.T

    te_data = pd.DataFrame(pd.read_pickle(testfile))
    test_data = te_data[[0]].to_numpy()
    test_data = test_data.T
    test_label = te_data[[1]].to_numpy()
    test_label = np.reshape(test_label, newshape=(test_label.shape[0],))

    if normalized:
        max_value = np.max(train_data)
        min_value = np.min(train_data)
        # test_data[:] = 2 * ((test_data[:] - min_value) / (max_value - min_value)) - 1
        # train_data[:] = 2 * ((train_data[:] - min_value) / (max_value - min_value)) - 1
        test_data[:] = ((test_data[:] - min_value) / (max_value - min_value))
        train_data[:] = ((train_data[:] - min_value) / (max_value - min_value))

    n_validate = int(test_data.shape[1] * validation_ratio)

    validate_data = test_data[:, 0: n_validate]
    # test_data = test_data[:, n_validate:]
    validate_label = test_label[0: n_validate]
    # test_label = test_label[n_validate:]

    dataset = {'train_data': train_data, 'test_data': test_data, 'test_label': test_label,
               'validate_data': validate_data, 'validate_label': validate_label}
    return dataset


def get_ECG_dataset(data_path,dataset_name,ts_num, normalized = True, validation_ratio=0.2):
    print(dataset_name[ts_num])
    trainfile = open(data_path + 'labeled/train/' + dataset_name[ts_num], 'rb')
    testfile = open(data_path + 'labeled/test/' + dataset_name[ts_num], 'rb')

    tr_data = pd.DataFrame(pd.read_pickle(trainfile))

    train_data = tr_data[[0, 1]].to_numpy()
    train_data = train_data.T

    te_data = pd.DataFrame(pd.read_pickle(testfile))
    test_data = te_data[[0, 1]].to_numpy()
    test_data = test_data.T
    test_label = te_data[[2]].to_numpy()
    test_label = np.reshape(test_label, newshape=(test_label.shape[0],))

    if normalized:
        max_x_train = np.max(train_data[0, :])
        min_x_train = np.min(train_data[0, :])
        max_y_train = np.max(train_data[1, :])
        min_y_train = np.min(train_data[1, :])
        f_max = max(max_x_train, max_y_train)
        f_min = min(min_x_train, min_y_train)
        for i in range(train_data.shape[1]):
            train_data[0][i] = ((train_data[0][i] - f_min) / (f_max - f_min))
            train_data[1][i] = ((train_data[1][i] - f_min) / (f_max - f_min))

        for j in range(test_data.shape[1]):
            test_data[0][j] = ((test_data[0][j] - f_min) / (f_max - f_min))
            test_data[1][j] = ((test_data[1][j] - f_min) / (f_max - f_min))

    # split to validation and testset

    n_validate = int(test_data.shape[1] * validation_ratio)
    validate_data = test_data[:, 0: n_validate]
    # test_data = test_data[:, n_validate:]
    validate_label = test_label[0: n_validate]
    # test_label = test_label[n_validate:]

    dataset = {'train_data': train_data, 'test_data': test_data, 'test_label': test_label,
               'validate_data': validate_data, 'validate_label': validate_label}
    return dataset



def get_NAB_dataset(data_path, normalized = True, validation_ratio=0.2):
    dataset = {}
    dataset_folder = data_path
    file_list = os.listdir(dataset_folder)
    for filename in file_list:
        if filename.startswith('ec2'):
            if filename.endswith('train.npy'):
                train_data = np.load(dataset_folder + filename)
                train_data = train_data[0:2001, :]
                train_data = train_data.T

                # plt.plot(train_data[:,0])
                # plt.title('training data')
                # plt.show()
                dataset = {**dataset, 'train_data': train_data}
            if filename.endswith('test.npy'):
                test_data = np.load(dataset_folder + filename)
                test_data = test_data[2001:, :]
                test_data = test_data.T
                # plt.plot(test_data[:, 0])
                # plt.title('testing data')
                # plt.show()
                dataset = {**dataset, 'test_data': test_data}

            if filename.endswith('labels.npy'):
                labels = np.load(dataset_folder + filename)
                print(len(labels))
                labels = labels.flatten()
                labels = labels[2001:]
                # print(labels.shape)
                # plt.plot(labels)
                # plt.title('labels')
                # plt.show()
                dataset = {**dataset, 'test_label': labels}

    if normalized:
        max_value = np.max(dataset['train_data'][0])
        min_value = np.min(dataset['train_data'][0])
        # test_data[:] = 2 * ((test_data[:] - min_value) / (max_value - min_value)) - 1
        # train_data[:] = 2 * ((train_data[:] - min_value) / (max_value - min_value)) - 1
        dataset['test_data'][0][:] = ((dataset['test_data'][0][:] - min_value) / (max_value - min_value))
        dataset['train_data'][0][:] = ((dataset['train_data'][0][:] - min_value) / (max_value - min_value))


    return dataset

def sliding_window_generation(dataset, window_size):
    train_samples, reconstruction_label = training_samples_generation(train_data=dataset['train_data'], window_size=window_size)
    trainset = {'samples': train_samples, 'labels':reconstruction_label}

    test_samples, anomaly_labels = testing_samples_generation(test_data=dataset['test_data'],test_label=dataset['test_label'],window_size=window_size)
    testset = {'samples':test_samples, 'labels': anomaly_labels}

    # validate_samples, val_anomaly_labels = testing_samples_generation(test_data=dataset['validate_data'], test_label=dataset['validate_label'], window_size=window_size)
    # validationset = {'samples': validate_samples, 'labels': val_anomaly_labels}

    # return trainset,testset,validationset
    return trainset, testset

def training_samples_generation(train_data, window_size):

    # generate training data
    dimension = train_data.shape[0]

    samples = np.zeros(shape=(train_data.shape[1] - window_size + 1, dimension, window_size))
    reconstruction_label = np.zeros(shape=(train_data.shape[1] - window_size + 1, dimension, window_size))

    for i in range(0, train_data.shape[-1] - window_size + 1):
        # generate data and reconstructed_label
        reconstruction_label[i] = np.copy(train_data[:, i:i + window_size])
        samples[i] = np.copy(train_data[:, i:i + window_size])

    return samples, reconstruction_label

def testing_samples_generation(test_data, test_label, window_size):

    dimension = test_data.shape[0]

    samples = np.zeros(shape=(test_data.shape[1] - window_size + 1, dimension, window_size))
    test_labels = np.zeros(shape=(test_data.shape[1] - window_size + 1, 1, window_size))

    for i in range(0, test_data.shape[-1] - window_size + 1):
        # generate data and anomaly labels
        test_labels[i] = np.copy(test_label[i:i + window_size])
        samples[i] = np.copy(test_data[:, i:i + window_size])

    return samples, test_labels
