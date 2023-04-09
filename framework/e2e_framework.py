import torch
from numba import cuda
from train.e2e_training import end_to_end_learning
from modules.Full_Model import Full_Model
from test_validate.end_to_end_testing import end_to_end_testing,end_to_end_validation, confident_based_testing, threshold_determination, threshold_list_determination
from utils.eval import F1_PA
from visualization.visualization import hybrid_visualization, reconstruction_visualization
import matplotlib.pyplot as plt
from pathlib import Path

from utils.ds_reader import Dataset_Loader


class e2e_framework():
    def __init__(self, dataset,
                 ts_num,
                 w,
                 batch_size,
                 net_components,
                 data_path,
                 model_name,
                 lr,
                 n_epochs,
                 saved_path,
                 n_dims,
                 masking_ratio=1.,
                 cosine=True,
                 lars=True):

        # initialize dataset_loader

        self.dataset = Dataset_Loader(dataset=dataset, data_path=data_path, ts_num=ts_num, window_size=w)
        self.ds_name = dataset
        self.w = w
        self.batch_size = batch_size
        self.net_components = net_components
        self.masking_ratio = masking_ratio
        self.model_name = model_name
        self.lr = lr
        self.n_epoch = n_epochs
        self.save_path = saved_path
        self.n_dims = n_dims
        self.cosine = cosine
        self.lars = lars

        # generate dataset class

    def plot_dataset(self, data_type, dataset_name):
        self.dataset.plot(data_type=data_type, dataset_name=dataset_name)
        pass

    def train(self):
        hb_model = Path(self.save_path + self.model_name)


        if hb_model.exists():
            raise ValueError('The model already trained ')

        else:

            # generate loader for contrastive learning
            train_loader = self.dataset.train_loader_generation(batch_size=self.batch_size, shuffle=True)
            F = Full_Model(net=self.net_components,
                           w_size=self.w,
                           n_dims=self.n_dims, mode=1)

            end_to_end_learning(model_name=self.model_name,
                                F=F,
                                train_loader=train_loader,
                                masking_factor=self.masking_ratio,
                                lr = self.lr,
                                w=self.w,
                                save_path=self.save_path,
                                n_epochs=self.n_epoch,
                                cosine_option=self.cosine,
                                lars_option=self.lars)

    def test(self, distance_type= 'eu', measure_method = 'average',ratio=1.):

        hb_model = Path(self.save_path + self.model_name)


        if hb_model.exists():
            state_dict = torch.load(hb_model)
            print('best epoch is: ' + str(state_dict['epoch']))
            print('best loss is: ' + str(state_dict['best_loss']))
            print('test')

            val_loader, test_loader = self.dataset.val_test_loader_generation(batch_size=self.batch_size,
                                                                              shuffle=False)
            train_loader = self.dataset.train_loader_generation(batch_size=self.batch_size, shuffle=False)


            F = Full_Model(net=self.net_components,
                           w_size=self.w,
                           n_dims=self.n_dims, mode=1)

            F.load_state_dict(state_dict['model'])

            # determine the threshold
            threshold = end_to_end_validation(F=F,
                                              val_loader=train_loader,
                                              masking_factor=1.,
                                              w=self.w,measure_method=measure_method,
                                              distance_type=distance_type,ratio=ratio)

            # print('threshold is')
            print(threshold)

            anomalies_values = end_to_end_testing(F=F, test_loader=test_loader, masking_factor=self.masking_ratio, w=self.w, threshold=threshold)

            score = F1_PA(groundtrue=self.dataset.dataset['test_label'], predicted=anomalies_values)
            return score

        else:
            raise ValueError('The model has not trained yet')

        pass

    def main_test(self, distance_type='eu', measure_method='average', delay=None):

        hb_model = Path(self.save_path + self.model_name)

        if hb_model.exists():
            state_dict = torch.load(hb_model)
            print('best epoch is: ' + str(state_dict['epoch']))
            print('best loss is: ' + str(state_dict['best_loss']))
            print('test')

            test_loader = self.dataset.val_test_loader_generation(batch_size=self.batch_size,
                                                                              shuffle=False)
            train_loader = self.dataset.train_loader_generation(batch_size=self.batch_size, shuffle=False)

            F = Full_Model(net=self.net_components,
                           w_size=self.w,
                           n_dims=self.n_dims, mode=1)

            F.load_state_dict(state_dict['model'])

            # determine the threshold
            threshold = confident_based_testing(F=F,
                                              val_loader=train_loader,
                                              masking_factor=1.,
                                              w=self.w, measure_method=measure_method,
                                              distance_type=distance_type, delay=delay)

            print('threshold is')
            print(threshold)

            anomalies_values = end_to_end_testing(F=F, test_loader=test_loader, masking_factor=self.masking_ratio,
                                                  w=self.w, threshold=threshold)

            score = F1_PA(groundtrue=self.dataset.dataset['test_label'], predicted=anomalies_values, delay=delay)
            return score

        else:
            raise ValueError('The model has not trained yet')

        pass




    def main_test2(self, distance_type='eu', f1_type='best',  delay=None):

        hb_model = Path(self.save_path + self.model_name)

        if hb_model.exists():
            state_dict = torch.load(hb_model)
            print('best epoch is: ' + str(state_dict['epoch']))
            print('best loss is: ' + str(state_dict['best_loss']))
            print('test')

            test_loader = self.dataset.val_test_loader_generation(batch_size=self.batch_size,
                                                                              shuffle=False)
            train_loader = self.dataset.train_loader_generation(batch_size=self.batch_size, shuffle=False)

            F = Full_Model(net=self.net_components,
                           w_size=self.w,
                           n_dims=self.n_dims, mode=1)

            F.load_state_dict(state_dict['model'])

            if f1_type == 'f1_pa':
                # determine the threshold
                threshold = threshold_determination(F=F,
                                                    val_loader=train_loader,
                                                    masking_factor=1.,
                                                    w=self.w,
                                                    distance_type=distance_type)

                print('threshold')
                print(threshold)

                anomalies_values = end_to_end_testing(F=F, test_loader=test_loader, masking_factor=self.masking_ratio,
                                                      w=self.w, threshold=threshold)

                score, precision, recall = F1_PA(groundtrue=self.dataset.dataset['test_label'], predicted=anomalies_values, delay=delay)
                return score,precision,recall

            elif f1_type=='f1_best':
                threshold_list = threshold_list_determination(F=F,
                                                              test_loader=test_loader,
                                                              masking_factor=1.,
                                                              w=self.w)
                best_score =[0,0,0]
                f1_list=[]
                # for delta in threshold_list:
                #     anomalies_values = end_to_end_testing(F=F, test_loader=test_loader,
                #                                           masking_factor=self.masking_ratio,
                #                                           w=self.w, threshold=delta)
                #
                #     score, precision, recall = F1_PA(groundtrue=self.dataset.dataset['test_label'],
                #                                      predicted=anomalies_values, delay=delay)
                #     print(delta,score,precision,recall)
                #
                #     if best_score[0]<=score:
                #         best_score[0] = score
                #         best_score[1] = precision
                #         best_score[2] = recall
                #     f1_list.append(score)
                # plt.plot(threshold_list,f1_list)
                # plt.xlabel(  r'threshold \Delta')
                # plt.ylabel('F1')
                # plt.title(self.ds_name)
                # plt.show()
                # return best_score

                for g in range (int(len(threshold_list)/2),len(threshold_list), 1):
                    anomalies_values = end_to_end_testing(F=F, test_loader=test_loader,
                                                          masking_factor=self.masking_ratio,
                                                          w=self.w, threshold=threshold_list[g])

                    score, precision, recall = F1_PA(groundtrue=self.dataset.dataset['test_label'],
                                                     predicted=anomalies_values, delay=delay)
                    print(threshold_list[g],score,precision,recall)

                    if best_score[0]<=score:
                        best_score[0] = score
                        best_score[1] = precision
                        best_score[2] = recall
                    f1_list.append(score)
                plt.plot(threshold_list,f1_list)
                plt.xlabel(  r'threshold \Delta')
                plt.ylabel('F1')
                plt.title(self.ds_name)
                plt.show()
                return best_score






        else:
            raise ValueError('The model has not trained yet')

        pass



    def contrastive_visualization(self, visualization_batchsize, pca_dimension=2):
        hb_model = Path(self.save_path + self.model_name)

        if hb_model.exists():
            state_dict = torch.load(hb_model)
            print('best epoch is: ' + str(state_dict['epoch']))
            print('best loss is: ' + str(state_dict['best_loss']))

            train_loader = self.dataset.train_loader_generation(batch_size=visualization_batchsize, shuffle=True)
            F = Full_Model(net=self.net_components,
                           w_size=self.w,
                           n_dims=self.n_dims, mode=1)

            hybrid_visualization(F=F, train_loader=train_loader, masking_factor=self.masking_ratio, w=self.w,
                                      pca_dimension=pca_dimension,ds_name = self.ds_name)

            print('load trained weights')
            F.load_state_dict(state_dict['model'])

            hybrid_visualization(F=F, train_loader=train_loader, masking_factor=self.masking_ratio, w=self.w,
                                      pca_dimension=pca_dimension, ds_name=self.ds_name)
        else:
            raise ValueError('The model has not trained yet')

    def reconstruction_visualization(self, visualization_batchsize):
        hb_model = Path(self.save_path + self.model_name)

        if hb_model.exists():
            state_dict = torch.load(hb_model)
            print('best epoch is: ' + str(state_dict['epoch']))
            print('best loss is: ' + str(state_dict['best_loss']))

            train_loader = self.dataset.train_loader_generation(batch_size=visualization_batchsize, shuffle=False)
            F = Full_Model(net=self.net_components,
                           w_size=self.w,
                           n_dims=self.n_dims, mode=1)


            print('load trained weights')
            F.load_state_dict(state_dict['model'])

            # for param in F.R.parameters():
            #     print(torch.mean(param.data))


            reconstruction_visualization(R=F.R, train_loader=train_loader, w=self.w, dataset_name=self.ds_name)



        else:
            raise ValueError('The model has not trained yet')




