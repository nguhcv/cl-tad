import argparse
import numpy as np

np.set_printoptions(threshold=np.inf)
import torch
import torch.nn as nn
from framework.e2e_frameworks import e2e_framework
from framework.net_generator import net_generator


def main_app(opt):

    #initialize the net components

    net = net_generator(r_module=opt.r_module,
                        t_module=opt.t_module,
                        e_module=opt.e_module,
                        n_dims=opt.dataset_dims,
                        r_channel_list=opt.r_channel_list,
                        e_channel_list=opt.e_channel_list,
                        dropout=opt.dropout,
                        activation=opt.activation,
                        window_size=opt.w,
                        r_kernel_size_list=opt.r_ks_list,
                        e_kernel_size_list=opt.e_ks_list,
                        sti_operations=opt.sti_operation)


    # set model_name
    model_names=[]
    if opt.dataset in ['pd', 'gesture','psm' ]:
        name = str(opt.dataset)+'_'+str(opt.w)+'_'+str(opt.batch_size)+'_R'+str(opt.r_module)+'_T'+str(opt.t_module)+'_E'+str(opt.e_module)
        model_names.append(name)
        #ecg_0_16_256_R1_T1_E1

    elif opt.dataset in ['ecg', 'ucr','nab', 'kpi']:
        name = str(opt.dataset)+'_'+str(opt.ts_num)+'_'+str(opt.w)+'_'+str(opt.batch_size)+'_R'+str(opt.r_module)+'_T'+str(opt.t_module)+'_E'+str(opt.e_module)
        model_names.append(name)

    framework = e2e_framework(dataset=opt.dataset,
                                     ts_num=opt.ts_num,
                                     w=opt.w,
                                     batch_size=opt.test_batch_size,
                                     net_components=net,
                                     data_path=opt.data_path,
                                     model_name=model_names[0]+'_new',
                                     lr=opt.lr,
                                     n_epochs=opt.n_epochs,
                                     saved_path=opt.save_path,
                                     n_dims=opt.dataset_dims,
                                     )

    # framework.plot_dataset(data_type='train', dataset_name='Original training set')   # train, val or test
    # framework.reconstruction_visualization(visualization_batchsize=5)
    # framework.contrastive_visualization(visualization_batchsize=5)


    # F1_score = framework.main_test(distance_type='eu',
    #                                measure_method='average',
    #                                delay=None)
    # print(F1_score)

    F1_score,precison,recall = framework.main_test2(distance_type='eu',f1_type='f1_best', delay=None)
    print(F1_score, precison,recall)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    #define net structure
    parser.add_argument('--r_module', default=1, help='model type: stacked_tcn:1, local_tcn:2, sti:3')
    parser.add_argument('--t_module', default=1, help='model type: stacked_tcn:1, linear:2')
    parser.add_argument('--e_module', default=1, help='model type: stacked-tcn:1, sti:2')
    parser.add_argument('--dropout', default=0., help='model type: stacked-tcn:1, sti:2')
    parser.add_argument('--r_ks_list', default=[3], help='ks-list of tcn: [2], [3],  [2,3], [2,3,5] number of elements = number of tcns')
    parser.add_argument('--e_ks_list', default=[3], help='ks-list of tcn: [2], [3],  [2,3], [2,3,5] number of elements = number of tcns')
    parser.add_argument('--r_channel_list', default=[128,128,128,128,128,128], help='Channel list at each layer of TCN')
    parser.add_argument('--e_channel_list', default=[128,128,128,128,128,128], help='Channel list at each layer of TCN')
    parser.add_argument('--activation', default='relu', help='activation in TCN')
    parser.add_argument('--sti_operation', default=('sub', 'plus'), help='operation pair : (sub, plus), (sub,sub), (plus,sub), (plus,plus)')


    #training setting
    parser.add_argument('--batch_size', default=256, help='number of samples in each batch') # 64 or 256
    parser.add_argument('--test_batch_size', default=256, help='number of samples in each batch')  # 64 or 256
    parser.add_argument('--w', default=16, type=int, help='window size of each sample')     # 16 or 32 or 64
    parser.add_argument('--n_epochs', default=3000, type=int, help='number of training epochs in reconstruction phase')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate of reconstruction phase')


    #dataset setting
    parser.add_argument('--dataset', default='ecg')
    parser.add_argument('--dataset_dims', default=2,type= int, help= 'number of dimension of dataset')
    parser.add_argument('--ts_num', default=0)
    parser.add_argument('--data_path', default='C:/Project/TSAD/data/ecg/', help='data-path')
    parser.add_argument('--save_path', default='C:/Project/TSAD/saved_modules/e2e/',
                        help='save-data-path')

    # parser.add_argument('--dataset_dims', default=1,type= int, help= 'number of dimension of dataset')
    # parser.add_argument('--dataset', default='pd')
    # parser.add_argument('--ts_num', default=0)
    # parser.add_argument('--data_path', default='C:/Project/TSAD/data/power_demand/', help='data-path')
    # parser.add_argument('--save_path', default='C:/Project/TSAD/saved_modules/e2e/',
    #                     help='save-data-path')

    # parser.add_argument('--dataset_dims', default=2,type= int, help= 'number of dimension of dataset')
    # parser.add_argument('--dataset', default='gesture')
    # parser.add_argument('--ts_num', default=0)
    # parser.add_argument('--data_path', default='C:/Project/TSAD/data/gesture/', help='data-path')
    # parser.add_argument('--save_path', default='C:/Project/TSAD/saved_modules/e2e/',
    #                     help='save-data-path')

    # parser.add_argument('--dataset_dims', default=1,type= int, help= 'number of dimension of dataset')
    # parser.add_argument('--dataset', default='nab')
    # parser.add_argument('--ts_num', default=0)
    # parser.add_argument('--data_path', default='C:/Project/TSAD/data/nab/', help='data-path')
    # parser.add_argument('--save_path', default='C:/Project/TSAD/saved_modules/e2e/',
    #                     help='save-data-path')

    # parser.add_argument('--dataset_dims', default=1,type= int, help= 'number of dimension of dataset')
    # parser.add_argument('--dataset', default='ucr')
    # parser.add_argument('--ts_num', default=0)
    # parser.add_argument('--data_path', default='C:/Project/TSAD/data/others/UCR/', help='data-path')
    # parser.add_argument('--save_path', default='C:/Project/TSAD/saved_modules/e2e/',
    #                     help='save-data-path')

    # parser.add_argument('--dataset_dims', default=1,type= int, help= 'number of dimension of dataset')
    # parser.add_argument('--dataset', default='kpi')
    # parser.add_argument('--ts_num', default=3)
    # parser.add_argument('--data_path', default='C:/Project/TSAD/data/others/kpi2/', help='data-path')
    # parser.add_argument('--save_path', default='C:/Project/TSAD/saved_modules/e2e/',
    #                     help='save-data-path')

    # parser.add_argument('--dataset_dims', default=25,type= int, help= 'number of dimension of dataset')
    # parser.add_argument('--dataset', default='psm')
    # parser.add_argument('--ts_num', default=0)
    # parser.add_argument('--data_path', default='C:/Project/TSAD/data/others/psm', help='data-path')
    # parser.add_argument('--save_path', default='C:/Project/TSAD/saved_modules/e2e/',
    #                     help='save-data-path')

    main_app(parser.parse_args())

    pass