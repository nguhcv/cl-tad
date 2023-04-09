import torch

from utils.mask import sequence_order_position, masked_batch_generation_modified
from utils.eval import euclidean,euclidean_testing, mahalanobis, mahalanobis_testing
import numpy as np
from utils.eval import euclidean,euclidean_testing, avg_euclidean
import matplotlib.pyplot as plt
import seaborn as sns
def confident_based_testing(
                         F,
                         val_loader,
                         masking_factor,
                         measure_method,
                         distance_type,

                         w):
    # 1. pre-training reconstruction model
    # Feed model to cuda
    F = F.cuda()
    value_list=[]
    with torch.no_grad():

        for b_index, (data, label) in enumerate(val_loader):
            if masking_factor == 1.:

                rand_pos_list = sequence_order_position(window_size=w, data_dimension=data.shape[1])

                # for each batch, generate a masked_batch

                generated_batch, generated_labels = masked_batch_generation_modified(data_batch=data,
                                                                                     random_pos=rand_pos_list,
                                                                                     window_size=w, mask_value=0.)
            else:
                raise ValueError('not accepted')

            # feed data to GPUs
            generated_batch = generated_batch.cuda()

            # forward generated-batch
            features, uncertainty = F(generated_batch)

            # feed data-batch to GPUs
            data = data.cuda()

            # generate encoder'output of data
            batch_output = F.E(data)
            batch_output = F.flat(batch_output)
            batch_output = F.linear(batch_output)
            if F.bn:
                batch_output = F.batchnorm(batch_output)
            batch_output = F.relu(batch_output)
            batch_output = F.linear1(batch_output)

            data_features = batch_output[:, :-1]
            # C = batch_output[:, -1]

            if distance_type =='eu':
                value_list = avg_euclidean(E=features, U=data_features, w=w, value_list=value_list)

        print(len(value_list))
        print('mean is:' + str(np.mean(np.asarray(value_list))))
        print('std is :' + str(np.std(np.asarray(value_list))))
        sns.histplot(value_list)
        plt.title('Histogram of (U,E) pairs in dataset ECG-A (16,32)')
        plt.xlabel('distance')
        plt.ylabel('number of appearances')
        plt.show()

        # print(st.t.interval(alpha=confident_value, df=len(value_list) - 1,
        #                     loc=np.mean(value_list),
        #                     scale=st.sem(value_list)))
        # breakpoint()
        mean = np.mean(np.asarray(value_list))
        std = np.std(np.asarray(value_list))
        # mean = round(mean, 1)
        # std = round(np.std(np.asarray(value_list)),2)

        for k in range (3,0,-1):
            if mean + (k*std)<=1:
                return mean + (k * std)
                # return round((mean + (k*std)),1)



def threshold_determination(
                         F,
                         val_loader,
                         masking_factor,
                         distance_type,
                         w):

    # Feed model to cuda
    F = F.cuda()
    value_list=[]
    with torch.no_grad():

        for b_index, (data, label) in enumerate(val_loader):
            if masking_factor == 1.:

                rand_pos_list = sequence_order_position(window_size=w, data_dimension=data.shape[1])

                # for each batch, generate a masked_batch

                generated_batch, generated_labels = masked_batch_generation_modified(data_batch=data,
                                                                                     random_pos=rand_pos_list,
                                                                                     window_size=w, mask_value=0.)
            else:
                raise ValueError('not accepted')

            # feed data to GPUs
            generated_batch = generated_batch.cuda()

            # forward generated-batch
            features, uncertainty = F(generated_batch)

            # feed data-batch to GPUs
            data = data.cuda()

            # generate encoder'output of data
            batch_output = F.E(data)
            batch_output = F.flat(batch_output)
            batch_output = F.linear(batch_output)
            if F.bn:
                batch_output = F.batchnorm(batch_output)
            batch_output = F.relu(batch_output)
            batch_output = F.linear1(batch_output)

            data_features = batch_output[:, :-1]
            # C = batch_output[:, -1]

            if distance_type =='eu':
                value_list = avg_euclidean(E=features, U=data_features, w=w, value_list=value_list)

        print(len(value_list))

        # hist, bin_edges = np.histogram(value_list,bins=np.arange(0.,2., 0.005))
        #
        # n = len(hist)
        # x_hist = np.zeros((n), dtype=float)
        # for ii in range(n):
        #     x_hist[ii] = (bin_edges[ii + 1] + bin_edges[ii]) / 2
        #
        # y_hist = hist
        # print(type(y_hist))
        #
        # param_optimised, param_covariance_matrix = curve_fit(Gauss, x_hist, y_hist)
        # print(param_optimised[0], param_optimised[1], param_optimised[2])
        #
        # fit_gauss = Gauss(data=x_hist, alpha=param_optimised[0], mean=param_optimised[1], sigma=param_optimised[2])
        # sns.histplot(value_list, bins=np.arange(0.,2.,0.05))
        # plt.xlabel('distance between positive pairs')
        # plt.ylabel('number of appearances')
        # plt.show()
        #
        #
        # sns.histplot(value_list, bins=np.arange(0., 2., 0.05))
        # plt.plot(x_hist, fit_gauss, color='red', label='Gaussian')
        # plt.title('ECG-A (16-32)')
        # plt.xlabel('distance between positive pairs')
        # plt.ylabel('number of appearances')
        # plt.show()

        '''----------------'''
        # sns.histplot(value_list, bins=200, color='blue')
        # plt.show()
        # f = Fitter(value_list, distributions=['gamma', 'norm'], bins=200)
        # f.fit()
        # f.summary()
        # # f.hist()
        # plt.show()
        #
        # #
        # # f.fitted_param()
        # f.get_best()
        # print(f.get_best())
        #
        # params = st.gamma.fit(value_list)
        # print(st.gamma.mean(*params))
        # print(st.gamma.std(*params))
        # print(params)
        # #
        # #
        # #
        # #
        # #
        # print('----')
        # params2 = st.norm.fit(value_list)
        # print(st.norm.mean(*params2))
        # print(st.norm.std(*params2))
        #
        # print('---')
        #
        # print('mean is:' + str(np.mean(np.asarray(value_list))))
        # print('std is :' + str(np.std(np.asarray(value_list))))
        # print(params2)
        #
        # print('----')
        # params3 = st.pareto.fit(value_list)
        # print(st.pareto.mean(*params3))
        # print(st.pareto.std(*params3))
        # print(params3)


        # pdf = st.norm.pdf(x_hist, *params2[:-2], loc=params2[-2], scale=params2[-1])
        # plt.plot(pdf)
        # plt.show()

        # plt.plot(y_hist)

        # dist_names = ['gamma', 'norm']
        #
        # for dist_name in dist_names:
        #     dist = getattr(scipy.stats, dist_name)
        #     params = dist.fit(y_hist)
        #     arg = params[:-2]
        #     loc = params[-2]
        #     scale = params[-1]
        #     if arg:
        #         pdf_fitted = dist.pdf(y_hist, *arg, loc=loc, scale=scale)
        #     else:
        #         pdf_fitted = dist.pdf(x_hist, loc=loc, scale=scale)
        #     plt.plot(pdf_fitted, label=dist_name)
        # plt.legend(loc='upper right')
        # plt.show()


        # breakpoint()


        #gamma fit
        # gamma = st.gamma
        # param_gamma= gamma.fit(y_hist, floc=0)
        # print(param_gamma)
        # pdf_fit = gamma.pdf(x_hist, *param_gamma)
        # sns.histplot(value_list, bins=np.arange(0., 2., 0.005))
        # plt.plot(x_hist, pdf_fit, color='green', label='Gamma')
        # plt.show()



        # breakpoint()



        #
        # print('mean is:' + str(np.mean(np.asarray(value_list))))
        # print('std is :' + str(np.std(np.asarray(value_list))))
        # sns.histplot(value_list)
        # plt.title('Histogram of (U,E) pairs in dataset ECG-A (16-32)')
        # plt.xlabel('distance')
        # plt.ylabel('number of appearances')
        # plt.show()
        #
        mean = np.mean(np.asarray(value_list))
        std = np.std(np.asarray(value_list))
        # # mean = round(mean, 1)
        # # std = round(np.std(np.asarray(value_list)),2)
        #
        # for k in range (3,0,-1):
        #     if param_optimised[1] + (k*param_optimised[2])<=1:
        #         return param_optimised[1] + (k * param_optimised[2])
        #         # return round((mean + (k*std)),1)

        for k in range (3,0,-1):
            if mean + (k*std)<=1:
                return mean + (k * std)






def contrastive_validation(
                         F,
                         val_loader,
                         masking_factor,
                         w,
                         distance_type):
    # 1. pre-training reconstruction model
    # Feed model to cuda
    F = F.cuda()
    max_validation = 0
    with torch.no_grad():

        for b_index, (data, label) in enumerate(val_loader):
            if masking_factor == 1.:

                rand_pos_list = sequence_order_position(window_size=w, data_dimension=data.shape[1])

                # for each batch, generate a masked_batch

                generated_batch, generated_labels = masked_batch_generation_modified(data_batch=data,
                                                                                     random_pos=rand_pos_list,
                                                                                     window_size=w, mask_value=0.)
            else:
                raise ValueError('not accepted')

            # feed data to GPUs
            generated_batch = generated_batch.cuda()

            # forward generated-batch
            features, uncertainty = F(generated_batch)

            # feed data-batch to GPUs
            data = data.cuda()

            # generate encoder'output of data
            batch_output = F.E(data)
            batch_output = F.flat(batch_output)
            batch_output = F.linear(batch_output)
            if F.bn:
                batch_output = F.batchnorm(batch_output)
            batch_output = F.relu(batch_output)
            batch_output = F.linear1(batch_output)

            data_features = batch_output[:, :-1]
            C = batch_output[:, -1]

            if distance_type=='eu':
                # mahalanobis(E=features, U=data_features, w=w)
                max_val = euclidean(E=features, U=data_features, w=w)

                max_validation = max(max_validation, max_val)
            elif distance_type=='ma':
                max_val = mahalanobis(E=features, U=data_features, w=w)
                max_validation = max(max_validation, max_val)

    return max_validation



def contrastive_testing(
                         F,
                         test_loader,
                         masking_factor,
                         threshold,
                         w,
                         distance_type):
    # 1. pre-training reconstruction model
    # Feed model to cuda
    F = F.cuda()
    anomalies_values =[]

    with torch.no_grad():

        for b_index, (data, label) in enumerate(test_loader):
            if masking_factor == 1:

                rand_pos_list = sequence_order_position(window_size=w, data_dimension=data.shape[1])

                # for each batch, generate a masked_batch

                generated_batch, generated_labels = masked_batch_generation_modified(data_batch=data,
                                                                                     random_pos=rand_pos_list,
                                                                                     window_size=w, mask_value=0.)
            else:
                raise ValueError ('not accepted')
            # feed data to GPUs
            generated_batch = generated_batch.cuda()

            # forward generated-batch
            features, uncertainty = F(generated_batch)

            # feed data-batch to GPUs
            data = data.cuda()

            # generate encoder'output of data
            batch_output = F.E(data)
            batch_output = F.flat(batch_output)
            batch_output = F.linear(batch_output)
            if F.bn:
                batch_output = F.batchnorm(batch_output)
            batch_output = F.relu(batch_output)
            batch_output = F.linear1(batch_output)

            data_features = batch_output[:, :-1]
            C = batch_output[:, -1]

            if distance_type=='eu':
                anomalies = euclidean_testing(E=features, U=data_features, w=w,th=threshold)
                if b_index ==0:
                    for i in range (len(anomalies[0])):
                        anomalies_values.append(anomalies[0][i])
                    for j in range (1, len(anomalies)):
                        anomalies_values.append(anomalies[j][-1])
                else:
                    for k in range (0, len(anomalies)):
                        anomalies_values.append(anomalies[k][-1])
            elif distance_type=='ma':
                anomalies = mahalanobis_testing(E=features, U=data_features, w=w, th=threshold)
                if b_index == 0:
                    for i in range(len(anomalies[0])):
                        anomalies_values.append(anomalies[0][i])
                    for j in range(1, len(anomalies)):
                        anomalies_values.append(anomalies[j][-1])
                else:
                    for k in range(0, len(anomalies)):
                        anomalies_values.append(anomalies[k][-1])
    # print(len(anomalies_values))
    # print(anomalies_values)
    return anomalies_values

