import torch
from utils.mask import sequence_order_position, masked_batch_generation_modified
from utils.eval import pca
import matplotlib.pyplot as plt



def contrastive_visualization(
                         F,
                         train_loader,
                         masking_factor,
                         w,
                        pca_dimension):
    # 1. pre-training reconstruction model
    # Feed model to cuda
    F = F.cuda()

    with torch.no_grad():

        for b_index, (data, label) in enumerate(train_loader):

            if b_index ==0:
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

                pca(output=features, data_features=data_features,w=w,s=data.shape[0],dimension=pca_dimension)



def reconstruction_visualization(
                            R,
                            train_loader,
                            w,dataset_name):

    #Feed model to cuda
    R = R.cuda()
    with torch.no_grad():

        for b_index, (data, label) in enumerate(train_loader):
            if b_index == 0:
                # generate a mask-position list
                sequence_list = sequence_order_position(window_size=w,data_dimension=1)

                # for each batch, generate a masked_batch

                generated_batch, generated_labels = masked_batch_generation_modified(data_batch=data, random_pos=sequence_list,
                                                                            window_size=w, mask_value=0.)


                #print masked samples
                if dataset_name in ['ecg', 'gesture']:

                    for i in range (5, 16, 5):
                        plt.plot(generated_batch[i][0])
                        plt.plot(generated_batch[i][1])
                        plt.title('Mask sample at ' + str(i))
                        plt.xlabel('time point')
                        plt.ylabel('value')
                        plt.show()

                    plt.plot(generated_labels[5][0])
                    plt.plot(generated_labels[5][1])
                    plt.title('reconstruction label')
                    plt.xlabel('time point')
                    plt.ylabel('value')
                    plt.show()


                if dataset_name =='pd':
                    for i in range (5, 16, 5):
                        plt.plot(generated_batch[i][0])
                        plt.title('Mask sample at ' + str(i))
                        plt.xlabel('time point')
                        plt.ylabel('values')
                        plt.show()
                    plt.plot(generated_labels[5][0])
                    plt.title('reconstruction label')
                    plt.xlabel('time point')
                    plt.ylabel('value')
                    plt.show()


                # feed data to GPUs
                generated_batch = generated_batch.cuda()
                # generated_labels = generated_labels.cuda()

                # forward generated-batch
                rc_output = R(generated_batch)

                rc_output = rc_output.cpu().detach().numpy()
                generated_labels = generated_labels.numpy()

                if dataset_name in ['ecg', 'gesture']:

                    for i in range(5, 16, 5):
                        plt.plot(rc_output[i][0])
                        plt.plot(rc_output[i][1])
                        plt.title('Reconstructed sample at' + str(i))
                        plt.xlabel('time point')
                        plt.ylabel('value')
                        plt.show()


                if dataset_name == 'pd':
                    for i in range(5, 16, 5):
                        plt.plot(rc_output[i][0])
                        plt.title('Reconstructed sample at' + str(i))
                        plt.xlabel('time point')
                        plt.ylabel('values')
                        plt.show()


                #print error
                if dataset_name in ['ecg', 'gesture']:

                    for i in range(5, 16, 5):
                        error1 = abs(rc_output[i][0] - generated_labels[i][0])
                        error2 = abs(rc_output[i][1] - generated_labels[i][1])
                        plt.plot(error1)
                        plt.plot(error2)
                        plt.title('Error at' + str(i))
                        plt.xlabel('time point')
                        plt.ylabel('value')
                        plt.show()

                if dataset_name =='pd':

                    for i in range(5, 16, 5):
                        error1 = abs(rc_output[i][0] - generated_labels[i][0])
                        plt.plot(error1)
                        plt.title('Error at' + str(i))
                        plt.xlabel('time point')
                        plt.ylabel('value')
                        plt.show()


def hybrid_visualization(
                         F,
                         train_loader,
                         masking_factor,
                         w,
                        pca_dimension,
                        ds_name):
    # 1. pre-training reconstruction model
    # Feed model to cuda
    F = F.cuda()

    with torch.no_grad():

        for b_index, (data, label) in enumerate(train_loader):

            if b_index ==0:
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
                rc_output, features, uncertainty = F(generated_batch)

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

                pca(output=features, data_features=data_features,w=w,s=data.shape[0],dimension=pca_dimension,ds_name=ds_name)

