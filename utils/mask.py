import numpy as np
import torch


def random_position2(window_size, data_dimension, masking_factor):
    num_list = np.arange(0, window_size, 1)
    rand_list = []
    rand_list.append(np.random.choice(num_list, int(window_size * masking_factor), replace=False))
    return rand_list

def random_position(window_size, data_dimension, masking_factor):
    num_list = np.arange(0, window_size, 1)
    rand_list = []

    for j in range(data_dimension):
        rand_list.append(np.random.choice(num_list, int(window_size*masking_factor), replace=False))
    return rand_list

def sequence_order_position(window_size, data_dimension):
    num_list = np.arange(0, window_size, 1)
    rand_list = []
    rand_list.append(num_list)
    return rand_list



def masked_batch_generation(data_batch, random_pos, window_size, mask_value):

    n_batch_samples = len(data_batch)

    n_batch_masked_samples = len(random_pos[0])* n_batch_samples

    batch_data = torch.zeros((n_batch_masked_samples, data_batch.shape[1], window_size))
    reconstruct_label_batch = torch.zeros((n_batch_masked_samples, data_batch.shape[1], window_size))

    for i in range(n_batch_samples):  # for each each sample in data_batch
        sliding_window = torch.clone(data_batch[i])
        for j in range(i * len(random_pos[0]), i * len(random_pos[0]) + len(random_pos[0])):  # determine the range for filling data samples
            masked_data = torch.clone(sliding_window)
            reconstruct_label_batch[j] = torch.clone(sliding_window)
            for l in range(masked_data.shape[0]):  # random mask in each dimension
                mask_loc = random_pos[l][j - (i * len(random_pos[0]))]
                masked_data[l][mask_loc] = mask_value

            batch_data[j] = masked_data

    return batch_data, reconstruct_label_batch

    pass


def masked_batch_generation_modified(data_batch, random_pos, window_size, mask_value):

    n_batch_samples = len(data_batch)
    # print(n_batch_samples)

    n_batch_masked_samples = len(random_pos[0])* n_batch_samples



    # generate data_batch
    batch_data = torch.zeros((n_batch_masked_samples, data_batch.shape[1], window_size))
    reconstruct_label_batch = torch.zeros((n_batch_masked_samples, data_batch.shape[1], window_size))

    for i in range(n_batch_samples):  # for each each sample in data_batch
        sliding_window = torch.clone(data_batch[i])
        for j in range(i * len(random_pos[0]), i * len(random_pos[0]) + len(random_pos[0])):  # determine the range for filling data samples
            masked_data = torch.clone(sliding_window)
            reconstruct_label_batch[j] = torch.clone(sliding_window)
            for l in range(masked_data.shape[0]):  # random mask in each dimension
                mask_loc = random_pos[0][j - (i * len(random_pos[0]))]
                masked_data[l][mask_loc] = mask_value

            batch_data[j] = masked_data

    return batch_data, reconstruct_label_batch

    pass



def masked_batch_generation2(data_batch, masking_factor, window_size, mask_value):

    n_batch_samples = len(data_batch)
    n_dims = data_batch.shape[1]
    # print(n_batch_samples)
    n_batch_masked_samples = int(window_size* masking_factor* n_batch_samples)

