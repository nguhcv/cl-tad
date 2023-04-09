from torch import nn
import torch
import argparse
import numpy as np
from time_series.final_idea.framework.modules.sti_net import StackedSTINet
from time_series.final_idea.framework.modules.TCN_Module import Stacked_TCN
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    # print("Initial GPU Usage")
    # gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    # print("GPU Usage after emptying the cache")
    # gpu_usage()

class Construction_Model(nn.Module):
    def __init__(self,
                 prediction_output_len,
                 prediction_tcn_channels,
                 input_len,
                 input_dim,
                 prediction_output_dim,
                 operation,
                 tcn_extraction_ks_list,
                 tcn_dropout=0.25,
                 num_levels=4,
                 activation ='relu',
                 ):
        super(Construction_Model, self).__init__()

        self.input_len = input_len
        self.num_levels = num_levels

        #define STI-net encoder with decoder
        self.reconstructionNet = StackedSTINet(output_len=prediction_output_len,
                                               input_len=input_len,
                                               channels=prediction_tcn_channels,
                                               input_dim=input_dim,
                                               output_dim=prediction_output_dim,
                                               operation=operation,
                                               dropout=tcn_dropout,
                                               decoder_include=True,
                                               tcn_ks_list=tcn_extraction_ks_list
                                               ).cuda()

    def forward(self, masked_batch):

        #Check input length should evenly divided for 2^level
        assert self.input_len % (np.power(2,self.num_levels)) == 0  # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)

        # reconstruction Net
        reconstruct_op = self.reconstructionNet(masked_batch)
        return reconstruct_op


if __name__ == '__main__':
    free_gpu_cache()
    parser = argparse.ArgumentParser()

    model = Construction_Model(prediction_output_len=128,
                         prediction_tcn_channels=[16,16,16,16,16,16],
                         input_len=128,
                         input_dim=1,
                         prediction_output_dim=1,
                         operation=('sub','plus'),
                         tcn_extraction_ks_list=[4]).cuda()

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model_size is '+ str(size_all_mb))

    #
    masked_batch = torch.randn(2048, 128, 1).cuda()     #(batch-size, input_len, dimension )
    original_batch = torch.randn(1, 512, 1).cuda()     #(batch-size, input_len, dimension )
    y = model(masked_batch)
    print(y.shape)
    #
