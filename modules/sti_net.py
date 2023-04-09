import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import numpy as np
from modules.TCN_Module import Stacked_TCN


class CustomLinear(nn.Module):
    def __init__(self, input_shape:tuple, output_shape:tuple):
        super().__init__()
        weights = torch.Tensor(output_shape[0], input_shape[0])
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(output_shape[0], output_shape[1])
        self.bias = nn.Parameter(bias)

        # nn.init.kaiming_uniform_(self.weights,  a=math.sqrt(5)) # weight init
        nn.init.normal_(self.weights, mean=0.0, std=0.01)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        # print(self.weights.shape)
        w_times_x= torch.matmul(self.weights,x )
        return torch.add(w_times_x, self.bias)  # w times x + b



class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(self,
                 input_dim,
                 operation,
                 activation,
                 input_len,
                 current_level,
                 channels,
                 dropout,
                 tcn_ks_list,
                 splitting=True,
                 INN=True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.operation = operation
        self.channels = channels
        self.current_level = current_level
        self.dropout = dropout
        self.splitting = splitting
        self.split = Splitting()
        self.tcn_ks_list = tcn_ks_list




        #define 4 TCN module block
        self.TCN1 = Stacked_TCN(n_dims=input_dim,
                               num_channels=self.channels,
                               dropout=dropout,
                               activation=activation,
                               window_size=int(input_len/2),
                                ks_list=tcn_ks_list
                                )

        self.TCN2 = Stacked_TCN(n_dims=input_dim,
                               num_channels=self.channels,
                               dropout=dropout,
                               activation=activation,
                               window_size=int(input_len/2),
                                ks_list=tcn_ks_list)
        self.TCN3 =Stacked_TCN(n_dims=input_dim,
                               num_channels=channels,
                               dropout=dropout,
                               activation=activation,
                               window_size=int(input_len/2),
                               ks_list=tcn_ks_list)
        self.TCN4 = Stacked_TCN(n_dims=input_dim,
                               num_channels=channels,
                               dropout=dropout,
                               activation=activation,
                               window_size=int(input_len/2),
                                ks_list=tcn_ks_list)
        prev_size = 1



    def forward(self, x):
        # print('-------------------')
        # print('process block at current level ' + str(self.current_level))
        # print(x.shape)

        if self.splitting:
            (x_even, x_odd) = self.split(x)
            # print(x_even.shape, x_odd.shape)


        else:
            (x_even, x_odd) = x

        if self.modified:       #if interative learning

            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)


            d = x_odd.mul(torch.exp(self.TCN1(x_even)))      #odd_brand output
            c = x_even.mul(torch.exp(self.TCN2(x_odd)))     #even_brand output

            if self.operation ==('plus','sub'):
                x_even_update = c + self.TCN3(d)
                x_odd_update = d - self.TCN4(c)
                return (x_even_update, x_odd_update)
            elif self.operation ==('plus','plus'):
                x_even_update = c + self.TCN3(d)
                x_odd_update = d + self.TCN4(c)
                return (x_even_update, x_odd_update)
            elif self.operation ==('sub','plus'):
                x_even_update = c - self.TCN3(d)
                x_odd_update = d + self.TCN4(c)
                return (x_even_update, x_odd_update)
            elif self.operation ==('sub','sub'):
                x_even_update = c - self.TCN3(d)
                x_odd_update = d - self.TCN4(c)
                return (x_even_update, x_odd_update)

            # print(x_even_update.shape , x_odd_update.shape)



        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self,
                 input_dim,
                 operation,
                 activation,
                 input_len,
                 current_level,
                 channels,
                 dropout,
                 INN,
                 tcn_ks_list):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(input_dim=input_dim,
                                operation = operation,
                                activation = activation,
                                input_len = input_len,
                                current_level = current_level,
                                splitting=True,
                                channels=channels,
                                dropout=dropout,
                                INN=INN,
                                tcn_ks_list=tcn_ks_list)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)


class STI_Block(nn.Module):
    def __init__(self,
                 input_dim,
                 operation,
                 activation,
                 input_len,
                 channels,
                 dropout,
                 current_level,
                 INN,
                 tcn_ks_list):
        super(STI_Block, self).__init__()
        self.interact = InteractorLevel(input_dim=input_dim,
                                        operation = operation,
                                        activation = activation,
                                        input_len = input_len,
                                        channels=channels,
                                        dropout=dropout,
                                        INN=INN,
                                        current_level=current_level,
                                        tcn_ks_list=tcn_ks_list)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)  # even: B, T, D odd: B, T, D


class STINet_Tree(nn.Module):
    def __init__(self,
                 input_dim,
                 operation,
                 activation,
                 input_len,
                 current_level,
                 channels,
                 dropout,
                 INN,
                 tcn_ks_list):
        super().__init__()
        self.current_level = current_level

        self.block = STI_Block(
            operation = operation,
            activation = activation,
            input_dim=input_dim,
            input_len = input_len,
            channels=channels,
            dropout=dropout,
            INN=INN,
            current_level= current_level,
            tcn_ks_list=tcn_ks_list)

        if current_level != 0:
            self.STINet_Tree_odd = STINet_Tree( input_dim=input_dim,
                                                input_len=int(input_len/2),
                                                current_level=current_level-1,
                                                channels=channels,
                                                dropout=dropout,
                                                INN=INN,
                                                activation=activation,
                                                operation = operation,
                                                tcn_ks_list=tcn_ks_list)

            self.STINet_Tree_even = STINet_Tree( input_dim=input_dim,
                                                input_len=int(input_len/2),
                                                current_level=current_level-1,
                                                channels=channels,
                                                dropout=dropout,
                                                INN=INN,
                                                 activation=activation,
                                                 operation = operation,
                                                 tcn_ks_list=tcn_ks_list)


    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)  # L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len:
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_, 0).permute(1, 0, 2)  # B, L, D

    def forward(self, x):

        x_even_update, x_odd_update = self.block(x)
        # We recursively reordered these sub-series. You can run the ./utils/recursive_demo.py to emulate this procedure.
        if self.current_level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.STINet_Tree_even(x_even_update), self.STINet_Tree_odd(x_odd_update))


class STINet(nn.Module):
    def __init__(self,
                 input_dim,
                 operation,
                 activation,
                 input_len,
                 num_levels,
                 channels,
                 dropout,
                 INN,
                 tcn_ks_list):
        super().__init__()
        self.levels = num_levels                # total of level in each tree
        self.STINet_Tree = STINet_Tree(         # STI tree
            operation = operation,
            activation = activation,
            input_dim=input_dim,
            input_len = input_len,
            current_level=num_levels - 1,
            channels=channels,
            dropout=dropout,
            INN=INN,
            tcn_ks_list=tcn_ks_list)

    def forward(self, x):
        x = self.STINet_Tree(x)

        return x






class StackedSTINet(nn.Module):
    def __init__(self,
                 output_len,
                 input_len,
                 channels,
                 input_dim,
                 output_dim,
                 operation,
                 tcn_ks_list,
                 num_stacks=1,
                 num_levels=4,
                 dropout=0.5,
                 modified=True,
                 activation ='relu',
                 decoder_include = True,
                 ):
        super(StackedSTINet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.input_len = input_len
        self.output_len = output_len
        self.num_levels = num_levels
        self.modified = modified
        self.channels = channels
        self.dropout = dropout
        self.operation = operation
        self.decoder_include = decoder_include
        self.tcn_ks_list = tcn_ks_list

        self.STI_1 = STINet(
            operation = operation,
            activation = activation,
            input_dim=self.input_dim,
            num_levels=self.num_levels,
            channels=self.channels,
            dropout=self.dropout,
            INN=modified,
            input_len=input_len,
            tcn_ks_list=self.tcn_ks_list)


        self.linear = CustomLinear(input_shape=(self.input_len, self.input_dim),
                                   output_shape=(self.output_len, self.output_dim))


    def forward(self, x):

        #Check input length should evenly divided for 2^level
        assert self.input_len % (np.power(2,self.num_levels)) == 0  # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)

        # the first stack
        res1 = x
        x = self.STI_1(x)
        x += res1

        #define decoder
        if self.decoder_include ==True:
            # print('prediction')
            x = self.linear(x)
            return x
        elif self.decoder_include == False:
            # print('encoder')
            return x
        # print(x.shape)

def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--channels', type=list, default=[10,20,30])
    parser.add_argument('--operation', type=tuple, default=('plus', 'sub'))

    parser.add_argument('--hidden-size', default=1, type=int, help='hidden channel of module')
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--single_step_output_One', type=int, default=0)

    args = parser.parse_args()

    model = StackedSTINet(output_len = args.window_size,
                          input_len = args.window_size,
                          input_dim=9,
                          num_stacks=1,
                          num_levels=3,
                          channels=args.channels,
                          dropout=args.dropout,
                          modified=True,
                          operation = args.operation,
                          output_dim=9,
                          tcn_ks_list=[2,3]).cuda()
    x = torch.randn(32, 128, 9).cuda()
    y = model(x)
    print(y.shape)