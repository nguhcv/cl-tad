import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
class CustomLinear(nn.Module):
    def __init__(self, input_shape:tuple, output_shape:tuple):
        super().__init__()
        # print(output_shape, input_shape)
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
        w_times_x= torch.matmul(self.weights, x )
        return torch.add(w_times_x, self.bias)  # w times x + b

class Stacked_TCN(nn.Module):
    def __init__(self,
                 n_dims,
                 num_channels,
                 activation,
                 window_size,
                 ks_list,
                 dropout=0.,):
        super(Stacked_TCN, self).__init__()
        self.ks_list = ks_list
        nk_list = []
        self.dropout = dropout

        num_levels = len(num_channels)
        for k in range (len(ks_list)):
            brand=[]
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_channels = n_dims if i == 0 else num_channels[i - 1]
                out_channels = num_channels[i]
                brand += [TemporalBlock(n_inputs=in_channels, n_outputs=out_channels, kernel_size=self.ks_list[k], stride=1,
                                         dilation=dilation_size,
                                         padding=(self.ks_list[k]-1) * dilation_size, dropout=self.dropout, activation=activation)]
            nk_list.append(nn.Sequential(*brand))
        self.network_list = nn.ModuleList(nk_list[o] for o in range(len(self.ks_list)))

        self.linear = CustomLinear(input_shape=(len(self.ks_list) * num_channels[-1], window_size), output_shape=(n_dims, window_size))
        # self.tanh = nn.Tanh()

    def forward(self, x):
        output_list = []

        for k in range(len(self.ks_list)):
            op = self.network_list[k](x)
            output_list.append(op)


        if len(self.ks_list)==1:
            f_output = output_list[0]
            # f_output = self.tanh(f_output)
            f_output = self.linear(f_output)

            return f_output
            pass
        elif len(self.ks_list)==2:
            f_output = torch.cat((output_list[0], output_list[1]),dim=1)
            # f_output = self.tanh(f_output)
            f_output = self.linear(f_output)
            return f_output
            pass
        elif len(self.ks_list)==3:
            f_output = torch.cat((output_list[0], output_list[1],output_list[2]), dim=1)
            # f_output = self.tanh(f_output)
            f_output = self.linear(f_output)
            return f_output
        elif len(self.ks_list) == 4:
            f_output = torch.cat((output_list[0], output_list[1], output_list[2],output_list[3]), dim=1)
            # f_output = self.tanh(f_output)
            f_output = self.linear(f_output)
            return f_output
            pass









class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # print(x[0][0])
        # print(x[:, :, :-self.chomp_size].contiguous()[0][0])
        # print('-------------------')
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 activation,
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        if activation =='relu':
            self.act = nn.ReLU()
        elif activation == 'leak':
            self.act = nn.LeakyReLU()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)


        self.net = nn.Sequential(self.conv1, self.chomp1, self.act, self.dropout1,
                                 self.conv2, self.chomp2, self.act, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()



    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)
        # return self.relu(out + res)



class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, activation):
        super(TemporalConvNet, self).__init__()
        self.n_inputs = num_inputs
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, activation=activation)]

        self.network = nn.Sequential(*layers)
        self.flat = nn.Flatten()

    # def init_weights(self):
    #     # self.linear1.weight.data.normal_(0,0.01)
    #     # self.linear2.weight.data.normal_(0, 0.01)
    #     for i in range(self.n_inputs):
    #         getattr(self, "fc%d" % i).weight.data.normal_(0,0.01)

    def forward(self, x):
        x = self.network(x)

        return x



class Local_TCN(nn.Module):
    def __init__(self,
                 n_dims,
                 num_channels,
                 dropout,
                 activation,
                 window_size,
                 ks_list):
        super(Local_TCN, self).__init__()
        self.ks_list = ks_list
        nk_list = []

        num_levels = len(num_channels)


if __name__ == '__main__':
    pass
    x = torch.rand(size=(64, 5, 26)).cuda()  # 5:batch_size, 1:number channel, 20 time-length
    print(x.shape)

    net = Stacked_TCN(n_dims=5, num_channels=[64, 96, 128, 160], dropout=0.2,activation='relu', window_size=26,ks_list=[2,3]).cuda()
    output = net(x)
    print(output.shape)