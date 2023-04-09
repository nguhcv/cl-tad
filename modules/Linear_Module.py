import torch.nn as nn
import torch
import math

class Linear_Transform(nn.Module):
    def __init__(self, input_shape:tuple, output_shape:tuple):
        super().__init__()
        # print(output_shape, input_shape)
        weights = torch.Tensor(output_shape[1], input_shape[1])
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(output_shape[0], output_shape[1])
        self.bias = nn.Parameter(bias)

        # nn.init.kaiming_uniform_(self.weights,  a=math.sqrt(5)) # weight init
        nn.init.normal_(self.weights, mean=0.0, std=0.01)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        w_times_x= torch.matmul(x, self.weights)
        return torch.add(w_times_x, self.bias)  # w times x + b



if __name__ == '__main__':
    pass
