from modules.TCN_Module import Stacked_TCN
from modules.Linear_Module import Linear_Transform
from modules.sti_net import StackedSTINet

def net_generator(r_module,
                  t_module,
                  e_module,
                  n_dims,
                  r_channel_list,
                  e_channel_list,
                  dropout,
                  activation,
                  window_size,
                  r_kernel_size_list,
                  e_kernel_size_list,
                  sti_operations):
    net=[]
    # reconstruction model
    if r_module == 1:
        R = Stacked_TCN(n_dims=n_dims, num_channels=r_channel_list, dropout=dropout, activation=activation, window_size=window_size,ks_list=r_kernel_size_list)
        net.append(R)
    elif r_module == 2:
        R =  StackedSTINet(output_len=window_size,
                           input_len=window_size,
                           channels=r_channel_list,
                           input_dim=n_dims,output_dim=n_dims,operation=sti_operations,
                           tcn_ks_list=r_kernel_size_list,
                           num_levels=4,
                           dropout=dropout,
                           activation=activation,
                           decoder_include=True)
        net.append(R)


    #transformation layer
    # if t_module ==2:
    #     T = Stacked_TCN(n_dims=n_dims, num_channels=opt.tcn_channel_list, dropout=0.5, activation=opt.activation, window_size=opt.w,ks_list=opt.t_ks_list)
    #     net.append(T)
    if t_module == 1:
        T = Linear_Transform(input_shape=(n_dims,window_size), output_shape=(n_dims,window_size))
        net.append(T)
    else:
        raise ValueError ('we now only support linear layer')


    #encoder layer
    if e_module ==1:
        E = Stacked_TCN(n_dims=n_dims, num_channels=e_channel_list, dropout=dropout,
                    activation=activation, window_size=window_size, ks_list=e_kernel_size_list)
        net.append(E)
    elif e_module == 2:
        E = StackedSTINet(output_len=window_size,
                           input_len=window_size,
                           channels=e_channel_list,
                           input_dim=n_dims,output_dim=n_dims,operation=sti_operations,
                           tcn_ks_list=e_kernel_size_list,
                           num_levels=4,
                           dropout=dropout,
                           activation=activation,
                           decoder_include=False)
        net.append(E)


    return net

    pass
