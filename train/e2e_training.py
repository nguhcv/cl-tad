import torch
from utils.loss import ReconstructionLoss,OurContrastiveLoss
from utils.mask import sequence_order_position, masked_batch_generation_modified
from modules.sti_net import StackedSTINet
from utils.lars_wrapper import LARSWrapper
from utils.linearwarmup_cosineLR import LinearWarmupCosineAnnealingLR


def end_to_end_learning(model_name,
                         F,
                         train_loader,
                         masking_factor,
                         lr,
                         w,
                         save_path,
                         n_epochs,
                         cosine_option,
                         lars_option,):
    # 1. pre-training reconstruction model
    # Feed model to cuda
    F = F.cuda()

    optimizer = torch.optim.Adam(list(F.parameters()), lr=lr)

    if lars_option:
        optimizer = LARSWrapper(optimizer)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=15, max_epochs=n_epochs)



    ctloss_fn = OurContrastiveLoss(masking_factor=masking_factor, w_size=w,mode=3)
    rcloss_fn = ReconstructionLoss(window_size=w)


    # training model
    train_loss_list = []
    train_best_loss = 100000

    for e in range(n_epochs):
        print('---epoch ' + str(e) + '--------')
        train_loss_sum = 0.0

        for b_index, (data, label) in enumerate(train_loader):
            if (b_index + 1) % 100 == 0:
                print('batch number: ' + str(b_index))

            # for each batch, generate a masked_batch

            if masking_factor ==1:

                rand_pos_list = sequence_order_position(window_size=w, data_dimension=data.shape[1])

                # for each batch, generate a masked_batch

                generated_batch, generated_labels = masked_batch_generation_modified(data_batch=data,
                                                                                     random_pos=rand_pos_list,
                                                                                     window_size=w, mask_value=0.)
            else:
                raise ValueError ('now we only support masking ratio: 1')




            # reshape batch if reconstruction model is STI-net
            if type(F.R) is StackedSTINet:
                data = data.permute(0, 2, 1)
                # label = label.permute(0, 2, 1)
                generated_batch = generated_batch.permute(0, 2, 1)

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
            C = torch.unsqueeze(C, 1)

            # feed data to GPUs
            generated_batch = generated_batch.cuda()
            generated_labels = generated_labels.cuda()

            # forward generated-batch
            rc_output, features, uncertainty = F(generated_batch)

            ct_loss = ctloss_fn(features, uncertainty, data_features,C)
            rc_loss = rcloss_fn(rc_output, generated_labels)
            loss =ct_loss+rc_loss

            # calculate gradients
            loss.backward()

            # update weights
            optimizer.step()

            optimizer.zero_grad()
            train_loss_sum += ct_loss.item()

        if cosine_option:
            scheduler.step()

        print('n_epoch is: ' + str(e))
        print('train_loss_epoch is ' + str(train_loss_sum))
        train_loss_list.append(train_loss_sum)

        if train_loss_sum < train_best_loss:
            train_best_loss = train_loss_sum
            best_epoch = e
            print('New train_loss is: ' + str(train_best_loss))
            print('n_epoch is: ' + str(e))

            state_dict = {
                'epoch': best_epoch,
                'best_loss': train_best_loss,
                'model': F.state_dict(),
            }
            torch.save(state_dict, save_path + model_name )
            torch.save(train_loss_list, save_path + model_name + '_loss_list')

    torch.save(train_loss_list, save_path + model_name + '_loss_list')

