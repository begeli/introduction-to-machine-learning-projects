import torch.optim
import torch.utils.data
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable


#################################TRAIN###############################

#Function to train the net
def train(net, criterion, optimizer, start_epoch, epochs, is_gpu,trainloader,valloader,train_batch_size,val_batch_size):
    """
    Training process.
    Args:
        net: Triplet Net
        criterion: TripletMarginLoss
        optimizer: SGD with Nesterov Momentum
        trainloader: training set loader
        valloader: validation set loader
        start_epoch: 0
        epochs: number of training epochs
        is_gpu: True since we train on GPU
    """
    print("==> Start training ...")

    val_loss_list = []
    criterion_val = nn.TripletMarginLoss(margin=0.0, p=2, reduction='none')
    net.train()
    counter = 0
    ##################################FOR TRAINING##############################
    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        loss_train = 0.0
        for batch_idx, (data1, data2, data3) in enumerate(trainloader):

            if is_gpu:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

            # wrap in torch.autograd.Variable
            data1, data2, data3 = Variable(
                data1), Variable(data2), Variable(data3)

            # compute output and loss
            embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
            loss = criterion(embedded_a, embedded_p, embedded_n)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print the loss
            running_loss += loss.data

            loss_train_cls = torch.sum(
                1 * (criterion_val(embedded_a, embedded_p,
                                   embedded_n) > 0)) / train_batch_size  # CHANGED, MAY NEED TO REVERT BACK

            loss_train += loss_train_cls.data

            if batch_idx % 30 == 0:
                print("mini Batch Loss: {}".format(loss.data))


        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        loss_train /= len(trainloader)

        print("Training Epoch: {0} | Loss: {1}".format(epoch + 1, running_loss))

        print("Training Epoch: {0} | Classification Loss: {1}".format(epoch + 1, loss_train))

        ##################################FOR VALIDATION##############################
        net.eval()
        val_loss = 0.0
        for batch_idx, (data1, data2, data3) in enumerate(valloader):
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

            # wrap in torch.autograd.Variable
            data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

            with torch.no_grad():
                # compute output and loss
                embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
                loss_val = torch.sum(
                    1 * (criterion_val(embedded_a, embedded_p, embedded_n) > 0))/val_batch_size # CHANGED, MAY NEED TO REVERT BACK
                val_loss += loss_val.data

        val_loss /= len(valloader)

        val_loss_list.append(val_loss)

        print("Validation Epoch: {0} | Loss: {1}".format(epoch + 1, val_loss))

        counter += 1

        net.train()

    print('==> Finished Training ...')
    net.eval()

    #Return to trained net
    return net