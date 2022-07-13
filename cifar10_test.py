import os
from scipy.fft import dst
import torch.nn as nn
import argparse
import numpy as np
import deepcore.nets as nets
import deepcore.datasets as datasets
# import deepcore.methods as methods
from torchvision import transforms
from utils import *
from datetime import datetime
from time import sleep

## this is for cifar 10

def train(train_loader, args,  model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    model.train()

    print('\n=> Training Epoch #%d' % epoch)

    

    
    for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # Forward propagation, compute loss, get predictions
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update loss, backward propagate, update optimizer
            optimizer.zero_grad()
            # loss = loss.mean()  ## no need. Same value before & after mean

            # if i % args.print_freq == 0:
            #     print('| Epoch [%3d/%3d] \t\tLoss: %.4f' % (
            #     epoch, args.epochs, loss.item()))


            loss.backward()
            optimizer.step()
            

def test(test_loader, args, model, criterion, epoch):

    model.eval()


    # print('\n=> Testing Epoch #%d' % epoch)
    
    correct = 0.
    total = 0.

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):
            input, target = input.to(args.device), target.to(args.device)
            output = model(input)
            loss = criterion(output, target).sum()

            max_pred, predicted = torch.max(output.data, dim=1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            # if batch_idx % args.print_freq == 0:
            #     print('| Test Epoch [%3d/%3d] \t\tTest Loss: %.4f Test Acc: %.3f%%' % (
            #         epoch, args.epochs,loss.item(), 100. * correct / total))

        return 100. * correct / total


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--model', type=str, default='LeNet', help='model')
    # parser.add_argument('--selection', type=str, default="uniform", help="selection method")
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('--print_freq', '-p', default=20, type=int, help='print frequency (default: 20)')
    # parser.add_argument('--fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # parser.add_argument("--cross", type=str, nargs="+", default=None, help="models for cross-architecture experiments")

    # Optimizer and scheduler
    parser.add_argument('--optimizer', default="Adam", help='optimizer to use, e.g. SGD, Adam')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument("--nesterov", default=True, type=str_to_bool, help="if set nesterov")
    parser.add_argument("--scheduler", default="StepLR", type=str, help=
    "Learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
    parser.add_argument("--step_size", type=float, default=5, help="Step size for StepLR")

    parser.add_argument('--batch', '--batch-size', "-b", default=128, type=int, metavar='N',
                        help='mini-batch size (default: 128)')
    parser.add_argument("--train_batch", "-tb", default=None, type=int,
                     help="batch size for training, if not specified, it will equal to batch size in argument --batch")


    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_batch is None:
        args.train_batch = args.batch

    print("dataset: ", args.dataset, ", model: ", args.model,\
              ", epochs: ", args.epochs, ", seed: ", args.seed,\
                ", optimizer: ", args.optimizer,\
              ", lr: ", args.lr, ", device: ", args.device)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
            (args.data_path)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

    torch.random.manual_seed(args.seed)


    ## train_loader

    train_loader = torch.utils.data.DataLoader(dst_train, shuffle=True, batch_size=args.batch,
                                                   num_workers=args.workers, pin_memory=True)

    ## validation_loader (test, for now)

    test_loader = torch.utils.data.DataLoader(dst_test , batch_size=args.batch, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)

    model = nets.__dict__[args.model](args.channel, num_classes, im_size=im_size).to(args.device)


    if args.device == "cpu":
        print("Using CPU.")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu[0])
        model = nets.nets_utils.MyDataParallel(model, device_ids=args.gpu)
    elif torch.cuda.device_count() > 1:
        model = nets.nets_utils.MyDataParallel(model).cuda()


    criterion = nn.CrossEntropyLoss().to(args.device)
    criterion.__init__()

    # Setup optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                                   momentum=args.momentum,
                                                   weight_decay=args.weight_decay,
                                                   nesterov=args.nesterov)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
        #lr=args.lr,
                                                    # weight_decay=args.weight_decay,
                                                    )
    else:
        optimizer = torch.optim.__dict__[args.optimizer](model.parameters(),
                                                                       lr=args.lr,
                                                                       momentum=args.momentum,
                                                                       weight_decay=args.weight_decay,
                                                                       nesterov=args.nesterov)

    # if args.scheduler == "StepLR":
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # else:
    #     raise NotImplementedError


    for epoch in range(args.epochs):
        

        print("current LR: {:.6f}".format(optimizer.param_groups[0]['lr']))

        train(train_loader, args, model, criterion, optimizer, epoch)
        # scheduler.step()


        print('Val accuracy: %0.2f , Train accuracy : %0.2f'%(test(test_loader, args, model, criterion, epoch),\
            test(train_loader, args, model, criterion, epoch)))

                                        


if __name__ == "__main__":
    main()