"""
project: hand xray dataset kaggle
created with anaconda venv: "boneage_estimation" (infos in yml file)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import argparse
import os
import time
import xray_dataset
import models
import numpy as np
import utilities


def parse_arguments():

    print('\nParse arguments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str, default=None,
                        help='root directory of dataset')
    parser.add_argument('--dir_cropping_annotations', type=str, default=None,
                        help='directory with files that contain cropping area for each image')
    parser.add_argument('--lr_ann_file', type=str, default=None,
                        help='csv file with image id of right hands (if not in file, '
                             'images are assumed to be of left hand)')
    parser.add_argument('--train', type=str, choices=['Train','Test'],
                        help='Train or Test dataset')
    parser.add_argument('--architecture', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                                                             'resne152'],
                        help='Choose network architecture')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of label classes')
    parser.add_argument('--task', type=str, choices=['age', 'gender', 'leftorright'], default='leftorright',
                        help='select task: age, gender, leftorright')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='ratio of the dataset used for training')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batchsize for dataloader')
    parser.add_argument('--N_max', type=int, default=100,
                        help='max number of images used from dataset')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--dir_checkpoints', type=str, default=None,
                        help='directory of checkpoints')
    parser.add_argument('--dir_epochs', type=str, default=None,
                        help='directory of epochs')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')

    arguments = parser.parse_args()

    assert arguments.dir_data is not None

    if arguments.train == 'Train':
        arguments.train = True
    else:
        arguments.train = False

    for keys, values in arguments.__dict__.items():
        print('\t{}: {}'.format(keys, values))

    return arguments


def initialize_model(args):

    print('\nInitialize model')

    model = models.__dict__.get(args.architecture)(num_out_channels=args.num_classes)

    model = torch.nn.DataParallel(model).cuda()

    print('\tnum_params: {}\n'.format(models.get_num_params(model)))

    return model


def initialize_data(args, phase='train'):

    if phase == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.2])
        ])
        shuffle = True

        if args.dir_cropping_annotations:
            list_dataset = xray_dataset.XRAYBONES(root=args.dir_data, dir_cropping_info=args.dir_cropping_annotations,
                                                  train=True, split=args.split_ratio, task=args.task,
                                                  transform=transform)
        else:
            list_dataset = xray_dataset.XRAYBONES(root=args.dir_data, lr_ann_file_path=args.lr_ann_file,
                                                  train=True, Nmax=args.N_max, split=args.split_ratio,
                                                  task=args.task, transform=transform)

    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.2])
        ])
        shuffle = False

        if args.dir_cropping_annotations:
            list_dataset = xray_dataset.XRAYBONES(root=args.dir_data, dir_cropping_info=args.dir_cropping_annotations,
                                                  train=False, split=args.split_ratio, task=args.task,
                                                  transform=transform)
        else:
            list_dataset = xray_dataset.XRAYBONES(root=args.dir_data, lr_ann_file_path=args.lr_ann_file,
                                                  train=False, Nmax=args.N_max, split=args.split_ratio,
                                                  task=args.task, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=list_dataset, batch_size=args.batch_size, shuffle=shuffle)

    return data_loader


def train_epoch(epoch, data_loader):

    print('\nEpoch: %d' % epoch)
    model.train()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.task == "age":
            targets = targets.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def val_epoch(epoch, data_loader):

    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    loss_vec = np.ndarray((len(val_loader), 1))
    targets_vec = np.ndarray((len(val_loader), 1))
    outputs_vec = np.ndarray((len(val_loader), 1))
    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.task == "age":
                targets = targets.float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if args.task == "leftorright" or args.task == "gender":
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            elif args.task == "age":
                loss_vec[batch_idx] = loss.item()  # assumes batchsize is 1
                targets_vec[batch_idx] = targets.item()
                outputs_vec[batch_idx] = outputs.item()

    val_table = np.hstack([targets_vec, outputs_vec, loss_vec])

    '''display accuracy'''
    if args.task == "leftorright" or args.task == "gender":
        acc = 100. * correct / total
    elif args.task == "age":
        acc = np.mean(loss_vec)

    print("Result epoch ", epoch, ": accuracy = ", acc)
    utilities.pickle_result(val_table, args.dir_epochs + '/task_{}_{}_epoch_ [{}\{}].pickle'.format(args.task, args.architecture, epoch,
                                                                                           args.num_epochs))


    #print("validation table:", val_table)
    with open(logfile, 'a') as l:
        l.write("Nmax = {}: Result epoch {} = accuracy = {}\n".format(args.N_max, epoch, acc))

    # Save every epoch checkpoint.
    if True:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        checkpoint_dir = args.dir_checkpoints + '/checkpoint_epoch_{}'.format(epoch)
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(state, checkpoint_dir + '/ckpt.t7')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    args = parse_arguments()

    train_loader = initialize_data(args, phase='train')
    val_loader = initialize_data(args, phase='val')

    model = initialize_model(args)
    if args.task == "leftorright" or args.task == "gender":
        criterion = nn.CrossEntropyLoss()
    elif args.task == "age":
        criterion = nn.L1Loss()

    optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)

    start_epoch = 0
    logfile = "logfile.txt"

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        epoch_start_time = time.time()
        print("Training")
        train_epoch(epoch, train_loader)
        print("Evaluation")
        val_epoch(epoch, val_loader)

        epoch_finish_time = time.time()

        with open(logfile, "a") as l:
            l.write("epoch {} - duration (min): {} \n\n".format(epoch, (epoch_finish_time-epoch_start_time)/60))

        print("epoch duration (min): ", (epoch_finish_time-epoch_start_time)/60)


