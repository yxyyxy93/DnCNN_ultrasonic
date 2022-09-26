import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset
import numpy as np
import re
import matplotlib.pyplot as plt
import torch.nn.init as init
from torch.nn.modules.loss import _Loss
import time
import copy
from torchvision import datasets, models, transforms
import os
import torch.optim as optim
import sys
from time import sleep
from tqdm import tqdm
from sklearn.model_selection import KFold
import Transformer
import datetime
import glob
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from Dataset import UDataset, UDataset_snr

import data_generator as dg
from data_generator import DenoisingDataset

''' model '''


class DnCNN(nn.Module):
    def __init__(self, depth=20, n_channels=64, image_channels=1):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            nn.Conv2d(in_channels=image_channels,
                      out_channels=n_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels,
                          out_channels=n_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels,
                                         eps=0.0001,
                                         momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels,
                      out_channels=image_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        # self._initialize_weights()

    def forward(self, img):
        img_noisy = img
        noise = self.dncnn(img)
        return img_noisy - noise  # return residual


class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def train_model(model, datadir_train, criterion, optimizer, scheduler, n_epoch=25, k_folds=5, batch_size=64):
    # generate dataset
    # xs = dg.datagenerator(data_dir=os.path.join(datadir_train, 'origin_images_label'))
    # xs = xs.astype('float32') / 255.0
    # xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
    # DDataset = DenoisingDataset(xs,
    #                             sigma,
    #                             transforms=frame_transform)

    DDataset = UDataset(datadir_train,
                        transforms=frame_transform)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds,
                  shuffle=True)
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(2022)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(DDataset)):
        loss_train_batches = []
        loss_val_batches = []
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            DDataset,
            batch_size=batch_size,
            sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            DDataset,
            batch_size=batch_size,
            sampler=test_subsampler)

        for epoch in range(0, n_epoch):
            epoch_loss = 0
            start_time = time.time()

            # train
            for n_count, batch_yx in enumerate(trainloader):
                optimizer.zero_grad()
                # forward
                torch.set_grad_enabled(True)


                batch_x = batch_yx[1].to(device)
                batch_y = batch_yx[0].to(device)
                loss = criterion(model(batch_y), batch_x) / 2
                # loss = criterion(model(batch_yx[0]), batch_yx[1]) / 2
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                if n_count % 5 == 0:
                    print('%4d %4d / %4d loss = %2.4f' % (
                        epoch + 1, n_count, train_ids.size // batch_size, loss.item() / batch_size))

            scheduler.step()  # step to the learning rate in this epcoh
            elapsed_time = time.time() - start_time
            log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))

            # validation
            with torch.no_grad():
                for n_count, batch_yx in enumerate(testloader):
                    batch_x = batch_yx[1].to(device)
                    batch_y = batch_yx[0].to(device)
                    loss = criterion(model(batch_y), batch_x) / 2
                    epoch_loss += loss.item()
                    if n_count % 5 == 0:
                        print('%4d %4d / %4d val_loss = %2.4f' % (
                            epoch + 1, n_count, test_ids.size // batch_size, loss.item() / batch_size))

            np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
            # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
            torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))

    return model


if __name__ == '__main__':
    batch_size = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_epoch = 50
    sigma = 25
    learning_rate = 1e-2

    save_dir = os.path.join('models', 'DnCNN_' + 'sigma' + str(sigma))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # model selection
    print('===> Building model')

    model = DnCNN()
    model.train()
    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]  # param to update

    # optimizer = optim.Adam(params, lr=learning_rate)
    # optimizer = optim.NAdam(params, lr=learning_rate)
    optimizer = optim.SGD(params,
                          lr=1e-6,
                          weight_decay=0.0005,
                          momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,
                                           step_size=7,
                                           gamma=0.1)
    # scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    criterion = nn.MSELoss(reduction='sum')  # PyTorch 0.4.1

    datadir_train = r'./Dataset/train'

    ''' define transformer for augment '''
    frame_transform = Transformer.Compose([
        Transformer.RandomHorizontalFlip(p=0.5),
        Transformer.RandomVerticalFlip(p=0.5),
        Transformer.RandomRotation(degrees=180)
    ])

    model = train_model(model,
                        datadir_train,
                        criterion,
                        optimizer,
                        exp_lr_scheduler,
                        n_epoch=n_epoch,
                        k_folds=5,
                        batch_size=batch_size)

    ''' save the model '''
    torch.save(model.state_dict(), 'DnCNNmodel_weights.pth')
