import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset
import numpy as np
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
from datetime import datetime

from Dataset import UDataset, UDataset_snr

''' model '''


class DnCNN(nn.Module):
    def __init__(self, depth=20, n_channels=64, image_channels=1, kernel_size=3, padding=1):
        super(DnCNN, self).__init__()

        layers = [
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True), nn.ReLU(inplace=True)
        ]
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False)
            )
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False)
        )

        self.dncnn = nn.Sequential(*layers)

    def forward(self, noisy):
        noisy_img = noisy
        out = self.dncnn(noisy)
        return noisy_img - out  # return residual


class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)

        return torch.nn.functional.mse_loss(input, target, reduction='sum').div_(2)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, k_folds=5, batchsize=64):
    since = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = sys.float_info.max

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(2022)

    # Start print
    print('--------------------------------')
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

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
            dataset,
            batch_size=batchsize,
            sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            sampler=test_subsampler)

        # ''' visualization '''
        # # Get a batch of training data
        # inputs, origin, mark = next(iter(trainloader))
        # fig, axs = plt.subplots(1, inputs.size(0))
        # fig.suptitle(mark[0:2])
        # for idx, img in enumerate(inputs):
        #     axs[idx].imshow(np.array(img).squeeze(), cmap='gray')
        # fig, axs = plt.subplots(1, origin.size(0))
        # fig.suptitle('origin')
        # for idx, img in enumerate(origin):
        #     axs[idx].imshow(np.array(img).squeeze(), cmap='gray')
        # plt.show()

        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            model.train()  # Set model to training mode
            with tqdm(trainloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                running_loss = 0.0
                # Iterate over data.
                cnt_batch = 0
                for inputs, labels, img_mark in tepoch:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # torch.set_grad_enabled(True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / 2
                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()
                    # statistics
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())
                    sleep(0.1)
                    scheduler.step()
                    cnt_batch = cnt_batch + 1
                    # last_loss = running_loss / batchsize  # loss per batch

                avg_loss = running_loss / trainloader.__len__()
                loss_train_batches.append(avg_loss)

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'avg_train_loss: {avg_loss}')
            # # Print about testing
            # with torch.no_grad():
            #     running_loss = 0.0
            #     # Iterate over data.
            #     for inputs, labels, img_mark in tepoch:
            #         inputs = inputs.to(device)
            #         labels = labels.to(device)
            #         # forward
            #         torch.set_grad_enabled(True)
            #         outputs = model(inputs)
            #         loss = criterion(outputs, labels) / 2
            #         running_loss += loss.item()
            #     avg_vloss = running_loss / testloader.__len__()
            #     time_elapsed = time.time() - since
            #     print(f'Validating complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            #     print(f'avg_val_loss: {avg_vloss}')
            #     loss_val_batches.append(avg_vloss)
            #
            # # Track best performance, and save the model's state
            # if avg_vloss < best_loss:
            #     best_loss = avg_vloss
            #     best_model_wts = copy.deepcopy(model.state_dict())  #

        # results[fold] = avg_vloss

        # Saving the model
        save_path = f'./model_weights/model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)

        # save the curves
        loss_train_val = zip(loss_train_batches, loss_val_batches)
        with open('my_file_' + str(fold) + '_fold.csv', 'w') as my_file:
            for (loss_train_batches, loss_val_batches) in loss_train_val:
                my_file.write("{0},{1}\n".format(loss_train_batches, loss_val_batches))
        print('File created')

        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')

        sum_loss = 0.0
        for key, value in results.items():
            print(f'Fold {key} loss: {value}')
            sum_loss += value
        print(f'Average loss: {sum_loss / len(results.items())}')

        # # Track best performance, and save the model's state
        # if avg_vloss < best_loss:
        #     best_loss = avg_vloss
        #     best_model_wts = copy.deepcopy(model.state_dict())  #

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    plt.interactive(False)
    cudnn.benchmark = True

    ''' define transformer for augment '''
    transforms = [Transformer.RandomHorizontalFlip, Transformer.RandomVerticalFlip]
    frame_transform = Transformer.Compose([
        Transformer.RandomHorizontalFlip(p=0.5),
        Transformer.RandomVerticalFlip(p=0.5),
        Transformer.RandomRotation(degrees=180)
    ])

    data_dir = r'./Dataset'
    sigma = 20
    # image_datasets = {x: UDataset_snr(os.path.join(data_dir, x),
    #                                   transforms=frame_transform,
    #                                   sigma=sigma)
    #                   for x in ['train', 'val']}
    # # prepare and merge the train and val dataset
    # dataset = ConcatDataset([image_datasets['train'], image_datasets['val']])
    dataset = UDataset_snr(os.path.join(data_dir, 'train'),
                           transforms=None,
                           sigma=sigma)

    # model selection
    print('===> Building model')
    model = DnCNN()
    model.to(device)

    criterion = nn.MSELoss(reduction='sum')
    # criterion = sum_squared_error()

    # Observe that all parameters are being optimized
    # optimizer_tf = optim.Adam(model.parameters(), lr=1e-2)
    params = [p for p in model.parameters() if p.requires_grad]  # param to update
    optimizer_tf = optim.SGD(params,
                             lr=1e-2,
                             weight_decay=0.0005,
                             momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_tf,
                                           step_size=7,
                                           gamma=0.1)

    model_ft = train_model(model,
                           criterion,
                           optimizer_tf,
                           exp_lr_scheduler,
                           batchsize=16,
                           num_epochs=50)

    ''' save the model '''
    torch.save(model.state_dict(), 'DnCNNmodel_weights.pth')
