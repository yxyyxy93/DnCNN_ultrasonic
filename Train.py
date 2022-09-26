import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss
import time
import copy
import os
import sys
from time import sleep
from tqdm import tqdm

from Dataset import UDataset
import Transformer

''' model '''


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, kernel_size=3, padding=1):
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

        # self._initialize_weights()

    def forward(self, noisy):
        noisy_img = noisy
        out = self.dncnn(noisy)
        return noisy_img - out  # return residual

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.orthogonal_(m.weight)
    #             print('init weight')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)


''' loss function - MSE'''


class sum_squared_error(_Loss):
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = sys.float_info.max
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                running_loss = 0.0
                # Iterate over data.
                for inputs, labels, img_mark in tepoch:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())
                    sleep(0.1)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())  #

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for inputs, labels, img_mark in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ''' visualization '''
                # Get a batch of training data
                ax = plt.subplot(num_images, 2, images_so_far)
                ax.imshow(outputs.cpu()[j])
                ax.set_title(img_mark[j])
                plt.show()

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    plt.interactive(False)
    cudnn.benchmark = True

    data_dir = r'./Dataset'

    ''' define transformer for augment '''
    transforms = [Transformer.RandomHorizontalFlip, Transformer.RandomVerticalFlip]
    frame_transform = Transformer.Compose([
        Transformer.RandomHorizontalFlip(p=0.5),
        Transformer.RandomVerticalFlip(p=0.5),
        Transformer.RandomRotation(degrees=180)
    ])

    image_datasets = {x: UDataset(os.path.join(data_dir, x), transforms=frame_transform)
                      for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=False, num_workers=4)
                   for x in ['train', 'val']}

    ''' visualization '''
    # Get a batch of training data
    inputs, origin, mark = next(iter(dataloaders['train']))
    fig, axs = plt.subplots(3, inputs.size(0))
    fig.suptitle(mark[1:3])
    for idx, img in enumerate(inputs):
        axs[0, idx].imshow(np.array(img)[0, :, :].squeeze(), cmap='gray')
        axs[1, idx].imshow(np.array(img)[1, :, :].squeeze(), cmap='gray')
        axs[2, idx].imshow(np.array(img)[2, :, :].squeeze(), cmap='gray')
    fig, axs = plt.subplots(3, origin.size(0))
    fig.suptitle('origin')
    for idx, img in enumerate(origin):
        axs[0, idx].imshow(np.array(img)[0, :, :].squeeze(), cmap='gray')
        axs[1, idx].imshow(np.array(img)[1, :, :].squeeze(), cmap='gray')
        axs[2, idx].imshow(np.array(img)[2, :, :].squeeze(), cmap='gray')
    plt.show()

    # # model selection
    # print('===> Building model')
    # model = DnCNN()
    # model.to(device)
    # # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    # criterion = sum_squared_error()
    #
    # # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model.parameters(), lr=1e-6, momentum=0.6)
    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    #
    # model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
    #                        num_epochs=25)
    #
    # ''' save the model '''
    # torch.save(model.state_dict(), 'model_weights.pth')
    #
    # ''' after training '''
    # # visualize_model(model_ft)
