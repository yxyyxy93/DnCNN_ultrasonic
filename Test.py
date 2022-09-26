import matplotlib.pyplot as plt
from Train_KFold_v2 import DnCNN
import torch
from Dataset import UDataset
import numpy as np


def visualize_model(model, idx=1):
    model.eval()
    images_so_far = 0

    inputs, labels, img_mark = image_datasets.__getitem__(idx)
    inputs = torch.as_tensor(np.expand_dims(inputs, axis=0)).to(device)
    outputs = model(inputs)
    inputs = inputs.cpu()
    outputs = outputs.cpu()
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)

    ''' visualization '''
    z0 = axs[0].imshow(np.array(inputs).squeeze())
    axs[0].set_title('layer' + img_mark[2] + '_' + img_mark[6] + ' noisy')
    # plt.colorbar(z0, cax=axs[0])
    z1 = axs[1].imshow(outputs.detach().numpy().squeeze())
    axs[1].set_title('layer' + img_mark[2] + '_' + img_mark[6] + ' DnCNN')
    # plt.colorbar(z1, cax=axs[1])
    z2 = axs[2].imshow(outputs.detach().numpy().squeeze() - np.array(labels).squeeze())
    axs[2].set_title('layer' + img_mark[2] + '_' + img_mark[6] + ' learned residual')
    # plt.colorbar(z2, cax=axs[2])
    z3 = axs[3].imshow(labels.squeeze())
    axs[3].set_title('layer' + img_mark[2] + '_' + img_mark[6] + ' origin')
    # plt.colorbar(z3, cax=axs[3])


    plt.show()
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mode_dir = r'./models/DnCNN_sigma25/model_030.pth'
model = torch.load(mode_dir)
model.eval()

data_dir = r'./Dataset/train'
image_datasets = UDataset(data_dir, None)
idx = 10

visualize_model(model, idx=idx)
