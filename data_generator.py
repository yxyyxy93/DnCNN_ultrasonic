import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
import os

patch_size, stride = 100, 10
batch_size = 64


class DenoisingDataset_dir(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """

    def __init__(self, root, transforms, sigma):
        super(DenoisingDataset_dir, self).__init__()
        self.root = root
        self.transforms = transforms
        self.sigma = sigma
        image_dir = os.path.join(self.root, '')
        image_filenames = list(os.listdir(image_dir))
        self.image_path = [os.path.join(image_dir, file_name) for file_name in image_filenames if
                           os.path.splitext(file_name)[1] == '.png']

    def __getitem__(self, idx):
        patches = gen_patches(self.image_path[idx])

        # converting list to array
        batch_x = np.array(patches)
        # the batch size cannot be divived by the batch number !!

        # add Gaussian noise
        noise = torch.randn(batch_x.size()).mul_(self.sigma / 255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return len(self.image_path)


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """

    def __init__(self, xs, sigma, transforms):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma
        self.transforms = transforms

    def __getitem__(self, index):
        batch_x = self.xs[index]
        # add Gaussian noise
        noise = torch.randn(batch_x.size()).mul_(self.sigma / 255.0)
        batch_y = batch_x + noise

        if self.transforms is not None:
            batch_y, batch_x = self.transforms(batch_y, batch_x)

        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def gen_patches(file_name):
    # get multiscale patches from a single image
    # img = cv2.imread(file_name, 0)  # gray scale
    img = np.load(file_name)  # read .npy
    h, w = img.shape
    patches = []
    h_scaled, w_scaled = int(h), int(w)
    img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
    # extract patches
    for i in range(0, h_scaled - patch_size + 1, stride):
        for j in range(0, w_scaled - patch_size + 1, stride):
            x = img_scaled[i:i + patch_size, j:j + patch_size]
            patches.append(x)
    return patches


def datagenerator(data_dir='data/Train400', verbose=False):
    # generate clean patches from a dataset
    # file_list = glob.glob(data_dir + '/*.png')  # get name list of all .png files
    file_list = glob.glob(data_dir + '/*.npy')  # get name list of all .npy files
    # initialize
    pre_data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:
            pre_data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done')
    pre_data = np.array(pre_data)
    pre_data = np.expand_dims(pre_data, axis=3)
    discard_n = len(pre_data) - len(pre_data) // batch_size * batch_size  # batch intergration
    pre_data = np.delete(pre_data, range(discard_n), axis=0)
    print('training data finished')
    return pre_data


if __name__ == '__main__':
    # data = datagenerator(data_dir='dataset/train/Train400')

    # # Get a batch of training data
    # datadir_train = 'dataset/train/Train400'
    # image_dataset = DenoisingDataset_dir(datadir_train, None, sigma=25)
    # dataloader = torch.utils.data.DataLoader(image_dataset,
    #                                          batch_size=4,
    #                                          shuffle=False)
    # inputs, origin, mark = next(iter(dataloader))
    #
    print('Done.')
