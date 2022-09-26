import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import hdf5storage


class UDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # Read the path of the image
        image_dir = os.path.join(self.root, 'noisy_images')
        image_filenames = list(os.listdir(image_dir))
        self.image_path = [os.path.join(image_dir, file_name) for file_name in image_filenames if
                           os.path.splitext(file_name)[1] == '.npy']
        # Read the path of labels
        label_dir = os.path.join(self.root, 'origin_images_label')
        self.label_filenames = list(os.listdir(label_dir))
        self.label_path = [os.path.join(label_dir, file_name) for file_name in self.label_filenames if
                           os.path.splitext(file_name)[1] == '.npy']

    def __getitem__(self, idx):
        # load image from mat file
        # image = hdf5storage.loadmat(self.image_path[idx])["img_batch"]
        image = np.load(self.image_path[idx])

        # load corresponding label from mat file
        image_mark = os.path.split(self.image_path[idx])[1].split('_')  # layer1.mat
        assert len(image_mark) > 3, f"The label {self.image_path[idx]} is incorrect."
        label_file_name = 'origin' + '_' + image_mark[1] + '_' + image_mark[2] + '__' + image_mark[6]
        assert label_file_name in self.label_filenames, f"The label {label_file_name} file does not exist."
        label_index = self.label_filenames.index(label_file_name)
        label_file = self.label_path[label_index]
        mask = np.load(label_file)
        # mask = image - mask  # residual - added noises

        # # expand one axis for 1 channel image
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # for 3 channels image
        # image = np.transpose(image, [2, 0, 1])
        # mask = np.transpose(mask, [2, 0, 1])

        # transform to torch data type
        image = torch.as_tensor(image, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask, image_mark

    def __len__(self):
        return len(self.image_path)


class UDataset_snr(Dataset):
    """
     generate the mask by adding noise to the original image directly
    """

    def __init__(self, root, sigma, transforms):
        self.root = root
        self.transforms = transforms
        self.sigma = sigma
        # Read the path of the image
        # image_dir = os.path.join(self.root, 'origin_images_label')
        image_dir = os.path.join(self.root, 'noisy_images')

        image_filenames = list(os.listdir(image_dir))
        self.image_path = [os.path.join(image_dir, file_name)
                           for file_name in image_filenames
                           if os.path.splitext(file_name)[1] == '.npy']

    def __getitem__(self, idx):
        # load image
        mask = np.load(self.image_path[idx])
        # mask = cv2.imread(self.image_path[idx], 0)
        # expand one axis for 1 channel image
        # image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        mask = mask.astype('float32')
        # mask = torch.from_numpy(mask)  # tensor of the clean patches,

        image_mark = os.path.split(self.image_path[idx])[1].split('_')  # layer1.mat

        # add Gaussian noise
        # noise = torch.randn(mask.size()).mul_(self.sigma / 255.0)
        # image = mask + noise

        noise = np.random.randn(mask.shape[1], mask.shape[2]) * self.sigma
        image = mask + noise
        # mask = image - mask  # residual - added noises

        # for 3 channels image
        # image = np.transpose(image, [2, 0, 1])
        # mask = np.transpose(mask, [2, 0, 1])

        # transform to torch data type
        image = torch.as_tensor(image, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask, image_mark

    def __len__(self):
        return len(self.image_path)

# # Testing
# root = r'./Dataset/train'
# mydataset = UDataset(root, None)
# data_loader = torch.utils.data.DataLoader(mydataset, batch_size=8, shuffle=False)
# data, mask, mark =next(iter(data_loader))
#
# print('end')
