import numpy as np
import hdf5storage
import os
import time
import datetime
import numpy as np
import shutil
import os


class SliceMat:
    def __init__(self, image_dir):
        self.image_dir = image_dir

        # Read the path of the image
        image_filenames = list(os.listdir(image_dir))
        self.file_names = [file_name for file_name in image_filenames if
                           os.path.splitext(file_name)[1] == '.mat']

    def __SliceSave__(self, save_path):
        for file_name in self.file_names:
            path = os.path.join(self.image_dir, file_name)
            image = hdf5storage.loadmat(path)["img_batch"]
            image = np.array(image)

            # Check whether the specified path exists or not
            isExist = os.path.exists(save_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(save_path)
                print("The new directory is created")

            # slice the dataset - 3 channels
            # for idx in range(0, np.ma.size(image, axis=2)-2):
            #     if int(file_name.split('_')[2]) <= 10:
            #         filepath = os.path.join(save_path, (file_name.split('.')[0] + '_' + str(idx)))
            #         np.save(filepath, image[:, :, idx:idx+3], allow_pickle=True)

            # 1 channel
            for idx in range(0, np.ma.size(image, axis=2)):
                if int(file_name.split('_')[2]) <= 10:
                    filepath = os.path.join(save_path, (file_name.split('.')[0] + '_' + str(idx)))
                    np.save(filepath, image[:, :, idx], allow_pickle=True)

image_dir = r'F:\Xiayang\results\Woven_samples\Dataset_learning\noisy_images'
save_dir = r'F:\Xiayang\python_work\dncnn_ultrasonic\Dataset\train\noisy_images'

# image_dir = r'F:\Xiayang\results\Woven_samples\Dataset_learning\origin_images_label'
# save_dir = r'F:\Xiayang\python_work\dncnn_ultrasonic\Dataset\train\origin_images_label'

Matreader = SliceMat(image_dir)
Matreader.__SliceSave__(save_dir)
