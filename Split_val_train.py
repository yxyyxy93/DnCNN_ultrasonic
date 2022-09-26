import os
import shutil
import random

# move files to
# Providing the folder path
ori_image_mark = 'noisy_images'
label_image_mark = 'origin_images_label'

origin = os.path.join(os.curdir, "Dataset_learning", "train", ori_image_mark)
target = os.path.join(os.curdir, "Dataset_learning", "val", ori_image_mark)

# Fetching the list of all the files
files = os.listdir(origin)
files_len = len(files)
files_val_len = int(files_len / 5)
random.seed(2022)
files_val = random.sample(files, files_val_len)

# Fetching all the files to directory
for f in files_val:
    if os.path.join(origin, f):
        shutil.move(os.path.join(origin, f), target)
    image_mark = f.split('_')
    label_file_name = 'origin' + '_' + image_mark[1] + '_' + image_mark[2]
    label_f = os.path.join(os.curdir, "Dataset_learning", "train", label_image_mark, label_file_name)
    label_target = os.path.join(os.curdir, "Dataset_learning", "val", label_image_mark, label_file_name)
    if label_f:
        shutil.move(label_f, label_target)