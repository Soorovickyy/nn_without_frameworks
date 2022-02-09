from imutils import paths
import numpy as np
import shutil
import os
dev_split = 0.02
test_split = 0.02
def data_set(image_paths, folder):
    if not os.path.exists(folder): #if folder not exist,
        os.makedirs(folder)            #then create it
    for path in image_paths:
        image_name = path.split(os.path.sep)[-1] #split image path by '/' and store the name in a variable
        label = path.split(os.path.sep)[1] #split image path by '/' and store the name of label in a variable
        label_folder = os.path.join(folder, label) #define path for data set/label
        if not os.path.exists(label_folder): #if folder not exist,
            os.makedirs(label_folder)          #then create it
        destination = os.path.join(label_folder, image_name) #define destination path for image
        shutil.copy(path, destination) #copy the file path to the directory destination

image_paths = list(paths.list_images('PetImages')) #list of paths to images
np.random.shuffle(image_paths) #randomly shuffle images
dev_len = int(len(image_paths) * dev_split)
test_len = int(len(image_paths) * test_split)
dev_paths = image_paths[:dev_len]
test_paths = image_paths[-test_len:]
train_paths = image_paths[dev_len:-test_len]
train = os.path.join('dataset', 'train') #define path for train set
dev = os.path.join('dataset', 'dev') #define path for dev set
test = os.path.join('dataset', 'test') #define path for test set

# train/dev/test distribution
data_set(train_paths, train)
data_set(dev_paths, dev)
data_set(test_paths, test)