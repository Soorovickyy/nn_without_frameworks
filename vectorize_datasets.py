import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import os
num_px = 64
def make_set(dirs, normalize_method):
    X = []
    Y = []
    image_names = []
    for dir in dirs:
        for file in os.listdir(dir):
            if file == '.DS_Store':
                continue
            if dir.split(os.path.sep)[-1] == 'Dog':
                label = 1
            elif dir.split(os.path.sep)[-1] == 'Cat':
                label = 0
            fileImage = Image.open(os.path.join(dir, file)).convert("RGB").resize([num_px, num_px], Image.ANTIALIAS)
            image = np.array(fileImage)
            my_image = image.ravel()
            my_image = my_image / 255.
            X.append(my_image)
            Y.append(str(label))
            image_names.append(str(dir.split(os.path.sep)[-2]) + '/' + str(dir.split(os.path.sep)[-1]) + '/' + str(file))
    X = np.asarray(X, dtype='float64')
    X = X.T  # (12288, 24000)
    Y = np.asarray(Y, dtype='float64')
    Y = Y[np.newaxis, :]  # (1, 24000)
    if normalize_method == 'Min-Max':
        X = (X - X.min()) / (X.max() - X.min())
    elif normalize_method == 'Mean-Std':
        X = (X - X_mean) / X_std
    np.save('X_' + str(dir.split(os.path.sep)[1]) + '.npy', X)
    np.save('Y_' + str(dir.split(os.path.sep)[1]) + '.npy', Y)
    np.save('image_names_' + str(dir.split(os.path.sep)[1]) + '.npy', image_names)
    # return X, Y


data_set = []
for dir in ['PetImages/Cat', 'PetImages/Dog']:
    for file in os.listdir(dir):
        if file == '.DS_Store':
            continue
        fileImage = Image.open(os.path.join(dir, file)).convert("RGB").resize([num_px, num_px], Image.ANTIALIAS)
        image = np.array(fileImage)
        my_image = image.ravel()
        my_image = my_image / 255.
        data_set.append(my_image)
data_set = np.asarray(data_set, dtype='float64').T
X_mean, X_std = data_set.mean(), data_set.std()



make_set(['dataset/train/Cat', 'dataset/train/Dog'], normalize_method = 'Mean-Std')
make_set(['dataset/dev/Cat', 'dataset/dev/Dog'], normalize_method = 'Mean-Std')
make_set(['dataset/test/Cat', 'dataset/test/Dog'], normalize_method = 'Mean-Std')
