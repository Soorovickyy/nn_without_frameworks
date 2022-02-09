from helper_function import *
import imgaug.augmenters as iaa
import numpy as np
import random

def model(X, Y, layers_dims, lr = 0.0001, mini_batch_size = 64, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
          num_epochs = 20, lambd = 0.001, print_cost = True, L2 = True):
    costs = []
    t = 0
    m = X.shape[1]
    num_aug_img = 12000
    param = initialize_parameters_he(layers_dims)
    v, s = initialize_adam(param)
    seq = iaa.Sequential([iaa.Sometimes(0.5, iaa.Crop(px=(0, 16))), #px=(0, 16)percent=(0, 0.1)
                          iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0))),
                          iaa.Flipud(0.5),
                          iaa.Sometimes(0.5, iaa.Affine(
                              scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                              translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                              rotate=(-45, 45),
                              shear=(-16, 16)))]) # random_order=True

    for i in range(1, num_epochs + 1):
        #if num_epochs % 10 == 0:
            #lr /= 10
        idx_aug = random.sample(range(X.shape[1]), num_aug_img)
        images_aug = seq(images=X[:, idx_aug])
        Y_model = np.append(Y, Y[:, idx_aug], axis=1)
        X_model = np.append(X, images_aug, axis=1)
        mini_batches = random_mini_batches(X_model, Y_model, mini_batch_size)
        cost_total = 0
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, param)
            cost_total += cost_with_L2(AL, mini_batch_Y, param, lambd)
            #cost_total += cost(AL, mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y, caches, lambd=lambd, L2=L2)
            t += 1
            param, v, s = update_parametrs_with_adam(param, grads, v, s, t=t, lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
            #param = update_parametrs_with_gd(param, grads, lr)
        cost_avg = cost_total / m
        if print_cost and i % 1 == 0:
            print('Cost after epoch %i: %f' %(i, cost_avg))
        if print_cost and i % 1 == 0:
            costs.append(cost_avg)
    return param
# ia.seed(1)
# seq = iaa.Sequential([iaa.Fliplr(0.5),iaa.Crop(percent=(0, 0.1)),
#                     iaa.Sometimes(0.5,
#                     iaa.GaussianBlur(sigma=(0, 0.5))),iaa.LinearContrast((0.75, 1.5)),iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#                     iaa.Multiply((0.8, 1.2), per_channel=0.2),
#                     iaa.Affine(
#                           scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#                           translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#                           rotate=(-25, 25),
#                           shear=(-8, 8))], random_order=True)

X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
image_names_train  = np.load('image_names_train.npy')
X_dev = np.load('X_dev.npy')
Y_dev = np.load('Y_dev.npy')
image_names_dev  = np.load('image_names_dev.npy')
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')
image_names_test  = np.load('image_names_test.npy')
layers_dims = [X_train.shape[0], 256, 128, 64, 32, 1] #256, 128, 64, 32, 16, 8, 4, 1
param = model(X_train, Y_train, layers_dims)
predictions = predict(X_train, Y_train, param)
pred_dev = predict(X_dev, Y_dev, param)
error_images = error_analysis(Y_dev, pred_dev, image_names_dev)
#pred_test = predict(X_test, Y_test, param)

