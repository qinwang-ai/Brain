import scipy
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, eakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv3D, UpSampling3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
from keras.applications import VGG19
import sys
from data_loader import DataLoader
import numpy as np
import os
import keras.backend as K

class BrainArea(object):
    def __init__(self, lr, hr, classes=4):
        # Input shape
        self.lr_shape = lr
        self.hr_shape = hr
        self.channels = 1

        optimizer = Adam(0.0002, 0.5)

        # Configure data loader
        self.data_loader = DataLoader(img_h_res=self.hr_shape, img_l_res=self.lr_shape)
        self.vgg = self.build_vgg(classes=4)
        self.vgg.compile(loss=['categorical_crossentropy'],
                              loss_weights=[1],
                              metrics=['acc'],
                              optimizer=optimizer)

    def build_vgg(self, classes=4):
        # Determine proper input shape
        input_shape = self.lr_shape

        img_input = Input(shape=input_shape)

        # Block 1
        x = Conv3D(64, 3,
                          activation='relu',
                          padding='same',
                          name='block1_conv1')(img_input)
        x = Conv3D(64, 3,
                          activation='relu',
                          padding='same',
                          name='block1_conv2')(x)
        x = MaxPooling3D(2, strides=2, name='block1_pool')(x)

        # Block 2
        x = Conv3D(128, 3,
                          activation='relu',
                          padding='same',
                          name='block2_conv1')(x)
        x = Conv3D(128, 3,
                          activation='relu',
                          padding='same',
                          name='block2_conv2')(x)
        x = MaxPooling3D(2, strides=2, name='block2_pool')(x)

        # Block 3
        x = Conv3D(256, 3,
                          activation='relu',
                          padding='same',
                          name='block3_conv1')(x)
        x = Conv3D(256, 3,
                          activation='relu',
                          padding='same',
                          name='block3_conv2')(x)
        x = Conv3D(256, 3,
                          activation='relu',
                          padding='same',
                          name='block3_conv3')(x)
        x = Conv3D(256, 3,
                          activation='relu',
                          padding='same',
                          name='block3_conv4')(x)
        x = MaxPooling3D(2, strides=2, name='block3_pool')(x)

        # Block 4
        x = Conv3D(512, 3,
                          activation='relu',
                          padding='same',
                          name='block4_conv1')(x)
        x = Conv3D(512, 3,
                          activation='relu',
                          padding='same',
                          name='block4_conv2')(x)
        x = Conv3d(512, 3,
                          activation='relu',
                          padding='same',
                          name='block4_conv3')(x)
        x = Conv3d(512, 3,
                          activation='relu',
                          padding='same',
                          name='block4_conv4')(x)
        x = MaxPooling3D(2, strides=2, name='block4_pool')(x)

        # Block 5
        x = Conv3d(512, 3,
                          activation='relu',
                          padding='same',
                          name='block5_conv1')(x)
        x = Conv3d(512, 3,
                          activation='relu',
                          padding='same',
                          name='block5_conv2')(x)
        x = Conv3d(512, 3,
                          activation='relu',
                          padding='same',
                          name='block5_conv3')(x)
        x = Conv3d(512, 3,
                          activation='relu',
                          padding='same',
                          name='block5_conv4')(x)
        x = MaxPooling3D(2, strides=2, name='block5_pool')(x)

        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
        model = models.Model(img_input, x, name='vgg19')
        return model

    def train(self, trainset_path, epochs, batch_size=1):

        for epoch in range(epochs):
            start_time = datetime.datetime.now()
            # ----------------------
            #  Train Discriminator
            # ----------------------
            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr, imgs_info, imgs_shape, imgs_path = self.data_loader.load_data(trainset_path, batch_size)
            X, Y = self.get_train_data(imgs_hr)
            losses = self.vgg.train_on_batch(X, Y)
            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print("iteration:", epoch, " time:", elapsed_time)
            print(self.vgg.metrics_names, losses)

    def get_train_data(self, hrs):
        Y = []
        X = []
        for hr in hrs:
            x = []
            y = []
            i,j,k = hr.shape
            x.append(hr[0:i//2 ,:, 0:k//2])
            y.append(0)
            x.append(hr[i//2: ,:, 0:k//2])
            y.append(1)
            x.append(hr[0:i//2 ,:, k//2:])
            y.append(2)
            x.append(hr[i//2: ,:, k//2:])
            y.append(3)

            nb_classes = len(y)
            targets = np.array([y]).reshape(-1)
            one_hot_targets = np.eye(nb_classes)[targets]
            print('one_hot', one_hot_targets)
            X.append(x)
            Y.append(one_hot_targets)
        return np.array(X), np.array(Y)

def get_name_by_path(path):
    return path.split('/')[-1]
