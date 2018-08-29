import scipy
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv3D, UpSampling3D
from keras.models import Sequential, Model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from scipy.ndimage import zoom
import datetime
from IPython.display import clear_output
import matplotlib.pyplot as plt
import sys
import data_loader
from data_loader import DataLoader
import numpy as np
import math
import os
import nibabel as nib
from keras.layers import Lambda
import keras.backend as K

class SRGAN():
    def __init__(self, lr, hr, n_residual_blocks=8, gpus=1):
        # Input shape
        self.lr_shape = lr
        self.hr_shape = hr
        self.channels = 1

        # Number of residual blocks in the generator
        self.n_residual_blocks = n_residual_blocks

        optimizer = Adam(0.0002, 0.5)

        # Configure data loader
        self.data_loader = DataLoader(img_h_res=self.hr_shape, img_l_res=self.lr_shape)

        # Calculate output shape of D (PatchGAN)
        self.disc_patch = (math.ceil(hr[0] / 2.0**3), math.ceil(hr[1] / 2.0**3), math.ceil(hr[2] / 2.0**3), 1)

        # Number of filters in the first layer of D
        self.df = 16

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        if gpus != 1:
            self.discriminator = multi_gpu_model(self.discriminator, gpus=gpus)

        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Low res. images
        img_lr = Input(shape=self.lr_shape)
        img_lr_mask = Input(shape=self.hr_shape)
        img_lr_large = Input(shape=self.hr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator([img_lr, img_lr_mask, img_lr_large])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_lr_mask, img_lr_large], [validity, fake_hr])
        if gpus != 1:
            self.combined = multi_gpu_model(self.combined, gpus=gpus)
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1, 0.01],
                              optimizer=optimizer)

    def build_generator(self):

        def residual_block(layer_input, filters=32):
            """Residual block described in paper"""
            d = Conv3D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv3D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv3d(layer_input, filters=32):
            """Layers used during upsampling"""
            u = UpSampling3D(size=(2, 1, 1))(layer_input)
            u = Conv3D(filters, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)
        img_lr_large = Input(shape=self.hr_shape)
        img_lr_mask = Input(shape=self.hr_shape)

        # Pre-residual block
        c1 = Conv3D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, filters=64)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, filters=64)

        # Post-residual block
        c2 = Conv3D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv3d(c2, filters=64)
        u2 = deconv3d(u1, filters=64)

        # Generate high resolution output
        gen_hr = Conv3D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        #def mask(data):
         #   gen_hr, img_lr_mask, img_lr_large = data
          #  return tf.add(tf.multiply(gen_hr, img_lr_mask), img_lr_large)

        hr = Add()([gen_hr, img_lr_large])

        return Model([img_lr, img_lr_mask, img_lr_large], hr)


    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        #d7 = d_block(d6, self.df*8)
        #d8 = d_block(d7, self.df*8, strides=2)

        d9 = Dense(self.df*8)(d6)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, trainset_path, iterations, batch_size=1, sample_interval=500, save_interval=200, num_g_per_d=3):

        for iteration in range(iterations):
            start_time = datetime.datetime.now()
            # ----------------------
            #  Train Discriminator
            # ----------------------
            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr, imgs_mask, imgs_lr_large, imgs_info, imgs_shape, imgs_path = self.data_loader.load_data(trainset_path, batch_size)
            # test show mask
            #imgs_lr_large = self.data_loader.unnormalize(imgs_lr_large)
            #imgs_mask = self.data_loader.unnormalize(imgs_mask)
            #self.show_img(imgs_lr_large, notebook=True)
            #self.show_img(imgs_mask, notebook=True)
            #return;

            # From low res. image generate high res. version
            fake_hr = self.generator.predict([imgs_lr, imgs_mask, imgs_lr_large])

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr, imgs_mask, imgs_lr_large, imgs_info, imgs_shape, imgs_path = self.data_loader.load_data(trainset_path, batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Train the generators
            g_loss=[]
            for i in range(num_g_per_d):
                print("training G net:", i)
                g_loss = self.combined.train_on_batch([imgs_lr, imgs_mask, imgs_lr_large], [valid, imgs_hr])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            clear_output()
            print(imgs_path[0]+'\n')
            print("%d time:%s d_loss:%f d_acc:%f g_d_loss:%f g_mse_loss:%f" % (iteration, elapsed_time, d_loss[0], d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if (iteration*num_g_per_d) % sample_interval == 0 and iteration != 0:
                self.sample_images(trainset_path, iteration)
            if (iteration*num_g_per_d) % save_interval == 0 and iteration != 0:
                self.save_model()

            # Show on notebook
            imgs_lr = self.data_loader.unnormalize(imgs_lr)
            imgs_hr = self.data_loader.unnormalize(imgs_hr)
            fake_hr = self.data_loader.unnormalize(fake_hr)
            self.show_img(imgs_lr, notebook=True)
            self.show_img(fake_hr, notebook=True)
            self.show_img(imgs_hr, notebook=True)


    def sample_images(self, dataset_path, iteration):
        os.makedirs('./sample_images', exist_ok=True)

        imgs_hr, imgs_lr, imgs_mask, imgs_lr_large, imgs_info, imgs_shape, imgs_path = self.data_loader.load_data(dataset_path, batch_size=1, is_testing=False)
        fakes_hr = self.generator.predict([imgs_lr, imgs_mask, imgs_lr_large])

        img_lr = imgs_lr[0]
        img_hr = imgs_hr[0]
        fake_hr = fakes_hr[0]
        img_path = imgs_path[0]
        img_info = imgs_info[0]
        img_shape = imgs_shape[0]
        name = get_name_by_path(img_path)

        img_lr = self.data_loader.unnormalize(img_lr)
        img_hr = self.data_loader.unnormalize(img_hr)
        fake_hr = self.data_loader.unnormalize(fake_hr)

        # Save low resolution images for comparison
        fig = self.show_img(img_lr, "low_rs")
        fig.savefig("./sample_images/LOW_RS_ITERATION_%d_%s.png" % (iteration, name))
        plt.close()
        save_nii_to_file("LOW_RES_ITERATION_%d_%s.nii" % (iteration, name), img_lr, img_info, img_shape, is_low=True)

        # Save generated images and the high resolution originals
        fig = self.show_img(fake_hr, "generated_hs")
        fig.savefig("./sample_images/GENERATE_HS_ITERATION_%d_%s.png" % (iteration, name))
        plt.close()
        save_nii_to_file("GENERATE_RES_ITERATION_%d_%s.nii" % (iteration, name), fake_hr, img_info, img_shape)

        # Save ground truth
        fig = self.show_img(img_hr, "ground truth")
        fig.savefig("./sample_images/GROUND_TRUTH_ITERATION_%d_%s.png" % (iteration, get_name_by_path(img_path)))
        plt.close()


    def show_img(self, epi_img_data_stand, title='', notebook=False):
        if (len(epi_img_data_stand.shape) == 5):
            epi_img_data_stand = epi_img_data_stand[0]
        x, y, z,_ = epi_img_data_stand.shape
        epi_img_data_stand = np.reshape(epi_img_data_stand,(x,y,z))
        slice0 = epi_img_data_stand[x//2,:,:]
        slice1 = epi_img_data_stand[:,y//2,:]
        slice2 = epi_img_data_stand[:,:,z//2]
        if notebook:
            show_on_notebook(slice0, slice1, slice2)
            return None
        return data_loader.show_slices([slice0, slice1, slice2], title)

    def free_memory(self):
        self.data_loader.resource_pool.clear()

    def load_model(self):
        self.generator.load_weights("models/generator_weights.h5")
        self.discriminator.load_weights("models/discriminator_weights.h5")
        self.combined.load_weights("models/combined_weights.h5")

    def save_model(self):
        print("save weight completed!")
        self.generator.save_weights("models/generator_weights.h5")
        self.discriminator.save_weights("models/discriminator_weights.h5")
        self.combined.save_weights("models/combined_weights.h5")

def get_name_by_path(path):
    filename = path.split('/')[-1]
    return filename.split('.')[0]

def save_nii_to_file(name, data, info, shape, is_low=False):
    if is_low:
        img = get_low_res_file_with_affine(data, info, shape)
    else:
        img = nib.Nifti1Image(data, info.affine, info.header)
        img.update_header()
    nib.save(img, "sample_niis/"+name)

def get_low_res_file_with_affine(data, info, shape):
    affine = np.eye(4)
    affine[0, 0] = shape[0] / data.shape[0]
    test_img = nib.Nifti1Image(data, affine, info.header)
    test_img.update_header()
    return test_img

def show_on_notebook(img0, img1, img2):
    from IPython.display import display
    from PIL import Image
    offset0 = img0.shape[0] - img1.shape[0]
    offset1 = img2.shape[1] - img1.shape[1]
    img0 = np.pad(img0, ((0,0), (20,20+offset1)), 'constant', constant_values=1000)
    img1 = np.pad(img1, ((0,0+offset0), (20,20+offset1)), 'constant', constant_values=1000)
    img2 = np.pad(img2, ((0,0+offset0), (20,20)), 'constant', constant_values=1000)
    img = np.concatenate((img0, img1, img2), axis=1)
    img = Image.fromarray(img)
    img = img.convert('RGB')
    display(img)





