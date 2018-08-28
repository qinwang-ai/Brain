import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
min_voxel_value = 0
#max_voxel_value = 5653.5
max_voxel_standard_value = 1024
class DataLoader(object):
    def __init__(self, img_h_res, img_l_res):
        self.h_img_res = img_h_res
        self.l_img_res = img_l_res
        self.resource_pool = {}

    def get_res_low_from_origin(self, img_h):
        lr = []
        mask = []
        lr_large = []
        data = np.copy(img_h)
        x_l = self.l_img_res[0]
        x_h = self.h_img_res[0]
        step = x_h//x_l
        start = 0
        plane0 = data[0]
        for i in range(x_l):
            if start < data.shape[0]:
                lr.append(data[start])
                lr_large.append(data[start])
                mask.append(np.zeros_like(plane0))
            start += step
            for j in range(step - 1):
                mask.append(np.ones_like(plane0))
                lr_large.append(np.zeros_like(plane0))
        if len(mask) > x_h:
            mask = mask[:x_h]
        if len(lr_large) > x_h:
            lr_large = lr_large[:x_h]
        lr = np.array(lr)
        mask = np.array(mask)
        lr_large = np.array(lr_large)
        return lr, mask, lr_large

    def get_file_count(self, dataset_path):
        paths = glob(dataset_path, recursive=True)
        return len(paths)

    def load_data(self, dataset_path, batch_size=1, is_testing=False):
        paths = glob(dataset_path, recursive=True)
        batch_images = np.random.choice(paths, size=batch_size)
        imgs_hr = []
        imgs_lr = []
        imgs_mask = []
        imgs_lr_large = []
        imgs_info = []
        imgs_path = []
        imgs_shape = []
        for path in batch_images:
            if path not in self.resource_pool:
                info, h_img = self.imread(path)
                # prevent run out of memory
                if (len(self.resource_pool) < 500):
                    self.resource_pool[path] = (info, np.copy(h_img))
            else:
                info, h_img = self.resource_pool[path]
                h_img = np.copy(h_img)

            if is_testing:
                h_img = test_preprocessing(h_img)
                h_img = self.normalize(h_img)
                h_img_stand = h_img
                l_img_stand = h_img
                # TODO test mask
                mask = None
                l_img_large = None
            else:
                x_raw, y_raw, z_raw = h_img.shape
                x, y, z,_ = self.h_img_res
                h_img_stand = zoom(h_img, (x/x_raw, y/y_raw, z/z_raw))
                h_img_stand = self.normalize(h_img_stand)
                l_img_stand, mask, l_img_large = self.get_res_low_from_origin(h_img_stand)
            h_img_stand = np.expand_dims(h_img_stand, axis=-1)
            l_img_stand = np.expand_dims(l_img_stand, axis=-1)
            mask = np.expand_dims(mask, axis=-1)
            l_img_large = np.expand_dims(l_img_large, axis=-1)

            imgs_hr.append(h_img_stand)
            imgs_lr.append(l_img_stand)
            imgs_mask.append(mask)
            imgs_lr_large.append(l_img_large)
            imgs_info.append(info)
            imgs_shape.append(h_img.shape)
            imgs_path.append(path)

            imgs_hr = np.array(imgs_hr)
            imgs_lr = np.array(imgs_lr)
            imgs_mask = np.array(imgs_mask)
            imgs_lr_large = np.array(imgs_lr_large)

        return imgs_hr, imgs_lr, imgs_mask, imgs_lr_large, imgs_info, imgs_shape, imgs_path

    def normalize(self, img):
        max_value = np.max(img)
        average = max_value/2.0
        return np.array(img) / float(average) - 1.

    def unnormalize(self, img, max_value=None):
        if max_value is None:
            max_value = max_voxel_standard_value
        average = max_value/2.0
        return (img + 1) * average

    def imread(self, path):
        mri_image = nib.load(path)
        mri_image_data = mri_image.get_fdata()
        return mri_image, mri_image_data

def test_preprocessing(X):
    x,y,z,_ = X.shape
    X = np.reshape(X, (x,y,z))
    X = np.rot90(X, 1, axes=(0,2))
    X = np.rot90(X, 2, axes=(1,2))
    return X

def show_slices(slices, title=''):
    fig, axes = plt.subplots(len(slices), 1, figsize=(16, 16))
    for i,slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[i].set_title(title)
    return fig

def clear_samples():
    import os
    os.system("rm sample_niis/*")
    os.system("rm sample_images/*")

