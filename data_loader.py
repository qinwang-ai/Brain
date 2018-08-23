import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
min_voxel_value = 0
max_voxel_value = 1300
class DataLoader(object):
    def __init__(self, img_h_res, img_l_res):
        self.h_img_res = img_h_res
        self.l_img_res = img_l_res

    def get_res_low_from_origin(self, img_h):
        res = []
        data = np.copy(img_h)
        x_l, _, _ = self.l_img_res
        x_h, _, _ = self.h_img_res
        step = x_h//x_l
        for i in range(0, x_l, step):
            res.append(data[i])
        return np.array(res)
        # return zoom(data, (x/x_raw, y/y_raw, z/z_raw))

    def load_data(self, batch_size=1, is_testing=False):
        path_st = "./trainset/*" if not is_testing else "./testset/*"
        path = glob(path_st)
        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            h_img = self.imread(img_path)
            x_raw, y_raw, z_raw = h_img.shape
            x, y, z = self.h_img_res
            h_img_stand = zoom(h_img, (x/x_raw, y/y_raw, z/z_raw))
            l_img_stand = self.get_res_low_from_origin(h_img_stand)
            imgs_hr.append(h_img_stand)
            imgs_lr.append(l_img_stand)

        average = max_voxel_value/2
        imgs_hr = np.array(imgs_hr) / average - 1.
        imgs_lr = np.array(imgs_lr) / average - 1.
        return imgs_hr, imgs_lr


    def imread(self, path):
        mri_image = nib.load(path)
        mri_image_data = mri_image.get_fdata()
        return mri_image_data

def test_preprocessing(X):
    x,y,z,_ = X.shape
    X = np.reshape(X, (x,y,z))
    X = np.rot90(X, 1, axes=(0,2))
    X = np.rot90(X, 2, axes=(1,2))
    return X

def show_slices(slices):
    fig,axes = plt.subplots(len(slices), 1, figsize=(16, 16))
    for i,slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

