import numpy as np
def imread(path):
    mri_image = nib.load(path)
    mri_image_data = mri_image.get_fdata()
    return mri_image_data
Min = 9999
Max = -1000
def summary(trainset_path):
    global Min, Max
    # summary all the trainset to find minimum and maximum voxel value, consume several minutes
    path_st = trainset_path
    paths = glob(path_st)
    if Min != 9999:
        return Min, Max
    for p in paths:
        img_data = imread(p)
        Min = min(np.min(img_data), Min)
        Max = max(np.max(img_data), Max)
    return Min, Max



