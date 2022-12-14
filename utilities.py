import pandas as pd
from skimage import io
from skimage.feature import hog
import numpy as np
import os
import shutil


def file_filter(dir, files):  # https://stackoverflow.com/questions/15663695/shutil-copytree-without-files
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


def img2hog(csv_path, col_img_path, rootdir) -> None:
    if not os.path.isdir(os.path.join(rootdir, 'hog')):
        shutil.copytree(rootdir, os.path.join(rootdir, 'hog'), ignore=file_filter)
    table = pd.read_csv(csv_path)
    for path in table[col_img_path]:
        image = io.imread(os.path.join('dataset_cards', path))
        image = np.squeeze(hog(image, channel_axis=2, block_norm='L2', pixels_per_cell=(4, 4), cells_per_block=(1, 1),
                               feature_vector=False))
        np.save(os.path.join(rootdir, 'hog', path.split('.')[0])+'.npy', image)


if __name__ == '__main__':
    img2hog(os.path.join('dataset_cards/cards.csv'), 'filepaths', 'dataset_cards')
