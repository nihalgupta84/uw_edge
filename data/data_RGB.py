import os
from .dataset_RGB import DataReader, NonRefDataReader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def get_data(img_dir, inp='raw', tar='ref', mode='train', ori=False, img_options=None):
    if not os.path.exists(img_dir):
        print(f"Warning: Directory '{img_dir}' does not exist. Returning empty dataset.")
        return []
    return DataReader(img_dir, inp, tar, mode, ori, img_options)


def get_data_nonref(img_dir, inp='raw', mode='train', ori=False, img_options=None):
    if not os.path.exists(img_dir):
        print(f"Warning: Directory '{img_dir}' does not exist. Returning empty dataset.")
        return []
    return NonRefDataReader(img_dir, inp, mode, ori, img_options)

