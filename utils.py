from typing import Union
import torch
from torch import Tensor
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import random_split
import shutil
import time
# from data_generate.data_generate_image import draw_rectangle,draw_circle,draw_cross,draw_ellipse,draw_ring,draw_rotated_polygon
def generate_square_subsequent_mask(size: int) -> Tensor:
    """
    参考代码：https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/36cab9d6dcdad84e3d1a69df5ab796cbf689c115/lab9/text_recognizer/models/transformer_util.py

    生成上三角掩码矩阵
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def first_element(x: torch.Tensor, element: Union[int, float], dim: int = 1) -> torch.Tensor:
    """
    参考代码：https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/text_recognizer/lit_models/util.py
    Return indices of first occurence of element in x. If not found, return length of x along dim.
    Based on https://discuss.pytorch.org/t/first-nonzero-index/24769/9
    Examples
    --------
    >>> first_element(torch.tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
    tensor([2, 1, 3])
    """
    nonz = x == element
    ind = ((nonz.cumsum(dim) == 1) & nonz).max(dim).indices
    ind[ind == 0] = x.shape[dim]
    return ind

def spec_interpolator(wavelength,intensity,freq):
    wav = np.loadtxt(wavelength)
    int = np.loadtxt(intensity)
    interpolator = interp1d(wav, int, kind='cubic')
    new_wavelengths = np.linspace(380, 800, freq)
    new_intensities = interpolator(new_wavelengths)
    return new_wavelengths, new_intensities

def split_data(wave,name):

    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1
    wave_train, wave_remain, labels_train, labels_remain = train_test_split(wave, name, test_size=(1 - train_ratio), random_state=42)
    relative_test_ratio = test_ratio / (test_ratio + validation_ratio)
    wave_val, wave_test, labels_val, labels_test = train_test_split(wave_remain, labels_remain, test_size=relative_test_ratio, random_state=42)
    return wave_train,labels_train,wave_val,labels_val,wave_test,labels_test

def extract_type(data):
    type_list = []
    for i in data:
        type_list.append(i.split('-')[0])
    return type_list

def backup_code(source_dir,backup_dir, backup_name, file_list = None,note=None):
    # 获取当前时间
    current_time = time.strftime("%Y%m%d_%H%M%S")
    # 构建备份文件名
    backup_filename = f"{backup_name}_{current_time}"
    # 确保备份目录存在，如果不存在则创建
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # 如果文件列表为空，则备份整个源代码目录
    if file_list is None:
        source_files = os.listdir(source_dir)
    else:
        source_files = file_list

    # 复制源代码到备份目录
    try:
        backup_path = os.path.join(backup_dir, backup_filename)
        os.makedirs(backup_path)
        for file_name in source_files:
            source_file_path = os.path.join(source_dir, file_name)
            if os.path.isfile(source_file_path):
                shutil.copy2(source_file_path, backup_path)
            if note is not None:
                with open(os.path.join(backup_path, "note.txt"), "w") as f:
                    f.write(note)
        print("Code backed up successfully!")
    except Exception as e:
        print(f"Backup failed: {e}")

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def create_folder_if_not_exists(folder_path):
    """
    检查文件夹是否存在，不存在则创建文件夹

    参数：
    folder_path (str): 要检查的文件夹路径

    返回：
    None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")




