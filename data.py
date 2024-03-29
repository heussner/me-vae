import os
import numpy as np
from skimage.util.dtype import img_as_ubyte
import torch
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from skimage.io import imread
from skimage.util import img_as_ubyte
from typing import Text, Any, Tuple


class SingleCellDataset(Dataset):
    """Custom PyTorch dataset for loading and iterating over images from disk"""

    def __init__(
        self, data_path1: Text, data_path2: Text, data_path3: Text, transform: Any = ToTensor(), eval: bool = False
    ):
        """
        Args:
            data_path (Text): Path to data dir with folders containing augmented set/unaugmented set
            transform (Any, optional): Optional data transofrmation. Defaults to None.
        """
        #sample directories (assuming augmented crops are in different directories, and that there are only 2 inputs/augmentations)
        input_dirs1 = os.listdir(data_path1)
        input_dirs2 = os.listdir(data_path2)
        output_dirs = os.listdir(data_path3)
        #lists of images
        self.img_files1 = []
        self.img_files2 = []
        self.img_files3 = []
        for d1, d2, d3 in zip(input_dirs1, input_dirs2, output_dirs):
            self.img_files1.append(os.path.join(data_path1, d1))
            self.img_files2.append(os.path.join(data_path2, d2))
            self.img_files3.append(os.path.join(data_path3, d3))
        self.transform = transform
        self.eval = eval

    def __len__(self):
        """Returns length of dataset (number of unique samples)"""
        return len(self.img_files1)

    def __getitem__(self, idx: int) -> torch.tensor:
        """Retrieves data sample by index.
        Args:
            idx (int): Index of desired sample
        Returns:
            torch.tensor: Data sample as tensor
        """
        #load images
        filepath1 = self.img_files1[idx]
        filepath2 = self.img_files2[idx]
        filepath3 = self.img_files3[idx]
        img1 = imread(filepath1)
        img2 = imread(filepath2)
        img3 = imread(filepath3)
        
        #for single-channel images
        if len(img1.shape) == 2:
            img1 = np.expand_dims(img1,2)
            img2 = np.expand_dims(img2,2)
            img3 = np.expand_dims(img3,2)
        
        #convert to tensors
        if img1.dtype == "uint16":
            img1 = img_as_ubyte(img1)
        tensor1 = torch.from_numpy(img1)
        tensor1 = tensor1.permute(2, 0, 1).float()
        tensor1 /= 255
        
        if img2.dtype == "uint16":
            img2 = img_as_ubyte(img2)
        tensor2 = torch.from_numpy(img2)
        tensor2 = tensor2.permute(2, 0, 1).float()
        tensor2 /= 255
        
        if img3.dtype == "uint16":
            img3 = img_as_ubyte(img3)
        tensor3 = torch.from_numpy(img3)
        tensor3 = tensor3.permute(2, 0, 1).float()
        tensor3 /= 255

        if self.transform:
            tensor1 = self.transform(tensor1)
            tensor2 = self.transform(tensor2)
            tensor3 = self.transform(tensor3)

        if self.eval:
            return filepath1, filepath2, tensor1, tensor2
        else:
        #returns list of both tensors
            return tensor1, tensor2, tensor3


class SetBackgroundIntensity(object):
    """Custom TorchVision transform to set background intensity value. Assumes background of passed in image is 0."""

    def __init__(self, intensity: int = 0):
        """
        Args:
            intensity (int, optional): Value to set background to. Defaults to 0.
        """
        self.intensity = intensity

    def __call__(self, img: np.ndarray):
        """Apply transformation
        Args:
            img ([np.ndarray]): Image to transform. Should have background (non-signal) values of 0.
        """
        img[img == 0] = self.intensity
        return img


class Saturate(object):
    """Custom TorchVision transform to saturate upper and lower percentils of input image channel-wise"""

    def __init__(self, lower_percentile: float = 0.0, upper_percentile: float = 99.0):
        """
        Args:
            lower_percentile (float, optional): Defaults to 0.0.
            upper_percentile (float, optional): Defaults to 99.0.
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Saturate image according to state upper and lower percentiles.
        Args:
            img (np.ndarray): Image to saturate
        Returns:
            (np.ndarray): Saturated image
        """
        upper = np.percentile(img, self.upper_percentile, axis=-1)
        lower = np.percentile(img, self.lower_percentile, axis=-1)
        img[img <= lower] = lower
        img[img >= upper] = upper
        return img


class MinMaxNormalize(object):

    def __init__(self, min_vals: list, max_vals: list):
        self.min_vals = np.array(min_vals)
        self.max_vals = np.array(max_vals)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.numpy()
        img = np.subtract(img, self.min_vals) / (self.max_val - self.min_val)
        return torch.from_numpy(img)


def load_data(
    data_path1: Text, data_path2: Text, data_path3: Text, eval: bool = False, **kwargs
) -> Tuple[Dataset, DataLoader]:
    """Construct PyTorch Dataloader
    Args:
        data_path1/2 (Text): Path to directory of transformed image data
        data_path3 (Text): Path to directory of transformed output image data
        **transform (Compose): Dataset transformation
        **batch_size (int): Dataloader batch size
        **shuffle (int): Shuffle dataloader or not
        **num_workers (int): Number dataloader workers
        **pin_memory (bool): Can make CUDA based computations more performant (https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
    Returns:
        DataLoader: PyTorch Dataloader
        Dataset: Correspond PyTorch Dataset
    """
    dataset = SingleCellDataset(data_path1, data_path2, data_path3, eval=eval, **kwargs["dataset"])
    loader = DataLoader(dataset, **kwargs["loader"])
    return dataset, loader
