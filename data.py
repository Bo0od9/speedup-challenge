import os
import urllib.request
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader


def download_if_needed(url: str, cache_dir: str = "./data") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, url.split("/")[-1])

    if not os.path.exists(filename):
        with tqdm(unit="B", unit_scale=True, desc=f"Downloading {filename}") as pbar:
            urllib.request.urlretrieve(
                url,
                filename,
                reporthook=lambda block_num, block_size, total_size: pbar.update(block_size),
            )
    return filename


def load_dataset(train_url: str, val_url: str, cache_dir: str = "./data"):
    train_file = download_if_needed(train_url, cache_dir)
    val_file = download_if_needed(val_url, cache_dir)

    train_tensors = torch.load(train_file, map_location="cpu", weights_only=True)
    val_tensors = torch.load(val_file, map_location="cpu", weights_only=True)

    train_X = torch.nan_to_num(train_tensors[0])
    train_y_raw = torch.nan_to_num(train_tensors[1])
    val_X = torch.nan_to_num(val_tensors[0])
    val_y = torch.nan_to_num(val_tensors[1])

    Y_mean = train_y_raw.mean()
    Y_std = train_y_raw.std()

    train_y = (train_y_raw - Y_mean) / Y_std
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)

    return train_dataset, val_dataset, Y_mean, Y_std


def create_dataloaders(train_dataset,
                       val_dataset,
                       batch_size_train=1024,
                       batch_size_val=2048,
                       num_workers=2,
                       ):

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_dl, val_dl
