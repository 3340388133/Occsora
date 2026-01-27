"""
Utility Functions
=================

Common utility functions used across the enhancement modules.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
import logging
import os


def setup_logging(
    name: str = "OccSoraEnhancement",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def ensure_tensor(
    data: Union[np.ndarray, torch.Tensor],
    device: str = "cuda",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Ensure data is a PyTorch tensor on the correct device.

    Args:
        data: Input data
        device: Target device
        dtype: Target dtype

    Returns:
        Tensor on correct device
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    return data.to(device=device, dtype=dtype)


def ensure_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Ensure data is a NumPy array.

    Args:
        data: Input data

    Returns:
        NumPy array
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device.

    Args:
        device: Device string or None for auto-detection

    Returns:
        PyTorch device
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Print model information.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }

    return info


def save_tensor(
    tensor: torch.Tensor,
    path: str,
    format: str = "npy"
) -> None:
    """
    Save tensor to file.

    Args:
        tensor: Tensor to save
        path: Output path
        format: Output format ("npy", "pt", "npz")
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if format == "npy":
        np.save(path, ensure_numpy(tensor))
    elif format == "pt":
        torch.save(tensor, path)
    elif format == "npz":
        np.savez_compressed(path, data=ensure_numpy(tensor))
    else:
        raise ValueError(f"Unknown format: {format}")


def load_tensor(
    path: str,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Load tensor from file.

    Args:
        path: Input path
        device: Target device

    Returns:
        Loaded tensor
    """
    if path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".pt"):
        data = torch.load(path, map_location=device)
        return data
    elif path.endswith(".npz"):
        data = np.load(path)['data']
    else:
        raise ValueError(f"Unknown file format: {path}")

    return ensure_tensor(data, device)


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        if torch.cuda.is_available():
            self.start.record()
        else:
            import time
            self._start_time = time.time()

        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            self.end.record()
            torch.cuda.synchronize()
            self.elapsed = self.start.elapsed_time(self.end) / 1000.0
        else:
            import time
            self.elapsed = time.time() - self._start_time

        if self.name:
            print(f"{self.name}: {self.elapsed:.3f}s")


def create_grid_visualization(
    data: torch.Tensor,
    num_cols: int = 4,
    padding: int = 2
) -> np.ndarray:
    """
    Create a grid visualization of frames.

    Args:
        data: Data tensor (T, H, W) or (T, C, H, W)
        num_cols: Number of columns in grid
        padding: Padding between frames

    Returns:
        Grid image as numpy array
    """
    data = ensure_numpy(data)

    if len(data.shape) == 3:
        data = data[:, np.newaxis, :, :]  # Add channel dim

    T, C, H, W = data.shape
    num_rows = (T + num_cols - 1) // num_cols

    grid_h = num_rows * H + (num_rows - 1) * padding
    grid_w = num_cols * W + (num_cols - 1) * padding

    if C == 1:
        grid = np.zeros((grid_h, grid_w))
    else:
        grid = np.zeros((grid_h, grid_w, C))

    for idx in range(T):
        row = idx // num_cols
        col = idx % num_cols

        y = row * (H + padding)
        x = col * (W + padding)

        if C == 1:
            grid[y:y+H, x:x+W] = data[idx, 0]
        else:
            grid[y:y+H, x:x+W] = data[idx].transpose(1, 2, 0)

    return grid
