import argparse
import os
import random

import numpy as np
import torch
from torchvision import transforms

_TRANSFORMS = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_save_path", type=str, default="outputs")

    return parser.parse_args()


def set_cuda_device(device: str) -> None:
    device_ids = device.split(",")

    if all(dev.isdigit() for dev in device_ids):
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        print(f"Using CUDA device:{device}")
    else:
        raise ValueError("Invalid device index. Must be a numeric value.")
    
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False