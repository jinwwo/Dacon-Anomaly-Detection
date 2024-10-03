import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df['img_path'].iloc[idx]
        image = Image.open(img_path)        
        if self.transform:
            image = self.transform(image)
        target = torch.tensor([0.]).float()
        
        return image, target