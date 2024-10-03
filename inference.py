import argparse
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from dataset import CustomDataset
from train import get_embeddings
from utils import _TRANSFORMS, set_cuda_device


def inference(
    csv_path,
    transforms,
    device
):
    with open('isolation_forest_model.pkl', 'rb') as file:
        classifier = pickle.load(file)
        
    test_data = CustomDataset(csv_path, transforms)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    test_embeddings = get_embeddings(classifier, test_loader, device)
    test_pred = classifier.predice(test_embeddings)
    
    test_pred = np.where(test_pred == -1, 1, 0)
    
    submit = pd.read_csv('./sample_submission.csv')
    submit['label'] = test_pred
    submit.head()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()
    
    device = args.device
    csv_path = args.csv_path
    
    set_cuda_device(device)    
    inference(csv_path, _TRANSFORMS)