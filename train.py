import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import CustomDataset
from utils import parse_arguments, set_cuda_device, _TRANSFORMS


def train_backbone(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=10,
    model_save_path="outputs",
):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_len = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            prediction = (torch.sigmoid(outputs) > 0.5).float()
            running_corrects += torch.sum(prediction == labels.view(-1, 1)).item()
            total_len += labels.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects / total_len

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        )

        save_path = os.path.join(model_save_path, f"resnet18_epoch_{num_epochs}.pth")
        torch.save(model.state_dict(), save_path)
        
        return model


def train_network() -> None:
    args = parse_arguments()
    set_cuda_device(args.device)

    device_ids = [int(d) for d in args.device.split(",")]
    device = torch.device(
        f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
    )

    model_save_path = args.model_save_path
    if model_save_path:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

    batch_size = args.batch_size
    num_epochs = args.epoch
    csv_path = args.csv_path
    seed = args.seed
    transforms = _TRANSFORMS
    
    train_data = CustomDataset(csv_path, transforms)
    train_loader = DataLoader(train_data, batch_size, shuffle=False)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
    model = model.to(device)

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)

    backbone_model = train_backbone(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        save_dir=model_save_path,
        file_name=f'backbone_epoch{num_epochs}_seed{seed}.pth',
    )
    
    train_embeddings = get_penultimate_feature(
        backbone_model, train_loader, device
    )
    
    classifier = IsolationForest(random_state=seed)
    classifier.fit(train_embeddings)
    
    with open(f'classifier_epoch{num_epochs}.pkl', 'wb') as file:
        pickle.dump(classifier, file)
    

def get_embeddings(model, dataloader, device):
    embeddings = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            emb = model(images)
            embeddings.append(emb.cpu().numpy().squeeze())
    return np.concatenate(embeddings, axis=0)


def get_penultimate_feature(model, train_loader, device):
    model.eval()
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.to(device)
    
    train_embeddings = get_embeddings(model, train_loader)
    return train_embeddings


if __name__ == "__main__":
    train_network()