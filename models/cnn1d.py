import logging
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from sklearn.metrics import roc_auc_score

from models.base import Model
from .sam import SAMSGD


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# データセット
class TrainDataset:
    def __init__(self, features, targets):
        self.features = features.values
        self.targets = targets.values
    def __len__(self):
        return (self.features.shape[0])
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx], dtype=torch.long)
        }
        return dct
class TestDataset:
    def __init__(self, features):
        self.features = features.values
    def __len__(self):
        return (self.features.shape[0])
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct


# モデル本体
class NNModel(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(NNModel, self).__init__()
        cha_1 = 32
        cha_2 = 64
        cha_3 = 64

        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(hidden_size/cha_1/2)
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0],self.cha_1,
                        self.cha_1_reshape) #

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)
        """
        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))"""
        x_s = x
        """
        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))"""

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x

# train,validation,inference
def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        """
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()"""
        def closure():
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        scheduler.step()
        final_loss += loss.item()
    final_loss /= len(dataloader)
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    for data in dataloader:
        inputs = data['x'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        preds.append(outputs.sigmoid().detach().cpu().numpy())
    preds = np.concatenate(preds)
    return preds

def get_scheduler(optimizer,CFG):
    if CFG["scheduler"]=='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG["factor"], patience=CFG["patience"], verbose=True, eps=CFG["eps"])
    elif CFG["scheduler"]=='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG["T_max"], eta_min=CFG["min_lr"], last_epoch=-1)
    elif CFG["scheduler"]=='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG["T_0"], T_mult=1, eta_min=CFG["min_lr"], last_epoch=-1)
    return scheduler

class CNN1d(Model):
    def __init__(self,seed_num, fold_num):
        self.seed_num = seed_num
        self.fold_num = fold_num
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        seed_torch(params["seed"])
        DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
        EPOCHS = params["epochs"]
        BATCH_SIZE = params["bs"]
        LEARNING_RATE = params["lr"]
        EARLY_STOPPING_STEPS = params["early_stopping_step"]
        EARLY_STOP = params["early_stop"]

        # データセットを生成する
        num_features=X_train.shape[1]
        num_targets= params["num_class"]
        hidden_size= params["hidden_size"]
        dropout_rate= params["dropout_rate"]

        # ロガー
        logger = logging.getLogger('main')

        # 上記のパラメータでモデルを学習する
        train_dataset = TrainDataset(X_train, y_train)
        valid_dataset = TrainDataset(X_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        model = NNModel(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size
        )
        model.to(DEVICE)


        optimizer = SAMSGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        scheduler = get_scheduler(optimizer,CFG=params)
        loss_fn = nn.CrossEntropyLoss()
        early_step = 0
        best_auc = 0
        for epoch in range(EPOCHS):
            train_loss = train_fn(model, optimizer,scheduler, loss_fn, trainloader, DEVICE)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
            valid_auc = roc_auc_score(y_valid, valid_preds[:,1])
            logger.debug(f"SEED:{self.seed_num}, FOLD:{self.fold_num}, EPOCH: {epoch}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, valid_auc:{valid_auc:.4f}")

            if valid_auc > best_auc:
                best_auc = valid_auc
                y_valid_pred = valid_preds
                torch.save(model.state_dict(), f"save/best.pth")
                early_step = 0
            elif(EARLY_STOP == True):
                early_step += 1
                if (early_step >= EARLY_STOPPING_STEPS):
                    break

        testdataset = TestDataset(X_test)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
        model = NNModel(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size
        )
        model.load_state_dict(torch.load(f"save/best.pth"))
        model.to(DEVICE)
        y_pred = inference_fn(model, testloader, DEVICE)

        return y_pred, y_valid_pred, model
