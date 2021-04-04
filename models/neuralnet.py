import logging

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

from models.base import Model
from .sam import SAMSGD

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
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.3)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.4)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.4)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def recalibrate_layer(self, layer):
        if(torch.isnan(layer.weight_v).sum() > 0):
            print ('recalibrate layer.weight_v')
            layer.weight_v = torch.nn.Parameter(torch.where(torch.isnan(layer.weight_v), torch.zeros_like(layer.weight_v), layer.weight_v))
            layer.weight_v = torch.nn.Parameter(layer.weight_v + 1e-7)
        if(torch.isnan(layer.weight).sum() > 0):
            print ('recalibrate layer.weight')
            layer.weight = torch.where(torch.isnan(layer.weight), torch.zeros_like(layer.weight), layer.weight)
            layer.weight += 1e-7
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        #self.recalibrate_layer(self.dense1)
        x = F.relu(self.dense1(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        #self.recalibrate_layer(self.dense2)
        x = F.relu(self.dense2(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        #self.recalibrate_layer(self.dense3)
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

class NeuralNet(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
        EPOCHS = params["epochs"]
        BATCH_SIZE = params["bs"]
        LEARNING_RATE = params["lr"]
        WEIGHT_DECAY = params["weight_decay"]
        EARLY_STOPPING_STEPS = params["early_stopping_step"]
        EARLY_STOP = params["early_stop"]

        # データセットを生成する
        num_features=X_train.shape[1]
        num_targets= params["num_class"]
        hidden_size= params["hidden_size"]

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
            hidden_size=hidden_size,
        )
        model.to(DEVICE)


        #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        optimizer = SAMSGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                                max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
        #loss_fn = nn.BCEWithLogitsLoss()
        loss_fn = nn.CrossEntropyLoss()
        early_step = 0
        best_loss = np.inf
        for epoch in range(EPOCHS):
            train_loss = train_fn(model, optimizer,scheduler, loss_fn, trainloader, DEVICE)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
            logger.debug(f"EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")

            if valid_loss < best_loss:
                best_loss = valid_loss
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
            hidden_size=hidden_size,
        )
        model.load_state_dict(torch.load(f"save/best.pth"))
        model.to(DEVICE)
        y_pred = inference_fn(model, testloader, DEVICE)

        return y_pred, y_valid_pred, model
