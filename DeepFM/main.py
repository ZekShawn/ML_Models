import pandas as pd
import torch.nn as nn
import torch.optim as optim

from dataset import DataSet
from DeepFMCG import DeepFM
from utils import *
from param import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

if __name__ == "__main__":

    # 定义参数
    con = {
        "epochs": 2,
        "lr": 0.005,
        "step_size": 1,
        "weight_decay": 0.001,
        "gamma": 0.8,
        "batch_size": 512,
        "split_rate": 0.2,
        "hid_dims": [256, 128],
        "drop_out": [0.2, 0.2]
    }

    # 定义模型
    train_loader, test_loader, unique_categories = DataSet(
        train=TRAIN, categories=CATEGORIES, numeric=NUMERICS, target=TARGET, label_func=label_func, test=TEST
    ).get_data()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DeepFM(categories_uniques=unique_categories, numeric_features=len(NUMERICS), num_classes=len(LABEL_LIST)+1)
    model.to(device)

    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)
    optimizer = optim.Adam(model.parameters(), lr=con["lr"], weight_decay=con["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=con["step_size"], gamma=con["gamma"])

    # 模型训练
    print(get_parameter_number(model))
    train_and_eval(
        model, train_loader, test_loader, con["epochs"], device, torch.nn.BCELoss, optimizer, scheduler)
