import torch
from tqdm import tqdm
from pandas import read_csv, concat
from torch.utils import data as tdata
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataSet:

    def __init__(self, train: str, categories: list, numeric: list, target: str,
                 label_func=None, test: str = None, batch_size: int = 512,
                 split_rate: float = 0.2) -> None:
        self.train = train
        self.test = test
        self.categories = categories
        self.numeric = numeric
        self.target = target
        self.label_func = label_func
        self.batch_size = batch_size
        self.split_rate = split_rate
        self._parse_data()

    def _parse_data(self):

        # 数据读取
        self.train = read_csv(self.train)
        if self.test is not None:
            self.test = read_csv(self.test)

        # 数据缺失值填充
        self.train[self.categories] = self.train[self.categories].fillna("缺失值")
        self.train[self.numeric] = self.train[self.numeric].fillna(-1)

        # 数据标签转换
        if self.label_func is not None:
            temp_series = self.train[self.target].apply(lambda x: self.label_func(x)).copy()
            self.train["label"] = temp_series
            if self.test is not None:
                temp_series = self.test[self.target].apply(lambda x: self.label_func(x)).copy()
                self.test["label"] = temp_series

        # 类别数据转换
        for cate in tqdm(self.categories):
            le = LabelEncoder()
            le.fit(self.train[cate])
            self.train[cate] = le.transform(self.train[cate])
            if self.test is not None:
                self.test[cate] = le.transform(self.test[cate])

        # 数值型特征数据转换
        for num in tqdm(self.numeric):
            mean = self.train[num].mean()
            std = self.train[num].std()
            self.train[num] = self.train[num].apply(lambda x: (x - mean) / (std + 1e-12))
            if self.test is not None:
                self.test[num] = self.test[num].apply(lambda x: (x - mean) / (std + 1e-12))

    def get_data(self):

        # 分割数据集
        if self.test is not None:
            train, valid = self.train, self.test
        else:
            train, valid = train_test_split(self.train, test_size=self.split_rate, random_state=2022)

        # 训练数据集
        train_dataset = tdata.TensorDataset(torch.LongTensor(train[self.categories].values),
                                            torch.FloatTensor(train[self.numeric].values),
                                            torch.FloatTensor(train[self.target].values), )
        train_loader = tdata.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        # 验证数据集
        valid_dataset = tdata.TensorDataset(torch.LongTensor(valid[self.categories].values),
                                            torch.FloatTensor(valid[self.numeric].values),
                                            torch.FloatTensor(valid[self.target].values), )
        valid_loader = tdata.DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)

        # 不重复数据集的
        if self.test is not None:
            data = concat([self.train, self.test], axis=0)
        else:
            data = self.train
        unique_categories = [data[fea].unique() for fea in self.categories]
        return train_loader, valid_loader, unique_categories
