""" torch """
import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, categories_uniques, numeric_features=0, emb_size=8,
                 hid_dims=None, num_classes=1, dropout=None):
        """
        categories_uniques: 类别特征的唯一值个数列表，也就是每个类别特征的属性数量所组成的列表
        numeric_features: 数值特征的个数，该模型会考虑到输入全为类别型，即没有数值特征的情况
        """
        super().__init__()
        if dropout is None:
            dropout = [0.2, 0.2]
        if hid_dims is None:
            hid_dims = [256, 128]
        self.categories_uniques_size = len(categories_uniques)
        self.numeric_features = numeric_features

        """FM部分"""
        # 一阶
        if self.numeric_features != 0:
            self.fm_1st_order_dense = nn.Linear(self.numeric_features, 1)  # 数值特征的一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in categories_uniques])  # 类别特征的一阶表示

        # 二阶
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in categories_uniques])  # 类别特征的二阶表示

        """DNN部分"""
        self.all_dims = [self.categories_uniques_size * emb_size] + hid_dims
        # 数值特征的维度变换到FM输出维度一致
        self.dense_linear = nn.Linear(self.numeric_features, self.categories_uniques_size * emb_size)
        self.relu = nn.ReLU()

        # for DNN
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i - 1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i - 1]))

        # for output
        self.dnn_linear = nn.Linear(hid_dims[-1], num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_categories, x_numeric=None):
        """
        x_categories: 类别型特征输入  [bs, cate_fea_size]
        x_numeric: 数值型特征输入（可能没有）  [bs, numeric_fea_size]
        """

        """FM 一阶部分"""
        fm_1st_sparse_res = [emb(x_categories[:, i].unsqueeze(1)).view(-1, 1)
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1, keepdim=True)  # [bs, 1]

        if x_numeric is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(x_numeric)
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res  # [bs, 1]

        """FM 二阶部分"""
        fm_2nd_order_res = [emb(x_categories[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  n为类别型特征个数(cate_fea_size)

        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed  # [bs, emb_size]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        # 相减除以2
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5  # [bs, emb_size]

        fm_2nd_part = torch.sum(sub, 1, keepdim=True)  # [bs, 1]

        """DNN部分"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)  # [bs, n * emb_size]

        if x_numeric is not None:
            dense_out = self.relu(self.dense_linear(x_numeric))  # [bs, n * emb_size]
            dnn_out = dnn_out + dense_out  # [bs, n * emb_size]

        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        dnn_out = self.dnn_linear(dnn_out)  # [bs, 1]
        out = fm_1st_part + fm_2nd_part + dnn_out  # [bs, 1]
        out = self.sigmoid(out)
        return out
