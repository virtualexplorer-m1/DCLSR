# here put the import lib
from importlib.metadata import distribution
from inspect import isfunction
from operator import concat
from os.path import exists
from platform import node

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from math import sqrt
from torch.nn import GroupNorm


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
    


class Contrastive_Loss2(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau


    def forward(self, X, Y):
        
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    


class CalculateAttention(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, Q, K, V, mask):

        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention


import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ELBO(nn.Module):
    def __init__(self, num_cluster, feat_dim, tau, kappa, eta, device):
        super(ELBO, self).__init__()
        print("--------------------------Initializing -----------------------------------")
        self.num_cluster = num_cluster
        self.feat_dim = feat_dim
        self.tau = tau
        self.kappa = kappa
        self.eta = eta
        self.device = device

        # 初始化聚类原型，形状为 [num_cluster, feat_dim]
        self.prototype = nn.Parameter(
            torch.nn.init.uniform_(torch.Tensor(self.num_cluster, self.feat_dim), a=0, b=1)
        )
        self.logSoftmax = nn.LogSoftmax(dim=1)
        print("----------------------Initialization Ends-----------------------------------")

    def update_cluster(self, emb):
        """
        更新聚类中心。这里对输入 emb（形状 [batch, seq_len, feat_dim]）先进行平均池化，
        每个样本的全局特征用于更新聚类中心。
        """
        B, T, D = emb.shape
        emb_pool = torch.mean(emb, dim=1)  # shape: [B, D]

        # 计算每个样本与每个聚类中心的相似度
        dist = torch.matmul(emb_pool, self.prototype.T)  # [B, num_cluster]
        cluster_assignments = torch.argmax(dist, dim=1)  # [B]

        new_center = torch.zeros(self.num_cluster, self.feat_dim, device=self.device)
        for i in range(self.num_cluster):
            cluster_points = emb_pool[cluster_assignments == i]
            if cluster_points.shape[0] > 0:
                new_center[i] = torch.mean(cluster_points, dim=0)

        # print("Prototype shape:", self.prototype.shape)
        # print("New center shape:", new_center.shape)
        self.prototype.data = new_center

    def compute_mean_cov(self, emb):
        """
        计算给定特征向量的均值和协方差矩阵。这里先对序列做平均池化，再计算均值和协方差。
        输入 emb 形状为 [batch, seq_len, feat_dim]。
        """
        B, T, D = emb.shape
        emb_pool = torch.mean(emb, dim=1)  # shape: [B, D]
        mean = torch.mean(emb_pool, dim=0)
        cov = torch.cov(emb_pool.T)
        return mean, cov

    def wasserstein_distance(self, mean1, cov1, mean2, cov2):
        # 均值项：欧几里得距离平方
        mean_diff = torch.sum((mean1 - mean2) ** 2)
        # 协方差项：基于特征值差异
        eigvals1, _ = torch.linalg.eigh(cov1)
        eigvals2, _ = torch.linalg.eigh(cov2)
        cov_term = torch.sum((torch.sqrt(torch.clamp(eigvals1, min=1e-24)) -
                              torch.sqrt(torch.clamp(eigvals2, min=1e-24))) ** 2)
        return mean_diff + cov_term

    def forward(self, emb, emb2):
        """
        前向传播，计算 ELBO 损失。为了降低计算复杂度和显存占用：
          1. 对输入的 3D 特征（[batch, seq_len, feat_dim]）先进行平均池化，得到全局特征 [batch, feat_dim]；
             拼接两个增强视图，得到 [2*batch, feat_dim]。
          2. 聚类部分基于全局特征与原型计算相似度，采用负熵作为聚类损失。
          3. 对比部分采用 NT-Xent 损失，在计算相似度矩阵前对特征进行归一化。
        """
        # 拼接两个增强视图，形状：[2B, seq_len, feat_dim]
        features = torch.cat((emb, emb2), dim=0)
        # 对序列维度做平均池化，得到全局特征：[2B, feat_dim]
        features_pool = torch.mean(features, dim=1)

        # ----------------------- 聚类损失 -----------------------
        # 计算全局特征与聚类原型之间的点积，形状：[2B, num_cluster]
        anchor_dot_cluster = torch.clamp(torch.matmul(features_pool, self.prototype.T), min=-100, max=100)
        pi_logit = anchor_dot_cluster / self.kappa
        log_pi = self.logSoftmax(pi_logit + 1e-6)
        pi = torch.exp(log_pi)
        # 使用负熵作为损失（熵越低越好）
        loss_cluster = - torch.mean(torch.sum(pi * log_pi, dim=1))

        # ----------------------- 对比损失（NT-Xent） -----------------------
        # 对全局特征先进行归一化
        features_norm = F.normalize(features_pool, p=2, dim=1)
        # 计算归一化后特征的相似度矩阵，形状：[2B, 2B]
        sim_matrix = torch.matmul(features_norm, features_norm.T) / self.tau
        # 将对角线（自相似）置为一个很小的值，防止影响 softmax
        mask = torch.eye(sim_matrix.size(0), device=self.device).bool()
        sim_matrix.masked_fill_(mask, -1e9)

        # 假设 emb 与 emb2 中相同索引的样本为正样本对，
        # 则前 B 个样本的正样本为索引 i+B，后 B 个样本的正样本为索引 i-B。
        B = emb.size(0)
        N = features_norm.size(0)  # 2B
        targets = torch.arange(B, device=self.device)
        targets = torch.cat([targets + B, targets])

        log_prob = F.log_softmax(sim_matrix, dim=1)
        loss_contrast = - log_prob[torch.arange(N), targets].mean()

        # 最终损失由对比损失和聚类损失构成
        loss_final = loss_contrast + self.eta * loss_cluster

        return loss_final

    # def forward(self, emb, emb2):
    #     """
    #     前向传播，计算 ELBO 损失。为降低计算复杂度和显存占用：
    #       1. 对输入的 3D 特征（[batch, seq_len, feat_dim]）先进行平均池化，得到每个样本的全局表示，
    #          形状变为 [batch, feat_dim]；然后拼接两个增强视图，得到 [2*batch, feat_dim]。
    #       2. 聚类部分基于全局特征与原型计算相似度，得到聚类对齐损失。
    #       3. 对比部分采用常用的 NT-Xent 损失，计算 [2*batch, 2*batch] 的相似度矩阵。
    #     """
    #     # 拼接两个增强视图，形状：[2*B, seq_len, feat_dim]
    #     features = torch.cat((emb, emb2), dim=0)
    #     # 对序列维度做平均池化，得到全局特征：[2*B, feat_dim]
    #     features_pool = torch.mean(features, dim=1)
    #
    #     # ----------------------- 聚类损失 -----------------------
    #     # 计算全局特征与聚类原型之间的点积，形状：[2B, num_cluster]
    #     anchor_dot_cluster = torch.clamp(torch.matmul(features_pool, self.prototype.T), min=-100, max=100)
    #     pi_logit = anchor_dot_cluster / self.kappa
    #     log_pi = self.logSoftmax(pi_logit + 1e-6)
    #     pi = torch.exp(log_pi)
    #     loss_cluster = torch.mean(torch.sum(pi * log_pi, dim=1))
    #
    #     # ----------------------- 对比损失（NT-Xent） -----------------------
    #     # 计算相似度矩阵，形状：[2B, 2B]
    #     sim_matrix = torch.matmul(features_pool, features_pool.T) / self.tau
    #     # 将对角线（自相似）置为很小的值，防止影响计算
    #     mask = torch.eye(sim_matrix.size(0), device=self.device).bool()
    #     sim_matrix.masked_fill_(mask, -1e9)
    #
    #     # 假设 emb 与 emb2 中相同索引的样本为正样本对，
    #     # 则对于前 B 个样本，其正样本为索引 i+B；对于后 B 个样本，其正样本为索引 i-B。
    #     B = emb.size(0)
    #     N = features_pool.size(0)  # 2B
    #     targets = torch.arange(B, device=self.device)
    #     targets = torch.cat([targets + B, targets])
    #
    #     log_prob = F.log_softmax(sim_matrix, dim=1)
    #     loss_contrast = -log_prob[torch.arange(N), targets].mean()
    #
    #     # 最终损失由对比损失和聚类损失构成
    #     loss_final = loss_contrast + self.eta * loss_cluster
    #
    #     return loss_final



# # 3 维  占用较大的显存空间。
# class ELBO(nn.Module):
#     def __init__(self, num_cluster, feat_dim, tau, kappa, eta, device):
#         super(ELBO, self).__init__()
#         print("--------------------------Initializing -----------------------------------")
#         self.num_cluster = num_cluster
#         self.feat_dim = feat_dim
#         self.tau = tau
#         self.kappa = kappa
#         self.eta = eta
#         self.device = device
#
#         # 初始化聚类原型，形状为 [num_cluster, feat_dim]
#         self.prototype = nn.Parameter(torch.nn.init.uniform_(torch.Tensor(self.num_cluster, self.feat_dim), a=0, b=1))
#         self.logSoftmax = torch.nn.LogSoftmax(dim=1)
#         print("----------------------Initialization Ends-----------------------------------")
#
#     def update_cluster(self, emb):
#         """
#         更新聚类中心，输入 emb 为 3 维张量 [batch, seq_len, feat_dim]，
#         展平后每个时间步均作为一个样本参与聚类更新。
#         """
#         B, T, D = emb.shape
#         emb_flat = emb.view(B * T, D)
#
#         # 计算每个样本与每个聚类中心的相似度
#         dist = torch.matmul(emb_flat, self.prototype.T)  # [B*T, num_cluster]
#         # 获取每个样本的聚类分配（选择相似度最大的聚类中心）
#         cluster_assignments = torch.argmax(dist, dim=1)  # [B*T]
#
#         new_center = torch.zeros(self.num_cluster, self.feat_dim, device=self.device)
#         for i in range(self.num_cluster):
#             cluster_points = emb_flat[cluster_assignments == i]
#             if cluster_points.shape[0] > 0:
#                 new_center[i] = torch.mean(cluster_points, dim=0)
#
#         print("Prototype shape:", self.prototype.shape)
#         print("New center shape:", new_center.shape)
#
#         # 更新聚类中心
#         self.prototype.data = new_center
#
#     def compute_mean_cov(self, emb):
#         """
#         计算给定特征向量的均值和协方差矩阵，
#         输入 emb 为三维张量 [batch, seq_len, feat_dim]，展平后计算。
#         """
#         B, T, D = emb.shape
#         emb_flat = emb.view(B * T, D)
#         mean = torch.mean(emb_flat, dim=0)
#         cov = torch.cov(emb_flat.T)
#         return mean, cov
#
#     def wasserstein_distance(self, mean1, cov1, mean2, cov2):
#         # 均值项：欧几里得距离平方
#         mean_diff = torch.sum((mean1 - mean2) ** 2)
#         # 协方差项：特征值差异
#         eigvals1, _ = torch.linalg.eigh(cov1)
#         eigvals2, _ = torch.linalg.eigh(cov2)
#         cov_term = torch.sum((torch.sqrt(torch.clamp(eigvals1, min=1e-24)) -
#                               torch.sqrt(torch.clamp(eigvals2, min=1e-24))) ** 2)
#         return mean_diff + cov_term
#
#     def forward(self, emb, emb2):
#         """
#         前向传播，计算 ELBO 损失。
#         输入：
#             emb, emb2: 两组增强后的嵌入，形状均为 [batch, seq_len, feat_dim]
#         """
#         # 拼接两个增强后的嵌入，得到形状 [2*batch, seq_len, feat_dim]
#         features = torch.cat((emb, emb2), dim=0)
#         B, T, D = features.shape
#         N = B * T  # 总样本数（每个时间步视为一个样本）
#
#         # 将三维特征展平为二维 [N, feat_dim]
#         features_flat = features.view(N, D)
#
#         # 计算特征与聚类原型之间的点积，形状 [N, num_cluster]
#         anchor_dot_cluster = torch.clamp(torch.matmul(features_flat, self.prototype.T), min=-100, max=100)
#         # 计算对比样本间的点积，形状 [N, N]
#         anchor_dot_contrast = torch.clamp(torch.matmul(features_flat, features_flat.T), min=-100, max=100)
#
#         # 聚类损失（原型对齐）
#         pi_logit = torch.div(anchor_dot_cluster, self.kappa)
#         log_pi = self.logSoftmax(pi_logit + 1e-6)
#         pi = torch.exp(log_pi)  # 聚类分布 p(c|v)
#         loss_0 = torch.mean(torch.sum(pi * log_pi, dim=1))
#
#         # 调整维度以便对齐计算
#         # 将 anchor_dot_cluster 调整为形状 [num_cluster, N, 1]
#         align_cluster = anchor_dot_cluster.T.view(self.num_cluster, N, 1).repeat(1, 1, N)
#         # 将 anchor_dot_contrast 调整为形状 [num_cluster, N, N]
#         align_contrast = anchor_dot_contrast.repeat(self.num_cluster, 1).view(self.num_cluster, N, N)
#
#         # 防止指数计算过大，先进行数值裁剪
#         exp_align_cluster = torch.exp(torch.clamp(align_cluster, min=-50, max=50))
#         exp_align_contrast = torch.exp(torch.clamp(align_contrast, min=-50, max=50))
#         weight_denom = exp_align_cluster + exp_align_contrast + 1e-6
#         weight1 = exp_align_cluster / weight_denom
#         weight2 = exp_align_contrast / weight_denom
#
#         # 计算增强后的对齐分数，并加入温度参数 self.tau
#         anchor_dot_augmentation = (weight1 * align_cluster + weight2 * align_contrast) / self.tau + 1e-6
#
#         if anchor_dot_augmentation.size(2) > 0:
#             logits_max, _ = torch.max(anchor_dot_augmentation, dim=2, keepdim=True)
#             logits = anchor_dot_augmentation - logits_max.detach()
#         else:
#             logits = anchor_dot_augmentation
#
#         # 构造 mask，屏蔽样本自身的对比项
#         logits_mask = torch.ones_like(anchor_dot_contrast)
#         logits_mask = logits_mask.scatter(1, torch.arange(N).view(-1, 1).to(self.device), 0)
#         logits_mask = logits_mask.unsqueeze(0).repeat(self.num_cluster, 1, 1)
#         mask = logits_mask
#
#         exp_logits = torch.exp(logits) * logits_mask
#         sum_exp_logits = torch.clamp(exp_logits.sum(2, keepdim=True), min=1e-6)
#         log_logits = logits - torch.log(sum_exp_logits)
#         normalized_logits = torch.exp(log_logits)
#
#         log_logits_pos = log_logits * mask
#         normalized_logits_pos = normalized_logits * mask
#
#         pi_normalized_logits_pos = pi.T.view(self.num_cluster, N, 1) * normalized_logits_pos
#         denom = torch.sum(pi_normalized_logits_pos, 0) + (1 - mask)
#         denom = torch.clamp(denom, min=1e-6)
#         posterior = torch.div(pi_normalized_logits_pos, denom)
#         posterior = posterior * mask
#
#         term = log_pi.T.view(self.num_cluster, N, 1) + log_logits_pos - torch.log(torch.clamp(posterior, min=1e-6))
#         loss = -torch.mean(torch.div(torch.sum(torch.sum(posterior * term, dim=0), dim=1),
#                                      torch.clamp(torch.sum(mask, dim=1), min=1e-6)))
#
#         # 最终损失：对比损失 + self.eta * 聚类损失
#         loss_final = loss + self.eta * loss_0
#
#         return loss_final





#  二维的
# class ELBO(nn.Module):
#     def __init__(self, num_cluster, feat_dim, tau, kappa, eta, device):
#         super(ELBO, self).__init__()
#
#         print("--------------------------Initializing -----------------------------------")
#         self.num_cluster = num_cluster
#         self.feat_dim = feat_dim
#         self.tau = tau
#         self.kappa = kappa
#         self.eta = eta
#         self.device = device
#
#         self.prototype = nn.Parameter(torch.nn.init.uniform_(torch.Tensor( self.num_cluster,self.feat_dim), a=0, b=1))
#         self.logSoftmax = torch.nn.LogSoftmax(dim=1)
#         print("----------------------Initialization Ends-----------------------------------")
#
#     def update_cluster(self, emb):
#         # 无监督情况下，通过嵌入来计算新的聚类中心
#
#             # 计算每个样本与每个聚类中心的距离
#         dist = torch.matmul(emb, self.prototype.T)  # BS x M
#         # 获取每个样本的聚类分配（选择最相似的聚类中心）
#         cluster_assignments = torch.argmax(dist, dim=1)  # BS, 聚类分配
#
#         # 为每个聚类中心计算新的样本
#         new_center = torch.zeros(self.num_cluster, self.feat_dim).to(self.device)
#         for i in range(self.num_cluster):
#             cluster_points = emb[cluster_assignments == i]
#             if len(cluster_points) > 0:
#                 new_center[i] = torch.mean(cluster_points, dim=0)
#
#         print("Prototype shape:", self.prototype.shape)
#         print("New center shape:", new_center.shape)
#
#         # 更新聚类中心
#         self.prototype.data = new_center
#
#     def compute_mean_cov(self, emb):
#         mean = torch.mean(emb, dim=0)  # 计算均值
#         cov = torch.cov(emb.T)  # 计算协方差矩阵
#         return mean, cov
#
#     def wasserstein_distance(self, mean1, cov1, mean2, cov2):
#         # 均值项：欧几里得距离平方
#         mean_diff = torch.sum((mean1 - mean2) ** 2)
#
#         # 协方差项：使用特征值分解进行计算
#         eigvals1, _ = torch.linalg.eigh(cov1)
#         eigvals2, _ = torch.linalg.eigh(cov2)
#
#         # 计算 Wasserstein 距离的协方差项
#         cov_term = torch.sum((torch.sqrt(torch.clamp(eigvals1, min=1e-24)) -
#                               torch.sqrt(torch.clamp(eigvals2, min=1e-24))) ** 2)
#
#         return mean_diff + cov_term
#
#
#
#     def compute_mean_cov(self, emb):
#         """
#         计算给定特征向量的均值和协方差矩阵
#         :param emb: 形状 (batch_size, feature_dim) 的张量
#         :return: (均值, 协方差矩阵)
#         """
#         mean = torch.mean(emb, dim=0)  # 按行计算均值，形状: (feature_dim,)
#
#         # 计算协方差矩阵 (feature_dim, feature_dim)
#         cov = torch.cov(emb.T)  # 需要转置，使得形状符合 torch.cov 的要求
#
#         return mean, cov
#
#     def forward(self, emb, emb2):
#         # 无监督情况下，拼接两个增强的嵌入
#         features = torch.cat((emb, emb2), dim=0)
#         batchSize = features.shape[0]
#
#         # 计算特征与聚类原型之间的点积，并限制数值范围以防爆炸
#         anchor_dot_cluster = torch.clamp(torch.matmul(features, self.prototype.T), min=-100, max=100)  # [BS, M]
#         anchor_dot_contrast = torch.clamp(torch.matmul(features, features.T), min=-100, max=100)  # [BS, BS]
#
#         # 聚类损失（原型对齐）
#         pi_logit = torch.div(anchor_dot_cluster, self.kappa)
#         # 在 logSoftmax 前加上 epsilon 防止 log(0)
#         log_pi = self.logSoftmax(pi_logit + 1e-6)  # [BS, M]
#         pi = torch.exp(log_pi)  # 聚类分布 p(c|v)
#         loss_0 = torch.mean(torch.sum(pi * log_pi, dim=1))
#
#         # 对比增强样本的正负对齐
#         # 将 anchor_dot_cluster 调整为形状 [M, BS, 1]
#         align_cluster = anchor_dot_cluster.T.view(self.num_cluster, batchSize, 1).repeat(1, 1, batchSize)
#         # 将 anchor_dot_contrast 调整为形状 [M, BS, BS]
#         align_contrast = anchor_dot_contrast.repeat(self.num_cluster, 1).view(self.num_cluster, batchSize, batchSize)
#
#         # 为防止指数计算过大，先对 align_cluster 和 align_contrast 限制范围
#         exp_align_cluster = torch.exp(torch.clamp(align_cluster, min=-50, max=50))
#         exp_align_contrast = torch.exp(torch.clamp(align_contrast, min=-50, max=50))
#         weight_denom = exp_align_cluster + exp_align_contrast + 1e-6
#         weight1 = exp_align_cluster / weight_denom
#         weight2 = exp_align_contrast / weight_denom
#
#         # 计算增强后的对齐分数，并加入 epsilon 保护
#         anchor_dot_augmentation = (weight1 * align_cluster + weight2 * align_contrast) / self.tau + 1e-6
#
#         # 计算 logits，减去每行最大值以稳定数值
#         if anchor_dot_augmentation.size(2) > 0:
#             logits_max, _ = torch.max(anchor_dot_augmentation, dim=2, keepdim=True)
#             logits = anchor_dot_augmentation - logits_max.detach()
#         else:
#             logits = anchor_dot_augmentation
#
#         # 构造 mask，用于屏蔽样本自身的对比
#         logits_mask = torch.ones_like(anchor_dot_contrast)
#         logits_mask = logits_mask.scatter(1, torch.arange(batchSize).view(-1, 1).to(self.device), 0)
#         # 扩展 mask 到形状 [M, BS, BS]
#         logits_mask = logits_mask.unsqueeze(0).repeat(self.num_cluster, 1, 1)
#         mask = logits_mask
#
#         # 计算 exp(logits) 后求和，再进行 log 运算时加入 clamp 防止 log(0)
#         exp_logits = torch.exp(logits) * logits_mask
#         sum_exp_logits = torch.clamp(exp_logits.sum(2, keepdim=True), min=1e-6)
#         log_logits = logits - torch.log(sum_exp_logits)
#         normalized_logits = torch.exp(log_logits)
#
#         # 计算正样本对的对数概率
#         log_logits_pos = log_logits * mask
#         normalized_logits_pos = normalized_logits * mask
#
#         # 计算后验概率 q(c|v,s)
#         pi_normalized_logits_pos = pi.T.view(self.num_cluster, batchSize, 1) * normalized_logits_pos
#         denom = torch.sum(pi_normalized_logits_pos, 0) + (1 - mask)
#         denom = torch.clamp(denom, min=1e-6)
#         posterior = torch.div(pi_normalized_logits_pos, denom)
#         posterior = posterior * mask
#
#         # 计算损失项，其中对 posterior 加入 clamp 防止 log(0)
#         term = log_pi.T.view(self.num_cluster, batchSize, 1) + log_logits_pos - torch.log(
#             torch.clamp(posterior, min=1e-6))
#         loss = -torch.mean(torch.div(torch.sum(torch.sum(posterior * term, dim=0), dim=1),
#                                      torch.clamp(torch.sum(mask, dim=1), min=1e-6)))
#
#         # 最终损失：对比损失 + self.eta * 聚类损失
#         loss_final = loss + self.eta * loss_0
#
#         return loss_final
#

#
# class ELBO(nn.Module):
#     def __init__(self, num_cluster, feat_dim, tau, kappa, eta, device):
#         super(ELBO, self).__init__()
#
#         print("--------------------------Initializing -----------------------------------")
#
#         self.num_cluster = num_cluster
#         self.feat_dim = feat_dim
#         self.tau = tau
#         self.kappa = kappa
#         self.eta = eta
#         self.device = device
#
#         self.logSoftmax = torch.nn.LogSoftmax(dim=1)
#         print("----------------------Initialization Ends-----------------------------------")
#
#     def forward(self, emb, emb2, y, cen):
#         features = torch.cat((emb, emb2), dim=0)
#         batchSize = features.shape[0]
#         y = y.contiguous().view(-1, 1)
#         mask = torch.eq(y, y.T).float().to(self.device)
#         mask = mask.repeat(2, 2)
#
#         anchor_dot_cluster = torch.matmul(features, cen.T)  # BS x M
#         anchor_dot_contrast = torch.matmul(features, features.T)  # BS x BS
#
#         # clusterscl loss
#         pi_logit = torch.div(anchor_dot_cluster, self.kappa)
#         log_pi = self.logSoftmax(pi_logit + 1e-18)  # BS x M
#         pi = torch.exp(log_pi)  # cluster distribution p(c | v), BS x M
#
#         loss_0 = torch.mean(torch.sum(pi * log_pi, dim=1))
#
#         # compute the alignment with the augmented positives and negatives
#         align_cluster = anchor_dot_cluster.T.view(self.num_cluster, batchSize, 1).repeat(1, 1, batchSize)
#         align_contrast = anchor_dot_contrast.repeat(self.num_cluster, 1).view(self.num_cluster, batchSize, batchSize)
#         weight1 = torch.div(torch.exp(align_cluster), (torch.exp(align_cluster) + torch.exp(align_contrast)))
#         weight2 = torch.div(torch.exp(align_contrast), (torch.exp(align_cluster) + torch.exp(align_contrast)))
#
#         anchor_dot_augmentation = (weight1 * align_cluster + weight2 * align_contrast) / self.tau + 1e-18  # M x BS x BS
#
#         logits_max, _ = torch.max(anchor_dot_augmentation, dim=2, keepdim=True)
#         logits = anchor_dot_augmentation - logits_max.detach()
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batchSize).view(-1, 1).to(self.device),
#             0
#         )
#         # set the diagonal elements to be 0
#         mask = mask * logits_mask
#
#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_logits = logits - torch.log(exp_logits.sum(2, keepdim=True))
#         normalized_logits = torch.exp(log_logits)  # p(s | v, c)
#         # remain the prob of each anchor's positives
#         log_logits_pos = torch.mul(log_logits, mask)
#         normalized_logits_pos = torch.mul(normalized_logits, mask)
#         del log_logits
#         del normalized_logits
#
#         pi_normalized_logits_pos = pi.T.view(self.num_cluster, batchSize, 1) * normalized_logits_pos
#         posterior = torch.div(pi_normalized_logits_pos, torch.add(torch.sum(pi_normalized_logits_pos, 0), 1 - mask))
#         posterior = torch.mul(posterior, mask)  # q(c | v, s)
#
#         loss = -torch.mean(torch.div(torch.sum(torch.sum(
#             posterior * (log_pi.T.view(self.num_cluster, batchSize, 1) + log_logits_pos - torch.log(posterior + 1e-18)),
#             0), 1), torch.sum(mask, 1)))
#
#         loss_final = loss + self.eta * loss_0
#
#         return loss_final

    # def update_cluster(self, new_center,num_cluster):
    #
    #     new_center = new_center.to(self.device)
    #     with torch.no_grad():
    #         out_ids = torch.arange(self.num_cluster).to(self.device)
    #         out_ids = out_ids.long()  # BS x 1
    #         self.prototype.index_copy_(1, out_ids, new_center.T)

    # def forward(self, emb, emb2, y, num_cluster):
    #
    #     features = torch.cat((emb, emb2), dim=0)
    #     batchSize = features.shape[0]
    #     y = y.contiguous().view(-1, 1)
    #     mask = torch.eq(y, y.T).float().to(self.device)
    #     mask = mask.repeat(2, 2)
    #
    #     anchor_dot_cluster = torch.matmul(features, num_cluster.T)  # BS x M
    #     anchor_dot_contrast = torch.matmul(features, features.T)  # BS x BS
    #
    #     # clusterscl loss
    #     pi_logit = torch.div(anchor_dot_cluster, self.kappa)
    #     log_pi = self.logSoftmax(pi_logit + 1e-18)  # BS x M
    #     pi = torch.exp(log_pi)  # cluster distribution p(c | v), BS x M
    #
    #     loss_0 = torch.mean(torch.sum(pi * log_pi, dim=1))
    #
    #
    #     align_cluster = anchor_dot_cluster.T.view(10, batchSize, 1).repeat(1, 1, batchSize)
    #
    #     align_contrast = anchor_dot_contrast.repeat(10, 1).view(10, batchSize, batchSize)
    #     weight1 = torch.div(torch.exp(align_cluster), (torch.exp(align_cluster) + torch.exp(align_contrast)))
    #     weight2 = torch.div(torch.exp(align_contrast), (torch.exp(align_cluster) + torch.exp(align_contrast)))
    #
    #     anchor_dot_augmentation = (weight1 * align_cluster + weight2 * align_contrast) / self.tau + 1e-18  # M x BS x BS
    #
    #     logits_max, _ = torch.max(anchor_dot_augmentation, dim=2, keepdim=True)
    #     logits = anchor_dot_augmentation - logits_max.detach()
    #     # mask-out self-contrast cases
    #     logits_mask = torch.scatter(
    #         torch.ones_like(mask),
    #         1,
    #         torch.arange(batchSize).view(-1, 1).to(self.device),
    #         0
    #     )
    #     # set the diagonal elements to be 0
    #     mask = mask * logits_mask
    #
    #     # compute log_prob
    #     exp_logits = torch.exp(logits) * logits_mask
    #     log_logits = logits - torch.log(exp_logits.sum(2, keepdim=True))
    #     normalized_logits = torch.exp(log_logits)  # p(s | v, c)
    #     # remain the prob of each anchor's positives
    #     log_logits_pos = torch.mul(log_logits, mask)
    #     normalized_logits_pos = torch.mul(normalized_logits, mask)
    #     del log_logits
    #     del normalized_logits
    #
    #     pi_normalized_logits_pos = pi.T.view(10, batchSize, 1) * normalized_logits_pos
    #     posterior = torch.div(pi_normalized_logits_pos, torch.add(torch.sum(pi_normalized_logits_pos, 0), 1 - mask))
    #     posterior = torch.mul(posterior, mask)  # q(c | v, s)
    #
    #     loss = -torch.mean(torch.div(torch.sum(torch.sum(
    #         posterior * (log_pi.T.view(10, batchSize, 1) + log_logits_pos - torch.log(posterior + 1e-18)),
    #         0), 1), torch.sum(mask, 1)))
    #
    #     loss_final = loss + self.eta * loss_0
    #
    #     return loss_final

class RecurCluster(nn.Module):
    def __init__(self, hidden_size=128, head_num=8, layer_idx=2):
        super().__init__()

        self.head_num = head_num
        self.hidden_size = hidden_size       # 输入维度
        self.head_size = hidden_size // head_num  # 输出维度

        # split qkv
        self.q1_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q2_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k1_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k2_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=False)  # V projects to 2 * n_embd

        self.c_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.attn_dropout = torch.nn.Dropout(p=0.15)
        self.resid_dropout = torch.nn.Dropout(p=0.15)
        self.subln = nn.LayerNorm(2 * self.head_size, elementwise_affine=False)
        self.lambda_init = lambda_init(layer_idx)

        # Init λ across heads
        self.lambda_q1 = nn.Parameter(torch.randn(head_num , self.head_size) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(head_num , self.head_size) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(head_num , self.head_size) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(head_num, self.head_size) * 0.1)

    def forward(self, x):
        # 将 llm 降维后的特征 ( id ) 进行聚类
        num_clusters = 10  # 定义聚类中心数量
        embedding_size = 64  # 嵌入向量的维度
        batch_size = x.size(0)  # 获取批次大小

        centroids = torch.rand(num_clusters, embedding_size, device=x.device)  # 初始化聚类中心

        # 为每个批次同时计算聚类
        for j in range(10):
            # 计算每个样本到聚类中心的距离
            dists = torch.cdist(x.view(-1, embedding_size), centroids)  # [batch_size * T, num_clusters]
            dists = torch.sigmoid(dists)  # 应用 sigmoid 函数

            # 得到每个样本最接近的聚类中心
            _, idxs = dists.min(dim=1)  # idxs: [batch_size * T]

            # 重新计算聚类中心
            new_centroids = torch.zeros(num_clusters, embedding_size, device=x.device)
            counts = torch.zeros(num_clusters, 1, device=x.device)

            # 累加所有样本到对应聚类中心的距离
            idxs = idxs.view(batch_size, 200)  # 变回 [batch_size, T]
            for b in range(batch_size):
                for t in range(200):
                    cluster_idx = idxs[b, t]
                    new_centroids[cluster_idx] += x[b, t]  # 聚类中心更新
                    counts[cluster_idx] += 1  # 聚类数量更新

            # 更新聚类中心
            centroids = new_centroids / counts.clamp(min=1)  # 避免除以零



        return 0



def lambda_init(depth):
    return 0.7 - 0.6 * math.exp(-0.3 * (depth - 1))


# Multi-head Differential Attention
# batch_size 128
# head_num 8
class MultiHeadDiffAttention(nn.Module):
    def __init__(self, hidden_size, head_num, droupoutd, layer_idx=2):
        super().__init__()
        assert hidden_size % head_num == 0
        # self.n_head = n_head
        # self.head_size = n_embd // n_head
        # self.lambda_init = lambda_init(layer_idx)
        self.head_num = head_num
        self.hidden_size = hidden_size
        self.head_size = hidden_size // head_num   # 128/8 = 16
        self.lambda_init = lambda_init(layer_idx)

        # split qkv
        self.q1_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q2_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k1_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k2_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=False)  # V projects to 2 * n_embd

        self.c_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.attn_dropout = torch.nn.Dropout(droupoutd)
        self.resid_dropout = torch.nn.Dropout(droupoutd)
        self.subln = nn.LayerNorm(2 * self.head_size, elementwise_affine=False)

        self.W_o = nn.Linear(head_num * self.head_size, hidden_size)

        # Init λ across heads
        self.lambda_q1 = nn.Parameter(torch.randn(head_num , self.head_size) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(head_num , self.head_size) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(head_num , self.head_size) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(head_num, self.head_size) * 0.1)



    def forward(self, x):

        # torch.Size([128, 200, 64])
        B, T, C = x.shape
        # O = GroupNorm([self.diff_attention(x) for i in range(4)])
        # print(O)
        # O = O * (1 -self.lambda_init)
        # c_o = concat(O) @ self.W_o
        # # print("c_o")
        # # print(c_o.shape)
        # Diff 主要需要的变量：
        # W_q:64*64 线性投影矩阵。
        # W_k:64*64 线性投影矩阵。
        # W_v:64*64 线性投影矩阵。
        # W_o:64*64 线性投影矩阵。
        # Q1:(128, 200, 32): x 经过线性变换投影后得到的查询(Query, Q)。
        # K1:(128, 200, 32): x 经过线性变换投影后得到的键(Key, K)。
        # V:(128, 200, 64)
        # Att1、Att2: (128, 200, 200)
        # 一个注意力头 Diff()

        # Project x to get q1, q2, k1, k2, v
        q1 = self.q1_proj(x).view(B, T, self.head_num, self.head_size).transpose(1, 2)  # (128, 8, 200, 16)

        # print("q1.shape")
        q2 = self.q2_proj(x).view(B, T, self.head_num, self.head_size).transpose(1, 2)  # (128, 8, 200, 16)
        k1 = self.k1_proj(x).view(B, T, self.head_num, self.head_size).transpose(1, 2)  # (128, 8, 200, 16)
        k2 = self.k2_proj(x).view(B, T, self.head_num, self.head_size).transpose(1, 2)  # (128, 8, 200, 16)
        v = self.v_proj(x).view(B, T, self.head_num, 2 * self.head_size).transpose(1, 2)  # (128, 8, 200, 32)

        scale = 1.0 / math.sqrt(self.head_size)
        #  交换 k1 的倒数第二个维度（-2）和最后一个维度（-1） 得到 128 * 8 * 16 * 200
        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale  # 128 * 8 * 200 * 200
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale   # 128 * 8 * 200 * 200

        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # 1 * 1 * 200 * 200
        # 将 att1 和 att2 中值为0 的替换为负无穷。
        att1 = att1.masked_fill(attn_mask == 0, float('-inf'))
        att2 = att2.masked_fill(attn_mask == 0, float('-inf'))

        # 归一化
        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)

        # Compute λ for each head separately
        # [4, 1, 1]
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        att = att1 - lambda_full * att2
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)  # [128, 4, 200, 64]

        # 归一化，缩放。
        y = self.subln(y)
        y = y * (1 - self.lambda_init)

        y = y.transpose(1, 2).contiguous().view(B, T, 2 * C)  # 128 * 200 * 128
        # self.c_proj(y) # 128 * 200 * 128
        y = self.resid_dropout(self.c_proj(y))
        # 128 * 200 * 128

        return y
    def diff_attention(self, x):
        """
        q, k, v: [B, T, C]
        attn_mask: [T, T]
        key_padding_mask: [B, T]
        """
        # print(x.shape) torch.Size([128, 200, 64])
        B, T, C = x.shape
        # Diff 主要需要的变量：
        # W_q:64*64 线性投影矩阵。
        # W_k:64*64 线性投影矩阵。
        # W_v:64*64 线性投影矩阵。
        # W_o:64*64 线性投影矩阵。
        # Q1:(128, 200, 32): x 经过线性变换投影后得到的查询(Query, Q)。
        # K1:(128, 200, 32): x 经过线性变换投影后得到的键(Key, K)。
        # V:(128, 200, 64)
        # Att1、Att2: (128, 200, 200)
        # 一个注意力头 Diff()

        # Project x to get q1, q2, k1, k2, v
        q1 = self.q1_proj(x).view(B, T, self.head_num, self.head_size).transpose(1, 2)  # (128, 4, 200, 32)
        print(q1.shape)
        # print("q1.shape")
        q2 = self.q2_proj(x).view(B, T, self.head_num, self.head_size).transpose(1, 2)  # (128, 4, 200, 32)
        k1 = self.k1_proj(x).view(B, T, self.head_num, self.head_size).transpose(1, 2)  # (128, 4, 200, 32)
        k2 = self.k2_proj(x).view(B, T, self.head_num, self.head_size).transpose(1, 2)  # (128, 4, 200, 32)
        v = self.v_proj(x).view(B, T, self.head_num, 2 * self.head_size).transpose(1, 2)  # (128, 4, 200, 64)

        scale = 1.0 / math.sqrt(self.head_size)
        #  交换 k1 的倒数第二个维度（-2）和最后一个维度（-1） 得到 128 * 4 * 32 * 200
        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale  # 128 * 4 * 200 * 200
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale  # 128 * 4 * 200 * 200

        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # 1 * 1 * 200 * 200
        # 将 att1 和 att2 中值为0 的替换为负无穷。
        att1 = att1.masked_fill(attn_mask == 0, float('-inf'))
        att2 = att2.masked_fill(attn_mask == 0, float('-inf'))

        # 归一化
        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)

        # Compute λ for each head separately
        # [4, 1, 1]
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        att = att1 - lambda_full * att2
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)  # [128, 4, 200, 64]

        # 归一化，缩放。
        y = self.subln(y)
        y = y * (1 - self.lambda_init)

        y = y.transpose(1, 2).contiguous().view(B, T, 2 * C)  # 128 * 200 * 128
        # self.c_proj(y) # 128 * 200 * 128
        y = self.resid_dropout(self.c_proj(y))
        # 128 * 200 * 128
        return y



import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttention(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim):
        super(CoAttention, self).__init__()

        self.attention_proj = nn.Linear(hidden_dim, 1)

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


    def forward(self, text_features, image_features):
        """
        :param text_features: (batch_size, num_text_tokens, text_dim)
        :param image_features: (batch_size, num_image_regions, image_dim)
        :return: attended_text_features, attended_image_features
        """
        batch_size, seq_len, dim = text_features.shape
        # 投影到同一维度
        text_transformed = torch.sigmoid(self.text_proj(text_features))  # (B, T, H)
        image_transformed = torch.sigmoid(self.image_proj(image_features))  # (B, I, H)

        # 计算注意力分数
        attention_scores = torch.bmm(text_transformed, image_transformed.transpose(1, 2))  # (B, T, I)

        # 创建对角线掩码
        mask = torch.eye(seq_len, dtype=torch.bool, device=attention_scores.device)  # (seq_len, seq_len)
        mask = mask.unsqueeze(0)  # 扩展 batch 维度 (1, seq_len, seq_len)

        # 将对角线元素设为 -inf
        attention_scores.masked_fill_(mask, float('-inf'))



        # 计算基于文本的图像注意力
        text_to_image_attention = F.softmax(attention_scores, dim=-1)  # (B, T, I)
        attended_image_features = torch.bmm(text_to_image_attention, image_features)  # (B, T, image_dim)

        return attended_image_features






class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)


    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    

    def forward(self,x,y,log_seqs):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为k和v的输入
        """

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # attention_mask = attention_mask.eq(0)
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output


import torch
import torch.nn as nn
import torch.nn.functional as F


class EBMAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, temperature=1.0, use_learnable_temp=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 定义Query/Key/Value的投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 能量模型的参数
        self.temperature = temperature
        self.use_learnable_temp = use_learnable_temp
        if use_learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))

        # 能量函数中的可学习参数（示例：简单的双线性变换）
        self.energy_W = nn.Parameter(torch.randn(embed_dim, embed_dim))


    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 投影到 Q/K/V
        q = self.q_proj(query)  # [B, L, D]
        k = self.k_proj(key)  # [B, S, D]
        v = self.v_proj(value)  # [B, S, D]

        # 切分多头
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, d]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, d]

        # 计算能量（优化 einsum）
        k_transformed = torch.matmul(k, self.energy_W)  # [B, H, S, d]
        energy = torch.matmul(q, k_transformed.transpose(-2, -1))  # [B, H, L, S]

        # 温度缩放
        temperature = torch.exp(self.log_temp) if self.use_learnable_temp else self.temperature
        energy = energy / temperature

        # 处理掩码
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        # 计算注意力
        attn_weights = F.softmax(-energy, dim=-1)  # [B, H, L, S]

        # 加权求和
        output = torch.matmul(attn_weights, v)  # [B, H, L, d]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return output, attn_weights



    # def forward(self, query, key, value, mask=None):
    #     batch_size = query.size(0)
    #
    #     # 投影得到Q/K/V
    #     q = self.q_proj(query)  # [B, L, D]
    #     k = self.k_proj(key)  # [B, S, D]
    #     v = self.v_proj(value)  # [B, S, D]
    #
    #     # 切分多头
    #     q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
    #     k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, d]
    #     v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, d]
    #
    #     # 计算能量（双线性形式）
    #     energy = torch.einsum("bhqd,bhkd->bhqk", q, torch.matmul(k, self.energy_W))  # [B, H, L, S]
    #
    #     # 应用温度参数
    #     if self.use_learnable_temp:
    #         temperature = torch.exp(self.log_temp)
    #     else:
    #         temperature = self.temperature
    #     energy = energy / temperature
    #
    #     # 能量模型的核心：能量越低表示相关性越高
    #     # 通过负能量计算注意力权重（可替换为其他能量函数）
    #     attn_weights = F.softmax(-energy, dim=-1)  # [B, H, L, S]
    #
    #     # 可选：添加掩码（如因果掩码）
    #     if mask is not None:
    #         attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
    #
    #     # 加权求和
    #     output = torch.matmul(attn_weights, v)  # [B, H, L, d]
    #     output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
    #
    #     return output, attn_weights


def safe_normalize(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class Kmeans(nn.Module):
    def __init__(self, num_heads, head_dim, num_clusters, ema_decay=0.999, commitment=1e-4):
        super().__init__()
        self.commitment = commitment
        self.ema_decay = ema_decay
        self.register_buffer('means', torch.randn(num_heads, num_clusters, head_dim))
        self.register_buffer('initted', torch.tensor(False))
        self.num_new_means = 0
        self.new_means = None

    @torch.no_grad()
    def init(self, x):
        if self.initted:
            return
        means = x.mean(dim=2)  # 简化初始化，避免全 0
        self.means.data.copy_(means)
        self.initted.data.fill_(True)

    def forward(self, x, update_means=False):
        self.init(x)
        x = safe_normalize(x)  # 避免除 0 产生 NaN

        with torch.no_grad():
            dists = torch.cdist(x, self.means, p=2)  # 计算距离
            buckets = dists.argmin(dim=-1)  # 选择最近的簇

        routed_means = self.means.gather(1, buckets.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        loss = F.mse_loss(x, routed_means) * self.commitment

        if update_means:
            new_means = x.mean(dim=2, keepdim=True)
            self.means.data = self.ema_decay * self.means + (1 - self.ema_decay) * new_means

        return dists, loss


def default( x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x


import torch
import torch.nn as nn
import torch.nn.functional as F
# helper functions

def exists(val):
    return val is not None

def identity(x, *args, **kwargs):
    return x

def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if exists(cache):
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def compose(*fns):
    def inner(x, *args, **kwargs):
        for fn in reversed(fns):
            x = fn(x, *args, **kwargs)
        return x
    return inner

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

def is_empty(t):
    return t.nelement() == 0

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(2, expand_dim(indices, -1, last_dim))

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def scatter_mean(src, t, index, dim, eps = 1e-5):
    numer = src.scatter_add(dim, index, t)
    denom = src.scatter_add(dim, index, torch.ones_like(t))
    return numer / (denom + eps)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim+1] = split_dims
    return t.reshape(shape)

def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))

# helper classes

def map_first_tuple_or_el(x, fn):
    if isinstance(x, tuple):
        return (fn(x[0]),) + x[1:]
    return fn(x)
TOKEN_SELF_ATTN_VALUE = -1e9 # carefully set for mixed precision training

import torch
import torch.nn as nn


class NeuralClusteringAttention(nn.Module):
    def __init__(self, input_dim, num_clusters, num_heads):
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_proj = nn.Linear(input_dim, num_clusters)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

    def forward(self, X):
        """
        输入 X 的形状为 (B, T, input_dim)，其中 B 为 batch 大小，T 为序列长度。
        1. 首先通过 cluster_proj 得到每个 token 的聚类分数，并计算聚类分配。
        2. 对每个簇：
           - 若该簇无 token，则跳过；
           - 使用 mask 将不属于该簇的 token 置为零；
           - 通过 MultiheadAttention 得到 attention 输出；
           - 最后仅对该簇对应位置进行更新。
        """
        B, T, D = X.shape
        # 计算聚类分数与聚类分配
        cluster_scores = self.cluster_proj(X)  # (B, T, num_clusters)
        cluster_assignments = torch.argmax(cluster_scores, dim=-1)  # (B, T)


        attended_outputs = torch.zeros_like(X)
        # 遍历每个簇，利用同一聚类结果进行注意力计算
        for cluster_idx in range(self.num_clusters):
            # mask: (B, T)，True 表示该 token 属于当前簇
            mask = (cluster_assignments == cluster_idx)
            if mask.sum() == 0:
                continue  # 当前簇没有有效 token，跳过
            # 转换为浮点型，并扩展最后一维： (B, T, 1)
            mask_float = mask.unsqueeze(-1).float()
            # 使用 mask 将不属于当前簇的 token 置为 0
            cluster_X = X * mask_float  # (B, T, D)
            # 注意力计算
            attn_output, _ = self.attention(cluster_X, cluster_X, cluster_X)
            # 仅对 mask 对应的位置进行更新
            attended_outputs += attn_output * mask_float

        return attended_outputs



#  可能导致空值
# class NeuralClusteringAttention(nn.Module):
#     def __init__(self, input_dim, num_clusters, num_heads):
#         super().__init__()
#         self.num_clusters = num_clusters
#         self.cluster_proj = nn.Linear(input_dim, num_clusters)
#         self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
#
#     def cluster_vectors(self, X):
#         """Group vectors into clusters based on similarity."""
#         cluster_scores = self.cluster_proj(X)  # (batch, seq_len, num_clusters)
#         cluster_assignments = torch.argmax(cluster_scores, dim=-1)  # (batch, seq_len)
#         clustered_X = []
#         for cluster_idx in range(self.num_clusters):
#             mask = (cluster_assignments == cluster_idx).unsqueeze(-1)  # (batch, seq_len, 1)
#             clustered_X.append(X * mask)
#         return clustered_X  # List of (batch, seq_len, input_dim)
#
#     def forward(self, X):
#         clustered_X = self.cluster_vectors(X)  # List of (batch, seq_len, input_dim)
#         attended_outputs = torch.zeros_like(X)  # 预分配和输入相同形状的张量
#
#         for cluster_idx, cluster_X in enumerate(clustered_X):
#             if cluster_X.abs().sum() == 0:
#                 # 如果当前簇没有有效向量，跳过或使用默认值
#                 continue
#             attn_output, _ = self.attention(cluster_X, cluster_X, cluster_X)
#             mask = (self.cluster_proj(X).argmax(dim=-1) == cluster_idx).unsqueeze(-1).float()
#             attended_outputs += attn_output * mask
#
#         return attended_outputs  # (batch, seq_len, input_dim)
#
#         # return torch.cat(.attended_outputs, dim=1)  # (batch, seq_len, input_dim)
# #

class KmeansAttention(nn.Module):
    """
    KMeans 注意力机制模块
    """

    def __init__(self, num_clusters, num_heads, head_dim, causal=False, dropout=0.25, ema_decay=0.999,
                 commitment=1e-4, context_window_size=None, receives_context=False,
                 num_mem_kv=0, shared_qk=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.head_dim = head_dim
        self.causal = causal
        self.shared_qk = shared_qk
        self.receives_context = receives_context
        self.kmeans = Kmeans(num_heads, head_dim, num_clusters, ema_decay, commitment)
        self.dropout = nn.Dropout(dropout)
        self.num_mem_kv = max(num_mem_kv, 1 if causal and not shared_qk else 0)
        self.mem_key = nn.Parameter(torch.randn(num_heads, num_clusters, self.num_mem_kv, head_dim))
        self.mem_value = nn.Parameter(torch.randn(num_heads, num_clusters, self.num_mem_kv, head_dim))

    def forward(self, q, k, v, query_mask=None, key_mask=None):
        b, t, d = q.shape
        h = self.num_heads  # 头数
        d = d // h  # 每个 head 的维度
        q = q.view(b, h, t, d)  # 重新调整 q 的形状

        k_b, k_t, k_d = k.shape
        k_h = self.num_heads  # 头数
        k_d = k_d // k_h  # 每个 head 的维度
        k = k.view(k_b, k_h, k_t, k_d)  # 重新调整 k 的形状

        v_b, v_t, v_d = v.shape
        v_h = self.num_heads  # 头数
        v_d = v_d // v_h  # 每个 head 的维度
        v = v.view(v_b, v_h, v_t, v_d)  # 重新调整  v 的形状



        device, dtype = q.device, q.dtype

        # 计算 KMeans 聚类
        dists, aux_loss = self.kmeans(q if self.shared_qk else torch.cat((q, k), dim=2), self.training)
        indices = distribution(dists, t)
        kv_indices = indices if self.shared_qk else distribution(dists, k.shape[2])

        # 选择聚类结果
        q, k, v = map(lambda x: batched_index_select(x, indices if x is q else kv_indices), (q, k, v))
        q, k, v = map(lambda x: x.view(b, h, self.num_clusters, -1, d), (q, k, v))

        # 添加记忆 KV
        k, v = map(lambda x: torch.cat((expand_dim(x, 0, b), x), dim=3), (self.mem_key, self.mem_value))

        # 计算注意力分数
        dots = torch.einsum('bhnid,bhnjd->bhnij', q, k) * (d ** -0.5)
        mask_value = max_neg_value(dots)

        # 应用 mask
        if query_mask is not None or key_mask is not None:
            q_mask = expand_dim(query_mask, 1, h).gather(2, indices)
            kv_mask = expand_dim(key_mask, 1, h).gather(2, kv_indices)
            dots.masked_fill_(~(q_mask.unsqueeze(-1) * kv_mask.unsqueeze(-2)), mask_value)

        if self.causal:
            dots.masked_fill_(indices.unsqueeze(-1) < kv_indices.unsqueeze(-2), mask_value)

        if self.shared_qk:
            dots.masked_fill_(indices.unsqueeze(-1) == kv_indices.unsqueeze(-2), TOKEN_SELF_ATTN_VALUE)

        # 计算注意力权重
        dots = self.dropout(dots.softmax(dim=-1))
        out = torch.einsum('bhnij,bhnjd->bhnid', dots, v).reshape(b, h, -1, d)

        return scatter_mean(torch.zeros_like(q, dtype=dtype), out, indices.unsqueeze(-1).expand_as(out), -2), aux_loss





