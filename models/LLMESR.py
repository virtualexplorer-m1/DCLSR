# here put the import lib
from platform import node

import numpy as np
import torch
import torch.nn as nn
from math import ceil
from networkx import clustering
from sklearn.decomposition import PCA

from models.DualLLMSRS import DualLLMSASRec, DualLLMGRU4Rec, DualLLMBert4Rec
from models.utils import Contrastive_Loss2
from models.utils import Multi_CrossAttention, MultiHeadDiffAttention, ELBO, CoAttention,EBMAttention,KmeansAttention, NeuralClusteringAttention
import torch.nn.functional as F

class LLMESR_SASRec(DualLLMSASRec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.alpha_item = args.alpha_item
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        # num_class, num_cluster, feat_dim, tau, kappa, eta, device):
        self.tau = args.tau_ccl
        self.kappa = args.kappa
        self.eta = args.eta
        self.device = device
        self.n_components= args.n_components
        self.logSoftmax = torch.nn.LogSoftmax(dim=1)

        self.diff = MultiHeadDiffAttention(args.hidden_size, args.num_heads, args.dropout_rate, layer_idx=2)
        # num_clusters, num_heads, head_dim,
        self.cluster_att_1 = NeuralClusteringAttention( args.hidden_size, args.cluster_num, args.num_heads  )
        self.cluster_att = KmeansAttention(args.cluster_num, args.num_heads, args.hidden_size )
        self.attention_1 = EBMAttention(args.hidden_size, args.num_heads, temperature=0.1)
        self.elbo = ELBO(args.cluster_num,  args.hidden_size,  args.tau_ccl, args.kappa, args.eta ,device )
        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.attention = nn.Sequential(
            nn.Linear(args.hidden_size ,args.hidden_size ),  # 计算每个类的权重
            nn.Softmax(dim=1)  # 归一化为概率
        )

        self.num_cluster = args.num_cluster
        self.sigma = args.sigma
        self.ccal = nn.Parameter(torch.tensor(args.ccal, dtype=torch.float32))
        self.elboccl = nn.Parameter(torch.tensor(args.elboccl, dtype=torch.float32))
        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()

    def forward(self,
                seq,
                pos,
                neg,
                positions,
                **kwargs):

        # 先调用父类获得原始损失
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # 原始 loss

        # ------------------ 用户部分 ------------------
        # 从 log2feats 获得用户特征（取最后一个时间步），假设拼接了 id 和 llm 两部分
        user_log, loss_llm  = self.log2feats(seq, positions)
        print(f'loss_llm:{loss_llm}')
        user_log_feats_diff = user_log[:, -1, :]
        split_dim = user_log_feats_diff.shape[-1] // 2  # 假设两部分等长
        user_id_log_feats_diff, user_llm_log_feats_diff = user_log_feats_diff.split(
            [split_dim, user_log_feats_diff.shape[-1] - split_dim],
            dim=-1
        )

        # 根据 user_sim_func 选择计算方式（这里 cl 和 kd 均调用 align）
        if self.user_sim_func in ["cl", "kd"]:
            align_loss = self.align(user_id_log_feats_diff, user_llm_log_feats_diff)
        else:
            align_loss = 0.0

        if self.item_reg:
        # 项目
            # 没有去噪 + backbone   最初始的项目在LLM后的特征表示！！！！！！！
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))  # 1244 * 64
            #
            # item_llm_data = self.pca_perturb(llm_item_emb,  self.sigma,  self.n_components)

            id_item_emb = self.id_item_emb(unfold_item_id)

            # loss_cl = self.align(id_item_emb, llm_item_emb).mean()

            reg_loss = self.reg(llm_item_emb, id_item_emb)
            reg_loss = reg_loss.mean()
            # loss_term =  self.beta * reg_loss + loss_cl * self.ccal
            loss_term =  self.beta * reg_loss
            loss += loss_term

        loss += self.alpha * align_loss + loss_llm * self.ccal
        return loss

    #
    # def forward(self,
    #             seq,
    #             pos,
    #             neg,
    #             positions,
    #             **kwargs):
    #
    #     loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
    #
    #     # 用户
    #     user_log_feats_diff  = self.log2feats(seq, positions)[:, -1, :]
    #
    #     # 计算 id_log_feats_diff 和 llm_log_feats_diff 的正确拆分位置
    #     split_dim = user_log_feats_diff.shape[-1] // 2  # 假设是等长拼接
    #
    #     user_id_log_feats_diff, user_llm_log_feats_diff = user_log_feats_diff.split([split_dim, user_log_feats_diff.shape[-1] - split_dim],
    #                                                                  dim=-1)
    #
    #     # # item
    #     # item_feats_diff = self.log2feats(seq, positions)
    #     # # 计算 id_log_feats_diff 和 llm_log_feats_diff 的正确拆分位置
    #     # split_dim_item = item_feats_diff.shape[-1] // 2  # 假设是等长拼接
    #     #
    #     # item_id_log_feats_diff, item_llm_log_feats_diff = item_feats_diff.split(
    #     #     [split_dim, item_feats_diff.shape[-1] - split_dim_item],
    #     #     dim=-1)
    #
    #     #
    #     # print("item_id_log_feats_diff.shape")
    #     # print(item_id_log_feats_diff.shape)
    #     # align_loss_item = self.item_cl(item_id_log_feats_diff, item_llm_log_feats_diff)
    #
    #
    #     if self.user_sim_func == "cl":
    #         align_loss = self.align( user_id_log_feats_diff, user_llm_log_feats_diff)
    #     elif self.user_sim_func == "kd":
    #         align_loss = self.align( user_id_log_feats_diff, user_llm_log_feats_diff)
    #
    #
    #     if self.item_reg:
    #     # 项目
    #     # 没有去噪 + backbone   最初始的项目在LLM后的特征表示！！！！！！！
    #         unfold_item_id = torch.masked_select(seq, seq>0)
    #         llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))  # 1244 * 64
    #         llm_features = llm_item_emb.unsqueeze(0)  # 在维度 1 添加一个新维度  1 * 1244 * 64
    #
    #         # llm_features_emb,_ = self.attention_1(llm_features, llm_features, llm_features)  # 1 * 1244 * 64
    #         # llm_features_emb = self.diff(llm_features )  # 1 * 1244 * 64
    #         llm_features_emb = self.cluster_att_1(llm_features  )  # 1 * 1244 * 64
    #         # print("llm_features_emb.shape")
    #         # print(llm_features_emb.shape)
    #         # diff_llm_feature
    #         # print(type(llm_features_emb))  # 确认是否为 tuple
    #         # print(llm_features_emb)  # 查看元组内容
    #         llm_features_att = llm_features_emb.squeeze(0)
    #
    #
    #         item_llm_data = self.pca_perturb(llm_features_att,  self.sigma,  self.n_components)
    #         # item_llm_data = self.mixup(llm_item_emb, alpha=0.55)
    #         # item_llm_data =torch.tensor(item_llm_data).to(device)
    #         # attention 1 和 attention 2
    #
    #         # similarity = F.cosine_similarity(item_llm_data, llm_features_att, dim=-1)  # (1244,)
    #         # print("similarity")
    #         # print(similarity)
    #         id_item_emb = self.id_item_emb(unfold_item_id)
    #
    #
    #         self.elbo.update_cluster(llm_features_att )
    #         # 协方差
    #
    #         # mean_llm, cov_llm= self.compute_mean_cov(llm_item_emb)
    #         # mean_id, cov_id = self.compute_mean_cov(id_item_emb)
    #         loss_cl = self.cosine_similarity_loss(id_item_emb, item_llm_data).mean()
    #         # loss_cl = self.alignment(llm_features_att, item_llm_data).mean()
    #         print(f'loss_cl:{loss_cl}')
    #         loss_ccl = self.elbo(llm_features_att, item_llm_data)
    #         # print(f'loss_ccl:{loss_ccl}')
    #
    #         reg_loss = self.reg(llm_item_emb, id_item_emb)
    #         reg_loss = reg_loss.mean()
    #         loss_term = loss_ccl * self.elboccl + self.beta * reg_loss + loss_cl * self.ccal
    #         # loss_term = torch.nan_to_num(loss_term, nan=0.0, posinf=1e6, neginf=-1e6)
    #         loss = loss_term + loss
    #     # loss += loss_ccl * self.elboccl + self.beta * reg_loss
    #
    #     loss += self.alpha * align_loss
    #
    #     return loss


    def alignment(self, x, y, alpha=2):

        return (x - y).norm(p=2, dim=1).pow(alpha).mean()


    from sklearn.decomposition import PCA
    def mixup(self, features, alpha=0.2):
        device = features.device  # 获取输入特征所在的设备
        n = features.shape[0]
        # 将 features 从 GPU 转到 CPU 进行 PCA 操作
        features_np = features.detach().cpu().numpy()  # 将 features 转为 numpy 数组


        indices = np.random.permutation(n)
        lam = np.random.beta(alpha, alpha)
        mixed = lam * features_np + (1 - lam) * features_np[indices]
        return mixed

    # 使用示例
    # mixed_features = mixup(original_features, alpha=0.5)
    # def pca_perturb(self,features, sigma=0.05, n_components=10):
    #     device = features.device  # 获取输入特征所在的设备
    #
    #     # 将 features 从 GPU 转到 CPU 进行 PCA 操作
    #     features_np = features.detach().cpu().numpy()  # 将 features 转为 numpy 数组
    #
    #     n_samples, n_features = features_np.shape
    #     n_min = min(n_samples, n_features)
    #
    #     # 检查 n_components 是否超出范围，如果超出，则自动调整
    #     if n_components > n_min:
    #         print(f"Warning: n_components={n_components} 超过了允许范围，自动设置为 {n_min}")
    #         n_components = n_min
    #
    #     # 初始化 PCA，并进行拟合
    #     pca = PCA(n_components=n_components, svd_solver='full')
    #     # pca = PCA(n_components=n_components)
    #     pca.fit(features_np)
    #
    #     components = pca.components_  # [n_components, 64]
    #     coeff = pca.transform(features_np)  # [128, n_components]
    #
    #     # 在主成分系数上添加噪声
    #     noise = np.random.normal(0, sigma, coeff.shape)
    #     perturbed_coeff = coeff + noise
    #
    #     # 通过逆变换还原数据
    #     a = pca.inverse_transform(perturbed_coeff)  # [128, 64]，恢复到特征空间
    #
    #     # 将还原后的数据转为 PyTorch 张量，并恢复到原始设备和数据类型
    #     perturbed_features = torch.tensor(a, dtype=torch.float32, device=device)  # 使用 torch.float32
    #
    #     # 重构特征
    #     return perturbed_features

    def pca_perturb(self, features, sigma=0.05, n_components=10):
        device = features.device  # 获取输入特征所在的设备

        # 将 features 从 GPU 转到 CPU 进行 PCA 操作，并转换为 numpy 数组
        features_np = features.detach().cpu().numpy()

        # 如果输入为空，则返回一个全0的张量，其形状设为 (1, n_features)
        if features_np.shape[0] == 0:
            print("Warning: 输入的 features 样本数为 0，返回一个全0张量，其形状为 (1, n_features)")
            # 注意：这里返回的张量样本数为1，可能与下游期望不符，
            # 请根据实际需求确认是否需要调整该策略。
            return torch.zeros(1, features_np.shape[1], device=device, dtype=torch.float32)

        n_samples, n_features = features_np.shape
        n_min = min(n_samples, n_features)

        # 检查 n_components 是否超出范围，如果超出，则自动调整
        if n_components > n_min:
            print(f"Warning: n_components={n_components} 超过允许范围，自动设置为 {n_min}")
            n_components = n_min

        # 初始化 PCA，并进行拟合
        pca = PCA(n_components=n_components, svd_solver='full')
        try:
            pca.fit(features_np)
        except Exception as e:
            print(f"PCA fit 失败: {e}")
            return features

        # 获取 PCA 的分量和转换后的系数
        components = pca.components_  # shape: [n_components, n_features]
        coeff = pca.transform(features_np)  # shape: [n_samples, n_components]

        # 在主成分系数上添加噪声
        noise = np.random.normal(0, sigma, coeff.shape)
        perturbed_coeff = coeff + noise

        # 通过逆变换还原数据（重构到原始特征空间）
        a = pca.inverse_transform(perturbed_coeff)  # shape: [n_samples, n_features]

        # 将还原后的数据转为 PyTorch 张量，并恢复到原始设备和数据类型
        perturbed_features = torch.from_numpy(a).to(device).float()

        return perturbed_features

    # def pca_perturb(self, features, sigma=0.05, n_components=10):
    #     device = features.device  # 获取输入特征所在的设备
    #
    #     # 将 features 从 GPU 转到 CPU 进行 PCA 操作，并转换为 numpy 数组
    #     features_np = features.detach().cpu().numpy()
    #
    #     # 检查输入是否为空
    #     if features_np.shape[0] == 0:
    #         print("Warning: 输入的 features 样本数为 0，直接返回原始 features")
    #         return features
    #
    #     n_samples, n_features = features_np.shape
    #     n_min = min(n_samples, n_features)
    #
    #     # 检查 n_components 是否超出范围，如果超出，则自动调整
    #     if n_components > n_min:
    #         print(f"Warning: n_components={n_components} 超过允许范围，自动设置为 {n_min}")
    #         n_components = n_min
    #
    #     # 初始化 PCA，并进行拟合
    #     pca = PCA(n_components=n_components, svd_solver='full')
    #     try:
    #         pca.fit(features_np)
    #     except Exception as e:
    #         print(f"PCA fit 失败: {e}")
    #         return features
    #
    #     # 获取 PCA 的分量和转换后的系数
    #     components = pca.components_  # shape: [n_components, n_features]
    #     coeff = pca.transform(features_np)  # shape: [n_samples, n_components]
    #
    #     # 在主成分系数上添加噪声
    #     noise = np.random.normal(0, sigma, coeff.shape)
    #     perturbed_coeff = coeff + noise
    #
    #     # 通过逆变换还原数据（重构到原始特征空间）
    #     a = pca.inverse_transform(perturbed_coeff)  # shape: [n_samples, n_features]
    #
    #     # 将还原后的数据转为 PyTorch 张量，并恢复到原始设备和数据类型
    #     perturbed_features = torch.from_numpy(a).to(device).float()
    #
    #     return perturbed_features
    def cosine_similarity_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 1 - torch.sum(x * y, dim=1).mean()

    def compute_mean_cov(self, emb):
        """
        计算给定特征向量的均值和协方差矩阵
        :param emb: 形状 (batch_size, feature_dim) 的张量
        :return: (均值, 协方差矩阵)
        """
        mean = torch.mean(emb, dim=0)  # 按行计算均值，形状: (feature_dim,)

        # 计算协方差矩阵 (feature_dim, feature_dim)
        cov = torch.cov(emb.T)  # 需要转置，使得形状符合 torch.cov 的要求

        return mean, cov



    def wasserstein_distance(self, mean1, cov1, mean2, cov2):
        # 均值项：欧几里得距离平方
        mean_diff = torch.sum((mean1 - mean2) ** 2)

        # 协方差项：使用特征值分解进行计算
        eigvals1, _ = torch.linalg.eigh(cov1)
        eigvals2, _ = torch.linalg.eigh(cov2)

        # 计算 Wasserstein 距离的协方差项
        cov_term = torch.sum((torch.sqrt(torch.clamp(eigvals1, min=1e-24)) -
                              torch.sqrt(torch.clamp(eigvals2, min=1e-24))) ** 2)

        return mean_diff + cov_term

    def item_cl(self,item1, item2):
        loss = nn.MSELoss()(item1, item2)

        return loss

    def cluster_scl(self, emb, emb2, y, cen):

        features = torch.cat((emb, emb2), dim=0)
        batchSize = features.shape[0]
        y = y.contiguous().view(-1, 1)
        mask = torch.eq(y, y.T).float().to(self.device)
        mask = mask.repeat(2, 2)

        anchor_dot_cluster = torch.matmul(features, cen.T)  # BS x M
        anchor_dot_contrast = torch.matmul(features, features.T)  # BS x BS

        # clusterscl loss
        pi_logit = torch.div(anchor_dot_cluster, self.kappa)
        log_pi = self.logSoftmax(pi_logit + 1e-18)  # BS x M
        pi = torch.exp(log_pi)  # cluster distribution p(c | v), BS x M

        loss_0 = torch.mean(torch.sum(pi * log_pi, dim=1))

        # compute the alignment with the augmented positives and negatives
        align_cluster = anchor_dot_cluster.T.view(self.num_cluster, batchSize, 1).repeat(1, 1, batchSize)
        align_contrast = anchor_dot_contrast.repeat(self.num_cluster, 1).view(self.num_cluster, batchSize,
                                                                              batchSize)
        weight1 = torch.div(torch.exp(align_cluster), (torch.exp(align_cluster) + torch.exp(align_contrast)))
        weight2 = torch.div(torch.exp(align_contrast), (torch.exp(align_cluster) + torch.exp(align_contrast)))

        anchor_dot_augmentation = ( weight1 * align_cluster + weight2 * align_contrast) / self.tau + 1e-18  # M x BS x BS

        logits_max, _ = torch.max(anchor_dot_augmentation, dim=2, keepdim=True)
        logits = anchor_dot_augmentation - logits_max.detach()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batchSize).view(-1, 1).to(self.device),
            0
        )
        # set the diagonal elements to be 0
        mask = mask * logits_mask
        # logits_mask = logits_mask.view(logits.size())
        # logits_mask = logits_mask.view(*logits.size())
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_logits = logits - torch.log(exp_logits.sum(2, keepdim=True))
        normalized_logits = torch.exp(log_logits)  # p(s | v, c)
        # remain the prob of each anchor's positives
        log_logits_pos = torch.mul(log_logits, mask)
        normalized_logits_pos = torch.mul(normalized_logits, mask)
        del log_logits
        del normalized_logits

        pi_normalized_logits_pos = pi.T.view(self.num_cluster, batchSize, 1) * normalized_logits_pos
        posterior = torch.div(pi_normalized_logits_pos, torch.add(torch.sum(pi_normalized_logits_pos, 0), 1 - mask))
        posterior = torch.mul(posterior, mask)  # q(c | v, s)

        loss = -torch.mean(torch.div(torch.sum(torch.sum(
            posterior * (log_pi.T.view(self.num_cluster, batchSize, 1) + log_logits_pos - torch.log(
                posterior + 1e-18)),
            0), 1), torch.sum(mask, 1)))

        loss_final = loss + self.eta * loss_0

        return loss_final

    def llm_augument(self, x, y):
        """
        使用 K-means 算法对 1244 个 64 维向量进行聚类，分成 10 组。
        输入：
            x: Tensor, shape=(N, 64)
            y: Tensor, shape=(N, 64)
        返回：
            loss1: 每个样本的损失，shape=(N,)
        """
        all_embs1 = x
        all_embs2 = y

        # 计算向量范数，保持维度用于广播
        all_embs1_abs = all_embs1.norm(dim=1, keepdim=True)  # shape: [N, 1]
        all_embs2_abs = all_embs2.norm(dim=1, keepdim=True)  # shape: [N, 1]

        # 防止范数为0
        all_embs1_abs = torch.clamp(all_embs1_abs, min=1e-8)
        all_embs2_abs = torch.clamp(all_embs2_abs, min=1e-8)

        # 计算余弦相似度
        cosine_sim = torch.matmul(all_embs1, all_embs2.T) / (all_embs1_abs * all_embs2_abs.T)
        cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)

        # 缩放相似度，并限制范围防止指数运算溢出
        scaled_sim = cosine_sim / 0.1
        scaled_sim = torch.clamp(scaled_sim, min=-50, max=50)
        sim_matrix = torch.exp(scaled_sim)

        # 正样本相似度（取对角线）
        pos_sim = torch.diag(sim_matrix)

        # 计算分母并 clamp 防止除零
        denom = sim_matrix.sum(dim=1) - pos_sim
        denom = torch.clamp(denom, min=1e-6)

        # 计算正样本比率
        ratio = pos_sim / denom
        ratio = torch.clamp(ratio, min=1e-8, max=1e8)

        # 计算损失，并使用 nan_to_num 对结果进行保护
        loss1 = -torch.log(ratio + 1e-8)
        loss1 = torch.nan_to_num(loss1, nan=0.0, posinf=1e6, neginf=-1e6)

        return loss1

    def kms(self, embeds):
        """
        使用 K-means 算法对样本进行聚类。
        embeds: (num_samples, embedding_size) 张量，每一行是一个样本的嵌入表示。

        返回:
            centroids: (num_clusters, embedding_size) 张量，代表聚类中心
            labels: (num_samples,) 张量，代表每个样本的类别索引
        """
        num_clusters = 10  # 设置聚类中心数
        num_samples, embedding_size = embeds.shape
        device = embeds.device

        # 如果样本数量小于聚类中心数，则直接使用所有样本作为聚类中心
        if num_samples < num_clusters:
            new_centroids = embeds.clone().to(device)
        else:
            # 随机初始化聚类中心
            centroids = embeds[torch.randperm(num_samples)[:num_clusters]].clone().to(device)
            # K-means 迭代
            for _ in range(30):  # 进行 10 轮迭代
                # 计算每个样本到聚类中心的欧几里得距离
                dists = torch.cdist(embeds, centroids)  # (num_samples, num_clusters)

                # 找到每个样本距离最近的聚类中心
                labels = dists.argmin(dim=1)  # (num_samples,)

                # 更新聚类中心
                new_centroids = torch.zeros_like(centroids)  # (num_clusters, embedding_size)

                for i in range(num_clusters):
                    mask = labels == i  # 选出属于第 i 类的点
                    if mask.sum() > 0:  # 防止某个类别为空
                        new_centroids[i] = embeds[mask].mean(dim=0)
                    else:
                        new_centroids[i] = embeds[torch.randint(0, num_samples, (1,)).item()]  # 重新随机选点

        centroids = new_centroids  # 更新聚类中心

        return centroids

    def kmeans_clustering(self, embeds, num_clusters=10, max_iter=30):
        """
        使用 K-means 算法对样本进行聚类，分成 num_clusters 组。
        如果样本数量小于聚类中心数，则直接使用所有样本作为聚类中心。

        embeds: (num_samples, embedding_size) 的张量，每一行是一个样本。
        num_clusters: 聚类中心的数量。
        max_iter: 最大迭代次数。

        返回：
            cluster_centers: 聚类中心（num_samples, embedding_size）
            labels: 每个样本的类别索引 (num_samples,)
        """
        num_samples, embedding_size = embeds.shape
        device = embeds.device

        # 如果样本数量小于聚类中心数，则直接使用所有样本作为聚类中心
        if num_samples < num_clusters:
            # 将每个样本作为一个聚类中心，标签就是其索引
            cluster_centers = embeds.clone().to(device)  # (num_samples, embedding_size)
            labels = torch.arange(num_samples).to(device)  # (num_samples,) 每个样本属于自己为一个类别
        else:
            # 随机初始化聚类中心
            centroids = embeds[torch.randperm(num_samples)[:num_clusters]].clone().to(device)

            # K-means 迭代
            for _ in range(max_iter):  # 进行 max_iter 轮迭代
                # 计算每个样本到聚类中心的欧几里得距离
                dists = torch.cdist(embeds, centroids)  # (num_samples, num_clusters)

                # 找到每个样本距离最近的聚类中心
                labels = dists.argmin(dim=1)  # (num_samples,)

                # 更新聚类中心
                new_centroids = torch.zeros_like(centroids)  # (num_clusters, embedding_size)

                for i in range(num_clusters):
                    mask = labels == i  # 选出属于第 i 类的点
                    if mask.sum() > 0:  # 防止某个类别为空
                        new_centroids[i] = embeds[mask].mean(dim=0)
                    else:
                        # 重新随机选点，避免空类别
                        new_centroids[i] = embeds[torch.randint(0, num_samples, (1,)).item()]

                # 更新聚类中心
                centroids = new_centroids

            # 计算每个样本的特征向量是所属类别的聚类中心
            cluster_centers = centroids[labels]  # (num_samples, embedding_size)

        return cluster_centers, labels

    def llm_kms(self, embeds, llm_centroids ):
        """
        基于已有的聚类中心进行 K-means 迭代并增强嵌入。
        embeds: 当前的样本嵌入 (num_samples, embedding_size)
        llm_centroids: 预训练的聚类中心 (num_clusters, embedding_size)
        llm_labels: 预训练的标签 (num_samples,)

        返回:
            enhanced_embeds: 增强后的嵌入
        """
        num_clusters = 10  # 聚类中心数
        num_samples, embedding_size = embeds.shape
        device = embeds.device
        # centroids = llm_centroids

        # 如果样本数量小于聚类中心数，则直接使用所有样本作为聚类中心
        if num_samples < num_clusters:
            cluster_centers = embeds.clone().to(device)
        else:
            # 随机初始化聚类中心
            centroids = embeds[torch.randperm(num_samples)[:num_clusters]].clone().to(device)
            # K-means 迭代
            for _ in range(30):  # 进行 10 轮迭代
                # 计算每个样本到聚类中心的欧几里得距离
                dists = torch.cdist(embeds, centroids)  # (num_samples, num_clusters)

                # 找到每个样本距离最近的聚类中心
                labels = dists.argmin(dim=1)  # (num_samples,)

                # 更新聚类中心
                new_centroids = torch.zeros_like(centroids)  # (num_clusters, embedding_size)

                for i in range(num_clusters):
                    mask = labels == i  # 选出属于第 i 类的点
                    if mask.sum() > 0:  # 防止某个类别为空
                        new_centroids[i] = embeds[mask].mean(dim=0)
                    else:
                        new_centroids[i] = embeds[torch.randint(0, num_samples, (1,)).item()]  # 重新随机选点

                centroids = new_centroids  # 更新聚类中心
            # 计算每个样本的特征向量是所属类别的聚类中心
            cluster_centers = centroids[labels]  # (num_samples, embedding_size)

        # 结合原始特征和聚类特征进行增强  可能不合适。
        enhanced_embeds = embeds + cluster_centers * self.clrate  # (batch_size, embedding_size)
        #  将每个样本与聚类中心进行对比。
        return enhanced_embeds


    # def kms(self, embeds):
    #     """
    #     使用 K-means 算法对 1244 个 64 维向量进行聚类，分成 10 组。
    #     embeds: (1244, 64) 的张量，每一行是一个样本。
    #     返回：
    #         centroids: (10, 64) 的张量，代表 10 个聚类中心
    #         labels: (1244,) 的张量，代表每个样本的类别索引
    #     """
    #
    #     num_clusters = 10  # 设定聚类中心数
    #     num_samples, embedding_size = embeds.shape  # 1244 个样本，每个是 64 维
    #     device = embeds.device
    #
    #     # 1. 随机初始化聚类中心 (10, 64)
    #     centroids = embeds[torch.randperm(num_samples)[:num_clusters]].clone().to(device)
    #
    #     # 2. K-means 迭代
    #     for _ in range(20):  # 进行 10 轮迭代
    #         # 计算每个样本到 10 个聚类中心的欧几里得距离
    #         dists = torch.cdist(embeds, centroids)  # (1244, 10)
    #
    #         # 找到每个样本距离最近的聚类中心
    #         labels = dists.argmin(dim=1)  # (1244,)
    #
    #         # 计算新的聚类中心
    #         new_centroids = torch.zeros_like(centroids)  # (10, 64)
    #         counts = torch.zeros(num_clusters, 1, device=device)  # (10, 1)
    #
    #         for i in range(num_clusters):
    #             mask = labels == i  # 选出属于第 i 类的点
    #             if mask.sum() > 0:  # 防止某个类别为空
    #                 new_centroids[i] = embeds[mask].mean(dim=0)
    #             else:
    #                 new_centroids[i] = embeds[torch.randint(0, num_samples, (1,))]  # 重新随机选点
    #
    #         centroids = new_centroids  # 更新聚类中心
    #
    #     return centroids, labels
    #
    # def llm_kms(self, embeds, llm_centroids, llm_labels):
    #     num_clusters = 10  # 设定聚类中心数
    #     num_samples, embedding_size = embeds.shape  # 1244 个样本，每个是 64 维
    #     device = embeds.device
    #     centroids = llm_centroids
    #
    #     for _ in range(20):  # 进行 10 轮迭代
    #         # 计算每个样本到 10 个聚类中心的欧几里得距离
    #         dists = torch.cdist(embeds, centroids)  # (1244, 10)
    #
    #         # 找到每个样本距离最近的聚类中心
    #         labels = dists.argmin(dim=1)  # (1244,)
    #
    #         # 计算新的聚类中心
    #         new_centroids = torch.zeros_like(centroids)  # (10, 64)
    #         counts = torch.zeros(num_clusters, 1, device=device)  # (10, 1)
    #
    #         for i in range(num_clusters):
    #             mask = labels == i  # 选出属于第 i 类的点
    #             if mask.sum() > 0:  # 防止某个类别为空
    #                 new_centroids[i] = embeds[mask].mean(dim=0)
    #             else:
    #                 new_centroids[i] = embeds[torch.randint(0, num_samples, (1,))]  # 重新随机选点
    #
    #         centroids = new_centroids  # 更新聚类中心
    #         cluster_centers = centroids[labels]  # (1244, 64)，每个样本的特征向量是所属类别的聚类中心
    #
    #         # 结合原始特征和聚类特征
    #         enhanced_embeds = embeds + cluster_centers * self.clrate # (batch_size, embedding_size)
    #
    #     return enhanced_embeds

class LLMESR_GRU4Rec(DualLLMGRU4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq)[:, -1, :]    # (bs*sim_num, hidden_size)
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        loss += self.alpha * align_loss

        return loss



class LLMESR_Bert4Rec(DualLLMBert4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.ccal = nn.Parameter(torch.tensor(args.ccal, dtype=torch.float32))
        if self.item_reg:
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        user_log, loss_llm = self.log2feats(seq, positions)
        print(f'loss_llm: {loss_llm}')
        user_log_feats_diff = user_log[:, -1, :]
        split_dim = user_log_feats_diff.shape[-1] // 2  # 假设两部分等长
        user_id_log_feats_diff, user_llm_log_feats_diff = user_log_feats_diff.split(
            [split_dim, user_log_feats_diff.shape[-1] - split_dim],
            dim=-1
        )

        if self.user_sim_func in ["cl", "kd"]:
            align_loss = self.align(user_id_log_feats_diff, user_llm_log_feats_diff)
        else:
            align_loss = 0.0


        loss += self.alpha * align_loss + loss_llm * self.ccal

        return loss



