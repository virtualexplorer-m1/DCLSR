# here put the import lib
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from models.GRU4Rec import GRU4Rec
from models.SASRec import SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.utils import Multi_CrossAttention, MultiHeadDiffAttention, RecurCluster, ELBO, NeuralClusteringAttention
# from models.utils import Multi_CrossAttention, MultiHeadDiffAttention, ELBO, CoAttention,EBMAttention
import torch.nn.functional as F

class DualLLMGRU4Rec(GRU4Rec):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att

        # load llm embedding as item embedding
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            # def __init__(self, batch_size, n_head,droupoutd, layer_idx=2): args.hidden_size, args.hidden_size
            self.llm2id = MultiHeadDiffAttention(args.hidden_size, args.num_heads,args.dropout_rate , layer_idx=2 )
            self.id2llm = MultiHeadDiffAttention(args.hidden_size, args.num_heads,args.dropout_rate , layer_idx=2 )
            # self.id2llm = RecurCluster(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs):

        id_seqs = self.id_item_emb(log_seqs)
        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(id_seqs )
            cross_llm_seqs = self.id2llm(llm_seqs )
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats
    


class DualLLMSASRec(SASRec_seq):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        # self.dev = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att

        # load llm embedding as item embedding
        # llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset, "pca_itm_emb_np.pkl"), "rb"))
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )
        self.cluster_att_1 = NeuralClusteringAttention(args.hidden_size, args.num_cluster, args.num_heads)
        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        # self.elbo = ELBO(args.cluster_num, args.hidden_size, args.tau_ccl, args.kappa, args.eta, device)
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.rate = nn.Parameter(torch.tensor(args.rate, dtype=torch.float32))
        self.num = args.num
        self.elbo =  ELBO(args.cluster_num,  args.hidden_size,  args.tau_ccl, args.kappa, args.eta ,device )

        if self.use_cross_att:
            # # def __init__(self, batch_size, n_head,droupoutd, layer_idx=2):
            self.llm2id = MultiHeadDiffAttention(args.hidden_size, args.num_heads,args.dropout_rate , layer_idx=2 )
            self.id2llm = MultiHeadDiffAttention(args.hidden_size, args.num_heads,args.dropout_rate , layer_idx=2 )
            # self.llm2id = self.noisef( )
            # self.id2llm = MultiHeadDiffAttention(args.hidden_size, args.num_heads,args.dropout_rate , layer_idx=2 )

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()




    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):
        # 聚类
        #
        # log_seqs: user 交互的项目： 128*200
        id_seqs = self.id_item_emb(log_seqs) # id_seqs 128 * 200 * 64
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.llm_item_emb(log_seqs)

        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)
        # print("llm_seqs.shape")
        # print(llm_seqs.shape)

        diffid_seqs = self.noisef(id_seqs)
        diffllm_seqs = self.noisef(llm_seqs)

        # 聚类 + attention
        llm_features_emb = self.cluster_att_1(llm_seqs)  # (1, N, 64)
        # print(f'llm_features_emb:{llm_features_emb}')
        self.elbo.update_cluster(llm_features_emb)
        loss_ccl = self.elbo(llm_features_emb, llm_seqs)
        print(f'loss_ccl: {loss_ccl}')



        id_log_feats = self.backbone(diffid_seqs, log_seqs)
        llm_log_feats = self.backbone(diffllm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)
        return log_feats , loss_ccl




    def noisef(self, x ):
        x_f = torch.fft.rfft(x, dim=1)  #
        m = torch.rand(x_f.shape, device=x_f.device) < self.rate
        amp = abs(x_f)  #
        _, index = amp.sort(dim=1, descending=True)  #
        dominant_mask = index >  self.num # dominant_mask = index > 5 # torch.bitwise_or
        m = torch.bitwise_and(m, dominant_mask)  # m = torch.bitwise_and(m,dominant_mask)
        freal = x_f.real.masked_fill(m, 0)
        fimag = x_f.imag.masked_fill(m, 0)
        x_f = torch.complex(freal, fimag)  #
        x = torch.fft.irfft(x_f, dim=1)  #
        return x

class DualLLMBert4Rec(Bert4Rec):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        # self.dev = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att

        # load llm embedding as item embedding
        # llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset, "pca_itm_emb_np.pkl"), "rb"))
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        self.cluster_att_1 = NeuralClusteringAttention(args.hidden_size, args.num_cluster, args.num_heads)
        self.rate = nn.Parameter(torch.tensor(args.rate, dtype=torch.float32))
        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        self.elbo =  ELBO(args.cluster_num,  args.hidden_size,  args.tau_ccl, args.kappa, args.eta ,device )

        self.num = args.num
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = MultiHeadDiffAttention(args.hidden_size, args.num_heads,args.dropout_rate , layer_idx=2 )
            self.id2llm = MultiHeadDiffAttention(args.hidden_size, args.num_heads,args.dropout_rate , layer_idx=2 )

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):

        id_seqs = self.id_item_emb(log_seqs)
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)


        diffid_seqs = self.noisef(id_seqs)


        diffllm_seqs = self.noisef(llm_seqs)

        llm_features_emb = self.cluster_att_1(llm_seqs)  # (1, N, 64)
        # print(f'llm_features_emb:{llm_features_emb}')
        self.elbo.update_cluster(llm_features_emb)
        loss_ccl = self.elbo(llm_features_emb, llm_seqs)
        print(f'loss_ccl: {loss_ccl}')

        id_log_feats = self.backbone(diffid_seqs, log_seqs)
        llm_log_feats = self.backbone(diffllm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats, loss_ccl

    def noisef(self, x ):
        x_f = torch.fft.rfft(x, dim=1)  #
        m = torch.rand(x_f.shape, device=x_f.device) < self.rate
        amp = abs(x_f)  #
        _, index = amp.sort(dim=1, descending=True)  #
        dominant_mask = index >  self.num # dominant_mask = index > 5 # torch.bitwise_or
        m = torch.bitwise_and(m, dominant_mask)  # m = torch.bitwise_and(m,dominant_mask)
        freal = x_f.real.masked_fill(m, 0)
        fimag = x_f.imag.masked_fill(m, 0)
        x_f = torch.complex(freal, fimag)  #
        x = torch.fft.irfft(x_f, dim=1)  #
        return x


