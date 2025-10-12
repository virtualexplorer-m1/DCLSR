# here put the import lib
import os
import copy
import random
import numpy as np
from torch.utils.data import Dataset
from utils.utils import random_neq
import pickle


class SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation'''

    def __init__(self, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.var_name = ["seq", "pos", "neg", "positions"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions
    


class SeqDatasetAllUser(SeqDataset):
    '''The train dataset for Sequential recommendation'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__(data, item_num, max_len, neg_num)
        self.sim_user_num = args.sim_user_num
        self.sim_users = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", "sim_user_100.pkl"), "rb"))
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        ### get the sequence of similar user
        sim_users = self.sim_users[index][:self.sim_user_num]
        sim_seq, sim_positions = [], []
        for sim_user in sim_users:
            meta_seq, meta_positions = self._get_user_seq(sim_user)
            sim_seq.append(meta_seq)
            sim_positions.append(meta_positions)
        
        sim_seq = np.array(sim_seq)
        sim_positions = np.array(sim_positions)

        return seq, pos, neg, positions, index, sim_seq, sim_positions
    

    def _get_user_seq(self, user):

        ### get the sequence of required user
        inter = self.data[user]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions



class Seq2SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation with seq-to-seq loss'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.aug_seq = args.aug_seq
        self.aug_seq_len = args.aug_seq_len
        self.var_name = ["seq", "pos", "neg", "positions"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num+1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        if self.aug_seq:
            seq_len = len(inter)
            pos[:- (seq_len - self.aug_seq_len) + 1] = 0
            neg[:- (seq_len - self.aug_seq_len) + 1] = 0
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions



class Seq2SeqDatasetAllUser(Seq2SeqDataset):

    def __init__(self, args, data, item_num, max_len, neg_num=1):

        super().__init__(args, data, item_num, max_len, neg_num)
        self.sim_user_num = args.sim_user_num
        self.sim_users = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", "sim_user_100.pkl"), "rb"))
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions"]


    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num+1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        if self.aug_seq:
            seq_len = len(inter)
            pos[:- (seq_len - self.aug_seq_len) + 1] = 0
            neg[:- (seq_len - self.aug_seq_len) + 1] = 0
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        ### get the sequence of similar user
        sim_users = self.sim_users[index][:self.sim_user_num]
        sim_seq, sim_positions = [], []
        for sim_user in sim_users:
            meta_seq, meta_positions = self._get_user_seq(sim_user)
            sim_seq.append(meta_seq)
            sim_positions.append(meta_positions)
        
        sim_seq = np.array(sim_seq)
        sim_positions = np.array(sim_positions)

        return seq, pos, neg, positions, index, sim_seq, sim_positions
    

    def _get_user_seq(self, user):

        ### get the sequence of required user
        inter = self.data[user]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions
    


class BertRecTrainDatasetAllUser(Dataset):
    '''The train dataset for Bert4Rec'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.mask_prob = args.mask_prob
        self.sim_user_num = args.sim_user_num
        self.mask_token = item_num + 1
        self.sim_users = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", "sim_user_100.pkl"), "rb"))
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions"]


    def __len__(self):

        return 2 * len(self.data)

    def __getitem__(self, index):

        tokens = []
        labels, neg_labels = [], []

        if index >= len(self.data):
            seq = self.data[index - len(self.data)]
            for s in seq:
                tokens.append(s)
                labels.append(0)
                neg_labels.append(0)
            labels[-1] = tokens[-1]
            neg_labels[-1] = random_neq(1, self.item_num+1, seq)
            tokens[-1] = self.mask_token

        else:
            seq = self.data[index]
   
            for s in seq:
                prob = random.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(random.randint(1, self.item_num))
                    else:
                        tokens.append(s)

                    labels.append(s)
                    neg = random_neq(1, self.item_num+1, seq)
                    neg_labels.append(neg)

                else:
                    tokens.append(s)
                    labels.append(0)
                    neg_labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        neg_labels = neg_labels[-self.max_len:]
        pos = list(range(1, len(tokens)+1))
        pos= pos[-self.max_len:]

        mask_len = self.max_len - len(tokens)
        
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        neg_labels = [0] * mask_len + neg_labels
        pos = [0] * mask_len + pos

        if index >= len(self.data):
            user_id = index - len(self.data)
        else:
            user_id = index

        ### get the sequence of similar user
        sim_users = self.sim_users[user_id][:self.sim_user_num]
        sim_seq, sim_positions = [], []
        for sim_user in sim_users:
            meta_seq, meta_positions = self._get_user_seq(sim_user)
            sim_seq.append(meta_seq)
            sim_positions.append(meta_positions)
        
        sim_seq = np.array(sim_seq)
        sim_positions = np.array(sim_positions)

        return np.array(tokens), np.array(labels), np.array(neg_labels), np.array(pos), user_id, sim_seq, sim_positions


    def _get_user_seq(self, user):

        ### get the sequence of required user
        inter = self.data[user]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions

