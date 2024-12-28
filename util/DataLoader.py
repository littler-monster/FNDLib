# this file contain load News body,comment,label.
# if you don't need one mentioned above,you can mask corresponding code

from torch.utils.data import Dataset, DataLoader
from data.dataset import GraphDataset_GACL, test_GraphDataset_GACL
import numpy as np
import torch
import os
import random
from random import shuffle


def load_data(cfg):
    print("load News Body")
    news_X = np.load(cfg.news_npy_path_p_7)

    print("load News label")
    data_Y = np.load(cfg.label_npy_path_p_7)

    # Split Train,Test,Val datasets
    train_size = int(len(data_Y) * (1 - cfg.percent_of_test - cfg.percent_of_val))
    test_size = int(len(data_Y) * cfg.percent_of_test)

    train_news_X = news_X[0:train_size]
    test_news_X = news_X[train_size:train_size + test_size]
    val_news_X = news_X[train_size + test_size:len(news_X)]

    train_Y = data_Y[0:train_size]
    test_Y = data_Y[train_size:train_size + test_size]
    val_Y = data_Y[train_size + test_size:len(data_Y)]

    if cfg.comments_need:
        print("load News Comments")
        comment_X = np.load(cfg.comment_npy_path_p_7)

        train_comment_X = comment_X[0:train_size]
        test_comment_X = comment_X[train_size:train_size + test_size]
        val_comment_X = comment_X[train_size + test_size:len(comment_X)]

        train_data = list(zip(train_Y, train_news_X, train_comment_X))
        test_data = list(zip(test_Y, test_news_X, test_comment_X))
        val_data = list(zip(val_Y, val_news_X, val_comment_X))

    else:

        train_data = list(zip(train_Y, train_news_X))
        test_data = list(zip(test_Y, test_news_X))
        val_data = list(zip(val_Y, val_news_X))

    print("*" * 80)
    return train_data, test_data, val_data

def stratified_split(data, labels, test_ratio=0.2, val_ratio=0.1):
    """
    根据标签进行分层抽样，返回训练集、验证集和测试集的索引。
    
    :param data: 输入数据列表（可以是样本ID等）
    :param labels: 标签列表（与data长度一致）
    :param test_ratio: 测试集比例
    :param val_ratio: 验证集比例
    :return: 返回训练集、验证集、测试集的索引
    """
    # 创建标签到数据的映射
    label_dict = {}
    for i, label in enumerate(labels):
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(data[i])

    # 初始化训练集、验证集、测试集
    train_set, val_set, test_set = [], [], []

    # 分层划分
    for label, instances in label_dict.items():
        # 打乱数据
        random.shuffle(instances)
        
        # 计算每个子集的大小
        n_total = len(instances)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        
        test_set += instances[:n_test]
        val_set += instances[n_test:n_test + n_val]
        train_set += instances[n_test + n_val:]

    # 返回分层后的数据集
    return train_set, val_set, test_set

def load_graph_data(cfg):

    cwd = os.getcwd()
    test_ratio = 0.2
    val_ratio = 0.2
    train_set, val_set, test_set = [], [], []


    labelPath = os.path.join(cwd, cfg.graph_label_path)
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    
    NR, F, T, U = [], [], [], []
    labelDic = {}
    """
    普通加载数据方法，将数据集划分为训练集、验证集和测试集，不使用五折验证。
    
    :param obj: 数据集名称 (例如 "Twitter15" 或 "Weibo")
    :return: 训练集列表、验证集列表和测试集列表
    """
    if cfg.datasetname == "Twitter15":
        # 固定测试集和验证集比例
        print("loading Twitter tree label")
        
        for line in open(labelPath):
            line = line.rstrip()
            label, eid = line.split('\t')[0], line.split('\t')[2]
            labelDic[eid] = label.lower()
            if label in labelset_nonR:
                NR.append(eid)
            elif label in labelset_f:
                F.append(eid)
            elif label in labelset_t:
                T.append(eid)
            elif label in labelset_u:
                U.append(eid)

        print("Total labels:", len(labelDic))
        random.shuffle(NR)
        random.shuffle(F)
        random.shuffle(T)
        random.shuffle(U)

        # 根据 test_ratio 和 val_ratio 分割训练集、验证集和测试集
        NR_test = NR[:int(len(NR) * test_ratio)]
        NR_val = NR[int(len(NR) * test_ratio):int(len(NR) * (test_ratio + val_ratio))]
        NR_train = NR[int(len(NR) * (test_ratio + val_ratio)):]
        
        F_test = F[:int(len(F) * test_ratio)]
        F_val = F[int(len(F) * test_ratio):int(len(F) * (test_ratio + val_ratio))]
        F_train = F[int(len(F) * (test_ratio + val_ratio)):]
        
        T_test = T[:int(len(T) * test_ratio)]
        T_val = T[int(len(T) * test_ratio):int(len(T) * (test_ratio + val_ratio))]
        T_train = T[int(len(T) * (test_ratio + val_ratio)):]
        
        U_test = U[:int(len(U) * test_ratio)]
        U_val = U[int(len(U) * test_ratio):int(len(U) * (test_ratio + val_ratio))]
        U_train = U[int(len(U) * (test_ratio + val_ratio)):]

        # 合并各类别的训练集、验证集和测试集
        train_set = NR_train + F_train + T_train + U_train
        val_set = NR_val + F_val + T_val + U_val
        test_set = NR_test + F_test + T_test + U_test
        shuffle(train_set)
        shuffle(val_set)
        shuffle(test_set)
        return train_set, val_set, test_set


    elif cfg.datasetname == "Twitter16":

        path = cfg.data_path_GACL
        label_path = cfg.graph_label_path
        labelPath = os.path.join(label_path)
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['non-rumor'], ['false'], ['true'], ['unverified']
        t_path = path
        file_list = os.listdir(t_path)
        print('The len of file_list: ', len(file_list))

        NR, F, T, U = [], [], [], []
        labelDic = {} 
        for line in open(labelPath): 
            line = line.rstrip() 
            label, eid = line.split('\t')[0], line.split('\t')[2] 

            if eid in file_list:
                labelDic[eid] = label.lower() 
                    
                if label in labelset_nonR: 
                    NR.append(eid)
                if labelDic[eid] in labelset_f: 
                    F.append(eid)
                if labelDic[eid] in labelset_t: 
                    T.append(eid)
                if labelDic[eid] in labelset_u: 
                    U.append(eid)
        print("Total samples:", len(labelDic))

        random.shuffle(NR)
        random.shuffle(F)
        random.shuffle(T)
        random.shuffle(U)

        def split_data(data, test_ratio, val_ratio):
            test_len = int(len(data) * test_ratio)
            val_len = int((len(data) - test_len) * val_ratio)
            test_data = data[:test_len]
            val_data = data[test_len:test_len + val_len]
            train_data = data[test_len + val_len:]
            return train_data, val_data, test_data

        train_NR, val_NR, test_NR = split_data(NR, test_ratio, val_ratio)
        train_F, val_F, test_F = split_data(F, test_ratio, val_ratio)
        train_T, val_T, test_T = split_data(T, test_ratio, val_ratio)
        train_U, val_U, test_U = split_data(U, test_ratio, val_ratio)

        train_data = train_NR + train_F + train_T + train_U
        val_data = val_NR + val_F + val_T + val_U
        test_data = test_NR + test_F + test_T + test_U

        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        return train_data, val_data, test_data


        # file_list = os.listdir(labelPath)
        # print('The len of file_list: ', len(file_list))
        # with open(labelPath) as f:
        #     for line in f:
        #         label, eid = line.strip().split('\t')[0], line.strip().split('\t')[2]
        #         labelDic[eid] = label.lower()
        #         if label in labelset_nonR: 
        #             NR.append(eid)
        #         elif label in labelset_f: 
        #             F.append(eid)
        #         elif label in labelset_t: 
        #             T.append(eid)
        #         elif label in labelset_u: 
        #             U.append(eid)

        # # 打乱各类样本
        # for lst in [NR, F, T, U]: 
        #     random.shuffle(lst)

        # # 分层抽样划分数据集
        # NR_train, NR_val, NR_test = stratified_split(NR, [labelDic[eid] for eid in NR], test_ratio, val_ratio)
        # F_train, F_val, F_test = stratified_split(F, [labelDic[eid] for eid in F], test_ratio, val_ratio)
        # T_train, T_val, T_test = stratified_split(T, [labelDic[eid] for eid in T], test_ratio, val_ratio)
        # U_train, U_val, U_test = stratified_split(U, [labelDic[eid] for eid in U], test_ratio, val_ratio)

        # # 汇总所有数据
        # train_ids = NR_train + F_train + T_train + U_train
        # val_ids = NR_val + F_val + T_val + U_val
        # test_ids = NR_test + F_test + T_test + U_test

        # # 打乱最终的数据集
        # random.shuffle(train_ids)
        # random.shuffle(val_ids)
        # random.shuffle(test_ids)

        # train_ids = [eid if isinstance(eid, str) else eid[0] for eid in train_ids]

        # # check_all_strings(train_ids, "train_ids")
        # # check_all_strings(val_ids, "val_ids")
        # # check_all_strings(test_ids, "test_ids") 
  
        
        # # # 假设你有 GraphDataset_GACL 来构建训练集、验证集和测试集
        # # train_set = GraphDataset_GACL(train_ids, droprate=cfg.GACLdroprate)  # 假设训练集的 drop_rate 为 0.2
        # # val_set = GraphDataset_GACL(val_ids, droprate=0)  # 验证集和测试集的 drop_rate 为 0
        # # test_set = GraphDataset_GACL(test_ids, droprate=0)
        
        # # # print("train_set", train_set[0])
  
        # # print(f"Loaded {len(train_set)} training samples, {len(val_set)} validation samples, {len(test_set)} test samples.")
    
        # return train_ids, val_ids, test_ids

def load_mul_data(cfg, mode):
    print("load News Body")
    data_bert_list_path =  cfg.mul_politic_body
    print("load News img")
    data_vgg_list_path = cfg.mul_politic_img
    print("load News label")
    data_label_list_path = cfg.mul_politic_label
    

    datasets = Dataset_mul(data_label_list_path, data_vgg_list_path, data_bert_list_path, mode)


    return datasets


# 检查 train_ids, val_ids 和 test_ids 是否全部为字符串
def check_all_strings(ids, name):
    all_strings = all(isinstance(i, str) for i in ids)
    if not all_strings:
        non_string_items = [i for i in ids if not isinstance(i, str)]
        print(f"{name} contains non-string items: {non_string_items[:5]}")  # 打印前5个非字符串项
    else:
        print(f"All items in {name} are strings.")


class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        assert item < len(self.data)
        data = self.data[item]
        label = torch.tensor(data[0], dtype=torch.float32)
        words = torch.tensor(data[1], dtype=torch.float32)
        # judge the comments exist?
        if len(data) > 2:
            comments = torch.tensor(data[2], dtype=torch.float32)
            return label, words, comments
        else:
            return label, words
        # according to your needs, adjust the return value
        # return label, words

class Dataset_mul(Dataset):
    def __init__(self, data_label_list_path, data_vgg_list_path, data_bert_list_path, mode='train'):
        
        data_labels = np.load(data_label_list_path)
        data_vgg = np.load(data_vgg_list_path,allow_pickle=True)
        x_features = np.load(data_bert_list_path)
        
        train_size = int(len(data_labels)*0.7)
        val_size = int(len(data_labels)*0.1)
        test_size = int(len(data_labels)*0.2)

        # print('train:{},val:{},test:{}'.format(train_size,val_size,test_size))

        train_data_labels = data_labels[0:train_size]
        train_data_vgg = data_vgg[0:train_size]
        train_x_features = x_features[0:train_size]

        test_data_labels = data_labels[train_size:train_size+test_size]
        test_data_vgg = data_vgg[train_size:train_size+test_size]
        test_x_features = x_features[train_size:train_size+test_size]

        val_data_labels = data_labels[train_size:train_size+val_size]
        val_data_vgg = data_vgg[train_size:train_size+val_size]
        val_x_features = x_features[train_size:train_size+val_size]

        test_data_labels = data_labels[train_size+val_size:train_size+val_size+test_size]
        test_data_vgg = data_vgg[train_size+val_size:train_size+val_size+test_size]
        test_x_features = x_features[train_size+val_size:train_size+val_size+test_size]

        # if index_id != None:
        #     data_labels = data_labels[index_id]
        #     data_vgg = data_vgg[index_id]
        #     x_features = x_features[index_id]
        
        if mode == 'train':
            self.data_labels = train_data_labels
            self.data_vgg = train_data_vgg
            self.x_features = train_x_features
        elif mode == 'val':
            self.data_labels = val_data_labels
            self.data_vgg = val_data_vgg
            self.x_features = val_x_features
        else:
            self.data_labels = test_data_labels
            self.data_vgg = test_data_vgg
            self.x_features = test_x_features


    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, index):
        return index, \
               self.data_labels[index], \
               self.data_vgg[index], \
               self.x_features[index]


