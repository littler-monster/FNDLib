import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import json
import pickle

class GraphDataset_Bi(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

def collate_fn(data):
    return data

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Twitter15graph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..','data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
    

# global
label2id = {
            "unverified": 0,
            "non-rumor": 1,
            "true": 2,
            "false": 3,
            }

def random_pick(list, probabilities): 
    x = random.uniform(0,1)
    cumulative_probability = 0.0 
    for item, item_probability in zip(list, probabilities): 
         cumulative_probability += item_probability 
         if x < cumulative_probability:
               break 
    return item 

class RumorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])#torch.LongTensor(batch_label).to(device1)
        #item['labels'] = torch.LongTensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class GraphDataset_GACL(Dataset):
    def __init__(self, fold_x, droprate): 

        
        self.fold_x = fold_x
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 

        id = self.fold_x[index]
        # print("foldx",self.fold_x)
        # print("index", index)
        # print(self.fold_x)
    

        # ====================================edgeindex========================================
        with open('./data/Twitter16/twitter16/' + id + '/after_tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)

        # 创建 dict
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index

        # 加载图的结构
        with open('./data/Twitter16/twitter16/' + id + '/after_structure.pkl', 'rb') as f:
            inf = pickle.load(f)

        # 进一步处理
        inf = inf[1:]
        new_inf = []
        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)
        new_inf = np.array(new_inf).T
        edgeindex = new_inf

        # 初始化边的索引
        init_row = list(edgeindex[0]) 
        init_col = list(edgeindex[1]) 
        burow = list(edgeindex[1]) 
        bucol = list(edgeindex[0]) 
        row = init_row + burow 
        col = init_col + bucol
        new_edgeindex = [row, col]

        # 处理丢弃、添加、误置操作
        choose_list = [1, 2, 3]  # 1-drop, 2-add, 3-misplace
        probabilities = [0.7, 0.2, 0.1]
        choose_num = random_pick(choose_list, probabilities)

        if self.droprate > 0:
            if choose_num == 1:
                length = len(row)
                poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                poslist = sorted(poslist)
                row2 = list(np.array(row)[poslist])
                col2 = list(np.array(col)[poslist])
                new_edgeindex2 = [row2, col2]
            elif choose_num == 2:
                length = len(list(set(sorted(row))))
                add_row = random.sample(range(length), int(length * self.droprate))
                add_col = random.sample(range(length), int(length * self.droprate))
                row2 = row + add_row + add_col
                col2 = col + add_col + add_row
                new_edgeindex2 = [row2, col2]
            elif choose_num == 3:
                length = len(init_row)
                mis_index_list = random.sample(range(length), int(length * self.droprate))
                Sort_len = len(list(set(sorted(row))))
                if Sort_len > int(length * self.droprate):
                    mis_value_list = random.sample(range(Sort_len), int(length * self.droprate))
                    for i, item in enumerate(init_row):
                        for mis_i, mis_item in enumerate(mis_index_list):
                            if i == mis_item and mis_value_list[mis_i] != item:
                                init_row[i] = mis_value_list[mis_i]
                    row2 = init_row + init_col
                    col2 = init_col + init_row
                    new_edgeindex2 = [row2, col2]
                else:
                    length = len(row)
                    poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                    poslist = sorted(poslist)
                    row2 = list(np.array(row)[poslist])
                    col2 = list(np.array(col)[poslist])
                    new_edgeindex2 = [row2, col2]
        else:
            new_edgeindex2 = [row, col]

        # =========================================X===============================================
        with open('./data/bert_w2c/t16_mask_00/' + id + '.json', 'r') as j_f0:
            json_inf0 = json.load(j_f0)

        x0 = json_inf0[id]
        x0 = np.array(x0)

        with open('./data/bert_w2c/t16_mask_015/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)

        x_list = json_inf[id]
        x = np.array(x_list)

        with open('./data/Twitter16/label_16.json', 'r') as j_tags:
            tags = json.load(j_tags)

        y = label2id[tags[id]]

        if self.droprate > 0:
            if choose_num == 1:
                zero_list = [0] * 768
                x_length = len(x_list)
                r_list = random.sample(range(x_length), int(x_length * self.droprate))
                r_list = sorted(r_list)
                for idx, line in enumerate(x_list):
                    for r in r_list:
                        if idx == r:
                            x_list[idx] = zero_list
                x2 = np.array(x_list)
                x = x2

        return Data(
            x0=torch.tensor(x0, dtype=torch.float32),
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.LongTensor(new_edgeindex),
            edge_index2=torch.LongTensor(new_edgeindex2),
            y1=torch.LongTensor([y]),
            y2=torch.LongTensor([y])
        )
 



class test_GraphDataset_GACL(Dataset):
    def __init__(self, fold_x, droprate): 
        
        self.fold_x = fold_x
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        id =self.fold_x[index] 
        # ====================================edgeindex==============================================
        with open('./data/Twitter16/twitter16/'+ id + '/after_tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        #print(tweets)
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
        #print('dict: ', dict)

        with open('./data/Twitter16/twitter16/'+ id + '/after_structure.pkl', 'rb') as f:
            inf = pickle.load(f)

        inf = inf[1:]
        new_inf = []
        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)
        new_inf = np.array(new_inf).T
        edgeindex = new_inf
        
        row = list(edgeindex[0]) 
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)

        new_edgeindex = [row, col] 
        new_edgeindex2 = [row, col]


        # =========================================X====================================================
        with open('./data/bert_w2c/t16_mask_00/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)
        
        x = json_inf[id]
        x = np.array(x)

        with open('./data/Twitter16/label_16.json', 'r') as j_tags:
            tags = json.load(j_tags)

        y = label2id[tags[id]]
        #y = np.array(y)


        return Data(x0=torch.tensor(x,dtype=torch.float32),
                    x=torch.tensor(x,dtype=torch.float32), 
                    edge_index=torch.LongTensor(new_edgeindex),
                    edge_index2=torch.LongTensor(new_edgeindex2), 
                    y1=torch.LongTensor([y]),
                    y2=torch.LongTensor([y]))
