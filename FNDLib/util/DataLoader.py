# this file contain load News body,comment,label.
# if you don't need one mentioned above,you can mask corresponding code

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


def load_data(cfg):
    print("load News Body")
    news_X = np.load(cfg.news_npy_path)

    print("load News label")
    data_Y = np.load(cfg.label_npy_path)

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
        comment_X = np.load(cfg.comment_npy_path)

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
