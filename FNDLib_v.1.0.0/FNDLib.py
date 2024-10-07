from util.DataLoader import load_data, MyDataSet
import time
from time import strftime, localtime, time
import torch
from torch.utils.data import DataLoader
from util.early_stop import EarlyStopping
from util.tool import metric_new


class FNDLib():
    def __init__(self, Model, data, cfg):
        """
        The Model is Fake news detection model,you can adjust the code according your needs
        :param Model: Fake news detection model
        :param cfg: detection parser
        """

        # Define data and training device
        train_data, test_data, val_data = data

        self.train_data = MyDataSet(train_data)
        self.train_loader = DataLoader(self.train_data, batch_size=cfg.batch_size, shuffle=True)

        self.test_data = MyDataSet(test_data)
        self.test_loader = DataLoader(self.test_data, batch_size=cfg.batch_size, shuffle=True)

        self.val_data = MyDataSet(val_data)
        self.val_loader = DataLoader(self.val_data, batch_size=cfg.batch_size, shuffle=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if cfg.device == 'cpu':
            print("adopt cpu to training")
        else:
            if str(self.device) != cfg.device:
                print("cuda is not work, automatic adjust to cpu ")
                cfg.device = self.device
            print("adopt cuda to training")

        self.model = Model
        self.model.to(cfg.device)
        self.device = cfg.device
        self.comments_need = cfg.comments_need

        # Define the loss and metric
        self.draw_train_loss = []
        self.draw_val_loss = []
        self.draw_val_F1 = []
        self.draw_val_ACC = []
        self.draw_val_AUC = []
        self.min_val_loss = 10.0

        # Define train strategy
        self.epoch = cfg.maxEpoch
        self.optimizer = torch.optim.Adam(Model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
        self.early_stopping = EarlyStopping(patience=cfg.patience, verbose=True, model_path=cfg.model_name)
        self.loss_fun_all = torch.nn.CrossEntropyLoss()

    def Train(self):
        print("*" * 80)
        print("Model is training...")
        # train_comments_data is a judge value, you don't need adjust it
        train_comments_data = None
        for epoch in range(self.epoch):
            self.model.train()
            epoch_loss_train = 0.0
            for i, batch in enumerate(self.train_loader):
                # need add some log file.
                if self.comments_need:
                    train_y_data, train_news_data, train_comments_data = batch
                    train_comments_data = torch.as_tensor(train_comments_data).to(self.device)
                else:
                    train_y_data, train_news_data = batch

                train_y_data = torch.as_tensor(train_y_data).to(self.device)
                train_news_data = torch.as_tensor(train_news_data).to(self.device)

                self.optimizer.zero_grad()

                pred_all = self.model.forward(train_news_data, train_comments_data)
                pred_all = torch.squeeze(pred_all)
                train_y_data = train_y_data.long()

                batch_loss_train = self.loss_fun_all(pred_all, train_y_data)
                epoch_loss_train += batch_loss_train.item()

                batch_loss_train.backward()
                self.optimizer.step()
            self.draw_train_loss.append(epoch_loss_train)
            print("******epoch:{}, train_loss:{:.6f}******".format(epoch + 1, epoch_loss_train))

            # Early Stopping in val datasets
            with torch.no_grad():
                self.model.eval()
                val_comment_data = None
                epoch_loss_val = 0.0
                pred_epoch = []
                pred_one_epoch = []
                val_y_epoch = []
                for i, batch in enumerate(self.val_loader):

                    if self.comments_need:
                        val_y_data, val_news_data, val_comment_data = batch
                        val_comment_data = torch.as_tensor(val_comment_data, dtype=torch.float32).to(self.device)
                    else:
                        val_y_data, val_news_data = batch

                    val_y_data = torch.as_tensor(val_y_data, dtype=torch.long).to(self.device)
                    val_news_data = torch.as_tensor(val_news_data, dtype=torch.float32).to(self.device)

                    pred_all = self.model.forward(val_news_data, val_comment_data)
                    pred_all = torch.squeeze(pred_all)

                    batch_loss_val = self.loss_fun_all(pred_all, val_y_data)

                    # epoch_loss_val += val_y_data.size()[0] * batch_loss_val.item() # 为什么要相乘 ***
                    epoch_loss_val += batch_loss_val.item()
                    # predict final result
                    pred_one = torch.argmax(pred_all, dim=1)
                    pred_epoch.append(pred_all)
                    pred_one_epoch.append(pred_one)
                    val_y_epoch.append(val_y_data)

                pred_one = torch.cat([i for i in pred_one_epoch], 0).cpu()
                pred = torch.cat([i for i in pred_epoch], 0).cpu()
                val_y_data = torch.cat([i for i in val_y_epoch], 0).cpu()
                pre, Rec, F1, Acc, Auc = metric_new(pred_one, pred, val_y_data, 'val')
                # epoch_loss_val /= len(val_data)
                self.draw_val_loss.append(epoch_loss_val)
                self.draw_val_F1.append(F1)
                self.draw_val_ACC.append(Acc)
                self.draw_val_AUC.append(Auc)
                print("******epoch:{}, val_loss:{:.6f}, pre:{}, rec:{}, F1:{}, Acc:{}, auc:{}******".format
                      (epoch + 1, epoch_loss_val, pre, Rec, F1, Acc, Auc))

                # early_stop
                self.early_stopping(epoch_loss_val, self.model)
                if self.early_stopping.early_stop:
                    print('Early stopping!')
                    break

    def DrawLoss(self):
        # draw the train loss
        pass

    def Test(self):
        print("*" * 80)
        print("Model is testing")
        test_comments_data = None
        pred_epoch, pred_one_batch, test_y_batch = [], [], []

        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(self.test_loader):

                if self.comments_need:
                    test_y_data, test_news_data, test_comments_data = batch
                    test_comments_data = torch.as_tensor(test_comments_data).to(self.device)
                else:
                    test_y_data, test_news_data = batch

                test_y_data = torch.as_tensor(test_y_data).to(self.device)
                test_news_data = torch.as_tensor(test_news_data).to(self.device)
                pred_all = self.model.forward(test_news_data, test_comments_data)
                pred_all = torch.squeeze(pred_all)

                pred_epoch.append(pred_all)
                pred_one = torch.argmax(pred_all, dim=1)

                pred_one_batch.append(pred_one)
                test_y_batch.append(test_y_data)

            pred_one = torch.cat([i for i in pred_one_batch], 0).cpu()
            pred = torch.cat([i for i in pred_epoch], 0).cpu()
            test_y_data_epoch = torch.cat([i for i in test_y_batch], 0).cpu()

            pre, Rec, F1, Acc, auc = metric_new(pred_one, pred, test_y_data_epoch, 'test')
            print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(Acc, pre, Rec, F1))

    def anyNewMethod(self):
        pass
