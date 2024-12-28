from sklearn import metrics
from util.DataLoader import load_data, MyDataSet, load_graph_data, Dataset_mul, load_mul_data
import time
from time import strftime, localtime, time
import torch
from torch.utils.data import DataLoader
from util.early_stop import EarlyStopping
from util.tool import metric_new
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from data.process import *
from torch_geometric.data import DataLoader as dataloader_graph  # 导入 torch_geometric 自带的 DataLoader
from model.graphBased.GACL import GACL,FGM



class FNDLib():
    def __init__(self, Model, data, cfg, cls):
        """
        The Model is Fake news detection model,you can adjust the code according your needs
        :param Model: Fake news detection model
        :param cfg: detection parser
        """
        self.cfg = cfg
        print("cls:{}".format(cls))
        if cls == 0 or cls == 1:
            # Define data and training device
            train_data, test_data, val_data = data

            self.train_data = MyDataSet(train_data)
            self.train_loader = DataLoader(self.train_data, batch_size=cfg.batch_size, shuffle=True)

            self.test_data = MyDataSet(test_data)
            self.test_loader = DataLoader(self.test_data, batch_size=cfg.batch_size, shuffle=True)

            self.val_data = MyDataSet(val_data)
            self.val_loader = DataLoader(self.val_data, batch_size=cfg.batch_size, shuffle=True)

            
        if cls == 2:
        # Define data and training device
            self.train_data = load_mul_data(cfg, "train")
            print(self.train_data)
            self.test_data = load_mul_data(cfg, "test")
            self.val_data = load_mul_data(cfg, "val")
            print("train_len:{},test_len:{},val_len:{}".format(self.train_data.__len__(),
                    self.test_data.__len__(),self.val_data.__len__()))
            self.train_dataloader = DataLoader(self.train_data, batch_size=cfg.batch_size, shuffle=True)
            # print(train_dataloader)
            # train_dataloader = torch.tensor(train_dataloader,dtype=torch.float32)

            self.test_dataloader = DataLoader(self.test_data, batch_size=cfg.batch_size, shuffle=True)

            self.val_dataloader = DataLoader(self.val_data, batch_size=cfg.batch_size, shuffle=True)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.TDdroprate = cfg.TDdroprate
        self.BUdroprate = cfg.BUdroprate
        self.GACLdroprate = cfg.GACLdroprate

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
        self.cls = cls
        self.datasetname = cfg.datasetname

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
        self.loss_func = F.cross_entropy
        

    def Train(self):
        
        if self.cls == 0:
            self.train_text()
        if self.cls == 1:
            self.train_graph()
            # self.trian_graph_GACL()
        if self.cls == 2:
            self.trainer()
    
    def train_text(self):
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
    

    def trainer(self):
        print("*" * 80)
        print("Model is training")
        self.model.to(self.device)
        for epoch in range(self.epoch):
            self.model.train()
            total_loss = 0
            for batch_idx, (index, y_train, f_train, x_train) in enumerate(self.train_dataloader):
                x_train = torch.as_tensor(x_train, dtype=torch.float32).to(self.device)
                f_train = torch.as_tensor(f_train, dtype=torch.float32).to(self.device)
                y_train = torch.LongTensor(y_train.long()).to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x_train, f_train)
                loss = self.loss_func(logits, y_train)
                loss.backward()
                self.optimizer.step()
                total_loss += loss

            # early_stop
            self.early_stopping(total_loss, self.model)
            if self.early_stopping.early_stop:
                print('Early stopping!')
                break

            # 打印 epoch 和 loss
            print(f"epoch={epoch+1}, loss={total_loss.item()}")

        # 在最后一轮中计算并打印 acc
        self.model.eval()
        pred_list = []
        label = []
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (index, y_test, f_test, x_test) in enumerate(self.val_dataloader):
                x_test = torch.as_tensor(x_test, dtype=torch.float32).to(self.device)
                f_test = torch.as_tensor(f_test, dtype=torch.float32).to(self.device)
                y_test = torch.LongTensor(y_test.long()).to(self.device)

                logits = self.model(x_test, f_test)
                loss = self.loss_func(logits, y_test)
                total_loss += loss
                logit = F.softmax(logits, dim=1)
                pred = torch.argmax(logit, dim=1)
                pred_list.append(pred.cpu().detach().numpy())
                label.append(y_test.cpu().detach().numpy())

        pred_result = torch.cat([torch.tensor(i) for i in pred_list], 0).cpu().numpy()
        label_result = torch.cat([torch.tensor(i) for i in label], 0).cpu().numpy()
        
        # 计算准确率
        acc = metrics.accuracy_score(label_result, pred_result)
        total_loss_value = total_loss.item()

        print(f"epoch={epoch+1}, total_loss={total_loss_value}, acc={acc}")


    def train_graph(self):
        print("*" * 80)
        print("Model is training...")
        self.model.train()
        train_losses = []
        val_losses = []
        train_accs = []
        x_train, x_test, x_valid=load_graph_data(self.cfg)
        

        treeDic = loadTree(self.datasetname)

        
        for epoch in range(self.epoch):
            traindata_list, testdata_list = loadData_BiGCN(self.datasetname, treeDic, x_train, x_test, self.TDdroprate, self.BUdroprate)
            # traindata_list, testdata_list = loadData_GACL(self.datasetname, x_train, x_test, self.GACLdroprate)
            train_loader = dataloader_graph(traindata_list, batch_size=128, shuffle=True, num_workers=5)
            test_loader = dataloader_graph(testdata_list, batch_size=128, shuffle=True, num_workers=5)

            avg_loss = []
            avg_acc = []
            
            for Batch_data in tqdm(train_loader):
                Batch_data = Batch_data.to(self.device)
                out_labels = self.model(Batch_data)
                loss = F.nll_loss(out_labels, Batch_data.y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                avg_loss.append(loss.item())
                _, pred = out_labels.max(dim=-1)
                correct = pred.eq(Batch_data.y).sum().item()
                train_acc = correct / len(Batch_data.y)
                avg_acc.append(train_acc)
                
            train_losses.append(np.mean(avg_loss))
            train_accs.append(np.mean(avg_acc))

            val_loss = self.validate(test_loader)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1:05d} | Train Loss {np.mean(avg_loss):.4f} | Train Accuracy {np.mean(avg_acc):.4f} | Val Loss {val_loss:.4f}")

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print('Early stopping!')
                break
            
        return train_losses, val_losses, train_accs
    
    def trian_graph_GACL(self):

        # self.model.train()
        fgm = FGM(self.model)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.005, weight_decay=1e-3)

        # 冻结部分模型层的参数
        for para in self.model.hard_fc1.parameters():
            para.requires_grad = False
        for para in self.model.hard_fc2.parameters():
            para.requires_grad = False



        # 解除冻结，优化 hard_fc1 和 hard_fc2 层
        for para in self.model.hard_fc1.parameters():
            para.requires_grad = True
        for para in self.model.hard_fc2.parameters():
            para.requires_grad = True

        optimizer_hard = torch.optim.SGD([{'params': self.model.hard_fc1.parameters()},
                                        {'params': self.model.hard_fc2.parameters()}], lr=0.001)

        self.model.train()
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        # early_stopping = EarlyStopping(patience=10, verbose=True)
        x_train, x_test, x_valid=load_graph_data(self.cfg)


        for epoch in range(self.epoch):
            # 加载数据
            traindata_list, valdata_list = loadData_GACL(self.datasetname, x_train, x_valid, droprate=0.4)
            train_loader = dataloader_graph(traindata_list, batch_size=128, shuffle=True, num_workers=5)
            val_loader = dataloader_graph(valdata_list, batch_size=128, shuffle=True, num_workers=5)
            avg_loss = []
            avg_acc = []
            batch_idx = 0
            tqdm_train_loader = tqdm(train_loader)

            for Batch_data in tqdm_train_loader:
                Batch_data.to(self.device)
                out_labels, cl_loss, y = self.model(Batch_data)
                finalloss = F.nll_loss(out_labels, y)
                loss = finalloss + 0.001 * cl_loss
                avg_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                fgm.attack()
                out_labels, cl_loss, y = self.model(Batch_data)
                loss_adv = F.nll_loss(out_labels, y) + 0.001 * cl_loss
                loss_adv.backward()
                fgm.restore()
                optimizer.step()

                _, pred = out_labels.max(dim=-1)
                correct = pred.eq(y).sum().item()
                train_acc = correct / len(y)
                avg_acc.append(train_acc)

                print(f"Epoch {epoch:05d} | Batch {batch_idx:02d} | Train_Loss {loss.item():.4f} | Train_Accuracy {train_acc:.4f}")
                batch_idx += 1

            train_losses.append(np.mean(avg_loss))
            train_accs.append(np.mean(avg_acc))

            # 验证集评估
            temp_val_losses = []
            temp_val_accs = []
            self.model.eval()
            tqdm_val_loader = tqdm(val_loader)

            for Batch_data in tqdm_val_loader:
                Batch_data.to(self.device)
                val_out, val_cl_loss, y = self.model(Batch_data)
                val_loss = F.nll_loss(val_out, y)
                temp_val_losses.append(val_loss.item())

                _, val_pred = val_out.max(dim=1)
                correct = val_pred.eq(y).sum().item()
                val_acc = correct / len(y)
                temp_val_accs.append(val_acc)

            val_losses.append(np.mean(temp_val_losses))
            val_accs.append(np.mean(temp_val_accs))

            print(f"Epoch {epoch:05d} | Val_Loss {np.mean(temp_val_losses):.4f} | Val_Accuracy {np.mean(temp_val_accs):.4f}")

            # 保存当前的验证结果
            res = ['Validation Accuracy: {:.4f}'.format(np.mean(temp_val_accs))]
            print('Validation results:', res)

            # Early stopping check
            
            self.early_stopping(np.mean(temp_val_losses), self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        return np.mean(temp_val_accs), np.mean(temp_val_losses)
    
    def save_checkpoint(self, val_loss, model):
        '''
        Saves model at the end of training.
        '''
        print(f'Final validation loss: {val_loss:.6f}. Saving model...')
        model_path = self.model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model.state_dict(), os.path.join(model_path, 'checkpointC.pt'))


    def validate(self, loader):
        temp_val_losses = []
        self.model.eval()
        
        with torch.no_grad():
            for Batch_data in tqdm(loader):
                Batch_data.to(self.device)
                val_out = self.model(Batch_data)
                val_loss = F.nll_loss(val_out, Batch_data.y)
                temp_val_losses.append(val_loss.item())

        return np.mean(temp_val_losses)


    def DrawLoss(self):
        # draw the train loss
        pass

    def Test(self):
        
        if self.cls == 0:
            self.Test_text()
        if self.cls == 1:
            self.Test_graph()
        if self.cls == 2:
            self.Test_mul()
        

    ####------------------Test_mul-------------------####
    # def Test_mul(self):
    #     self.model.eval()
    #     # label = np.array(label)
        
    #     logit = self.model(x_feature, f_feature)
    #     logit = F.softmax(logit, dim=1)
    #     pred = torch.argmax(logit, dim=1)
    #     a = pred.cpu().detach().numpy()
    #     return a
    def Test_mul(self):
        print("*" * 80)
        print("Model is testing")
        self.model.eval()
        pred_list = []
        pred_old_list = []
        label = []
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (index, y_test, f_test, x_test) in enumerate(self.val_dataloader):
                x_test = torch.as_tensor(x_test, dtype=torch.float32).to(self.device)
                f_test = torch.as_tensor(f_test, dtype=torch.float32).to(self.device)
                y_test = torch.LongTensor(y_test.long()).to(self.device)

                logits = self.model(x_test, f_test)
                loss = self.loss_func(logits, y_test)
                total_loss += loss
                logit = F.softmax(logits, dim=1)
                pred = torch.argmax(logit, dim=1)
                pred_list.append(pred.cpu().detach().numpy())
                pred_old_list.append(logits.cpu().detach())
                label.append(y_test.cpu().detach().numpy())

        pred_result = torch.cat([torch.tensor(i) for i in pred_list], 0).cpu().numpy()
        pred_old_result = torch.cat(pred_old_list, 0)
        label_result = torch.cat([torch.tensor(i) for i in label], 0).cpu().numpy()
        
        # 将 total_loss 转换为数值类型
        total_loss_value = total_loss.item()
        Acc, pre, Rec, F1, auc = metric_new(pred_result, pred_old_result, label_result, 'test')

        print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(Acc, pre, Rec, F1))


    def Test_text(self):
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
    
    def Test_graph(self):
        print("*" * 80)
        print("Model is testing")
        test_pred_all, test_pred_one_batch, test_y_batch = [], [], []
        
        x_train, x_test, x_valid = load_graph_data(self.cfg)
        treeDic = loadTree(self.datasetname)

        # 仅加载测试集数据
        _, testdata_list = loadData_BiGCN(self.datasetname, treeDic, x_train, x_test, self.TDdroprate, self.BUdroprate)
        # _, testdata_list = loadData_GACL(self.datasetname, x_train, x_valid, droprate=0.4)
        test_loader = dataloader_graph(testdata_list, batch_size=128, shuffle=False, num_workers=5)

        with torch.no_grad():
            self.model.eval()
            for i, Batch_data in enumerate(test_loader):
                # 将数据加载到设备
                Batch_data.to(self.device)
                
                # 模型前向传播
                out_labels = self.model(Batch_data)
                out_labels = torch.squeeze(out_labels)
                print("out_label", out_labels)
                
                # 存储预测和真实标签
                test_pred_all.append(out_labels)
                pred_one = torch.argmax(out_labels, dim=1)
                test_pred_one_batch.append(pred_one)
                print("test_pred_one_batch", test_pred_one_batch)
                test_y_batch.append(Batch_data.y)
                print("test_y_batch", test_y_batch)

            # 将预测和标签转换为张量
            pred_one = torch.cat([i for i in test_pred_one_batch], dim=0).cpu()
            pred_all = torch.cat([i for i in test_pred_all], dim=0).cpu()
            test_y_data_epoch = torch.cat([i for i in test_y_batch], dim=0).cpu()

            # 计算评估指标
            pre, Rec, F1, Acc, auc = metric_new(pred_one, pred_all, test_y_data_epoch, 'test')
            
            # 打印评估结果
            print(f"Test Accuracy: {Acc:.3f}\tPrecision: {pre:.3f}\tRecall: {Rec:.3f}\tF1 Score: {F1:.3f}\t")
        

        ####### 测试GACL  #################
        # print("*" * 80)
        # print("Model is testing")
        # test_pred_all, test_pred_one_batch, test_y_batch = [], [], []
        
        # x_train, x_test, x_valid = load_graph_data(self.cfg)

        # # 仅加载测试集数据
        # _, testdata_list = loadData_GACL(self.datasetname, x_train, x_test, droprate=0.4)
        # test_loader = dataloader_graph(testdata_list, batch_size=128, shuffle=False, num_workers=5)
        

        # with torch.no_grad():
        #     self.model.eval()
        #     for i, Batch_data in enumerate(test_loader):
        #         # 将数据加载到设备
        #         Batch_data.to(self.device)
                
        #         # 模型前向传播
        #         out_labels,_,_ = self.model(Batch_data)
        #         out_labels = torch.squeeze(out_labels)
        #         print("out_labels", out_labels)
                
        #         # 存储预测和真实标签
        #         test_pred_all.append(out_labels)
        #         pred_one = torch.argmax(out_labels, dim=1)
        #         test_pred_one_batch.append(pred_one)
        #         print("test_pred_one_batch", test_pred_one_batch)
        #         y_combined = torch.cat((Batch_data.y1, Batch_data.y2), dim=0)
        #         test_y_batch.append(y_combined)
        #         print("test_y_batch", test_y_batch)
        #         # if Batch_data.y is not None:
        #         #     test_y_batch.append(Batch_data.y)
        #         # else:
        #         #     print("Warning: Batch_data.y is None")

        #         # print("test_y_batch", test_y_batch)

        #     # 将预测和标签转换为张量
        #     pred_one = torch.cat([i for i in test_pred_one_batch], dim=0).cpu()
        #     pred_all = torch.cat([i for i in test_pred_all], dim=0).cpu()
        #     test_y_data_epoch = torch.cat([i for i in test_y_batch], dim=0).cpu()

        #     # 计算评估指标
        #     pre, Rec, F1, Acc, auc = metric_new(pred_one, pred_all, test_y_data_epoch, 'test')
            
        #     # 打印评估结果
        #     print(f"Test Accuracy: {Acc:.3f}\tPrecision: {pre:.3f}\tRecall: {Rec:.3f}\tF1 Score: {F1:.3f}\t")
        
        

    def anyNewMethod(self):
        pass



