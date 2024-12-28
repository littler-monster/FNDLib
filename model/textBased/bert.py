import torch
import numpy as np
import pandas as pd
import torch.utils
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader



local_model_path = '/usr/gao/gubincheng/article_rep/ENDEF-SIGIR2022/ENDEF-SIGIR2022-main/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(local_model_path)

example_text = 'I will watch Memento tonight '
bert_input = tokenizer(example_text,padding='max_length', 
                       max_length = 10, 
                       truncation=True,
                       return_tensors="pt")

# ------- bert_input ------
# print(bert_input['input_ids'])
# print(bert_input['token_type_ids']) # 是一个 binary mask，用于标识 token 属于哪个 sequence,如果我们只有一个 sequence，那么所有的 token 类型 id 都将为 0。对于文本分类任务，token_type_ids是 BERT 模型的可选输入参数。
# print(bert_input['attention_mask']) # 它是一个 binary mask，用于标识 token 是真实 word 还是只是由填充得到。如果 token 包含 [CLS]、[SEP] 或任何真实单词，则 mask 将为 1。如果 token 只是 [PAD] 填充，则 mask 将为 0。

# example_text = tokenizer.decode(bert_input.input_ids[0])
# print(example_text) # [CLS] i will watch memento tonight [SEP] [PAD] [PAD]


labels = {'business':0,
          'entertainment':1,
          'sport':2,
          'tech':3,
          'politics':4
          }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['type']]
        self.texts = [tokenizer(text, padding='max_length',
                                max_length = 10,
                                truncation = True,
                                return_tensors = "pt")
                        for text in df['news']]
    
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])
    
    def get_batch_text(self, idx):
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_text(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

# with open('/usr/gao/gubincheng/article_rep/bert微调/data/BBC/dataset.csv', 'rb') as f:
#     result = chardet.detect(f.read())
#     print("编码类型", result)

np.random.seed(112)
bbc_text_df = pd.read_csv(
    '/usr/gao/gubincheng/article_rep/bert微调/data/BBC/dataset.csv',
    encoding='ISO-8859-1'
)

# bbc_text_df.head()
df = pd.DataFrame(bbc_text_df)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])

print(len(df_train),len(df_val), len(df_test))


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(local_model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768 ,5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        # (batch_size, sequence_length, hidden_size),(batch_size, hidden_size) bert的返回值
        # 其中第一个返回值包括token的上下文表示，适用于序列标注任务；文本分类任务只使用句子嵌入向量即可
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_output = self.relu(linear_output)
        return final_output


def train(model, train_data, val_data, lr, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    # 使用DataLoader根据batch_size获取数据并随机打乱
    train_dataloader = DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=2, shuffle=True)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    if use_cuda:
        model = model.to(device)
        criterion = criterion.to(device)
    
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0

        optimizer.zero_grad()
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
        # 通过模型得到输出
            output = model(input_id, mask)
            loss = criterion(output, train_label)
            total_loss_train += loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            # model.zero_grad()
            loss.backward()
            optimizer.step()
        # ------ 验证模型 -----------
            # 定义两个变量，用于存储验证集的准确率和损失
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():
                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''') 

EPOCHS = 5
model = BertClassifier()
LR = 1e-6
train(model, df_train, df_val, LR, EPOCHS)
torch.save(model.state_dict(), 'bert_classifier_weights_1.pth')

def evaluate(model_path, test_data):
    model = BertClassifier()

    # 加载已保存的模型权重
    model.load_state_dict(torch.load(model_path))

    # 创建数据加载器
    test = Dataset(test_data)
    test_dataloader = DataLoader(test, batch_size=2)

    # 判断是否使用 GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    if use_cuda:
        model = model.to(device)

    # 评估模型
    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

# 调用 evaluate 函数
model_path = "/usr/gao/gubincheng/article_rep/bert微调/code/bert_classifier_weights_1.pth"  # 保存的模型文件路径
evaluate(model_path, df_test)