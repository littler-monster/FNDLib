import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import torch.nn.functional as F

# 引入FakeBERT模型
class FakeBERT(nn.Module):
    def __init__(self, num_classes=2):
        super(FakeBERT, self).__init__()
        # BERT层
        local_model_path = '/usr/gao/gubincheng/article_rep/ENDEF-SIGIR2022/ENDEF-SIGIR2022-main/bert-base-uncased'
        self.bert = BertModel.from_pretrained(local_model_path)
        
        # 并行的1D-CNN模块
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=0)  # 输入1000，输出998
        self.conv2 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=4, padding=0)  # 输入1000，输出997
        self.conv3 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=5, padding=0)  # 输入1000，输出996
        
        # 最大池化层
        self.maxpool1 = nn.MaxPool1d(kernel_size=5, stride=5)  # 输出199
        self.maxpool2 = nn.MaxPool1d(kernel_size=5, stride=5)  # 输出199
        self.maxpool3 = nn.MaxPool1d(kernel_size=5, stride=5)  # 输出199
        
        # 拼接后的卷积层
        self.conv_final1 = nn.Conv1d(in_channels=128 * 3, out_channels=128, kernel_size=5, padding=0)  # 输入597，输出593
        self.maxpool_final1 = nn.MaxPool1d(kernel_size=5, stride=5)  # 输出118
        
        self.conv_final2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=0)  # 输入118，输出114
        self.maxpool_final2 = nn.MaxPool1d(kernel_size=5, stride=5)  # 输出3
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3, 384)  # 输出384
        self.fc2 = nn.Linear(384, num_classes)  # 输出2

        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        # BERT输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_output.last_hidden_state  # [batch_size, seq_length, 768]
        
        # 转置以适应Conv1d输入
        x = hidden_states.permute(0, 2, 1)  # [batch_size, 768, seq_length]
        
        # 并行1D-CNN模块
        conv1_out = self.relu(self.maxpool1(self.conv1(x)))
        conv2_out = self.relu(self.maxpool2(self.conv2(x)))
        conv3_out = self.relu(self.maxpool3(self.conv3(x)))
        
        # 使用 adaptive_max_pool1d 统一张量长度
        target_length = 100  # 目标长度（可调整）
        conv1_out = F.adaptive_max_pool1d(conv1_out, target_length)  # [batch_size, 128, target_length]
        conv2_out = F.adaptive_max_pool1d(conv2_out, target_length)  # [batch_size, 128, target_length]
        conv3_out = F.adaptive_max_pool1d(conv3_out, target_length)  # [batch_size, 128, target_length]
        
        # 拼接
        concatenated = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)  # [batch_size, 128*3, target_length]
        
        # 拼接后的卷积和池化
        conv_final_out1 = self.relu(self.maxpool_final1(self.conv_final1(concatenated)))
        conv_final_out2 = self.relu(self.maxpool_final2(self.conv_final2(conv_final_out1)))
        
        # 全连接层
        flat = self.flatten(conv_final_out2)
        fc1_out = self.relu(self.fc1(flat))
        output = self.fc2(self.dropout(fc1_out))
        
        return output

# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

# 数据加载
df = pd.read_json('/usr/gao/gubincheng/article_rep/bert微调/data/gossip/gossip_news.json') 
df['label'] = df['label'].map({'fake': 0, 'real': 1})

# 数据分割
train_texts, temp_texts, train_labels, temp_labels = train_test_split(df['input'].tolist(), df['label'].tolist(), test_size=0.4, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

# Tokenizer
local_model_path = '/usr/gao/gubincheng/article_rep/ENDEF-SIGIR2022/ENDEF-SIGIR2022-main/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(local_model_path)

# 构建DataLoader
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

def train(model, dataloader, optimizer, criterion, device):
    """
    优化后的训练函数
    """
    model.train()  # 切换到训练模式
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # 遍历所有数据批次
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()  # 梯度清零

        # 取出数据，并转移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播，获取模型输出
        outputs = model(input_ids, attention_mask)
        # 模型输出 logits，无需 softmax，直接用于计算交叉熵损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录损失和准确性
        total_loss += loss.item() * input_ids.size(0)  # 累积每个样本的损失
        preds = torch.argmax(outputs, dim=1)  # 获取预测类别
        total_correct += (preds == labels).sum().item()  # 累积预测正确的样本数量
        total_samples += labels.size(0)  # 累积样本数量

    avg_loss = total_loss / total_samples  # 平均损失
    accuracy = total_correct / total_samples  # 准确率

    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """
    验证函数：计算平均损失和准确率
    """
    model.eval()  # 切换到评估模式
    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    # 返回分类报告
    report = classification_report(all_labels, all_preds, target_names=['fake', 'real'], zero_division=0)
    return avg_loss, accuracy, report


# 初始化模型
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = FakeBERT()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
# 训练
for epoch in range(3):
    print(f"Epoch {epoch+1}")
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_report = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_acc}")

# 测试
test_loss, test_acc, test_report = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
print(test_report)

# 保存模型
torch.save(model.state_dict(), 'fakebert_model.pth')