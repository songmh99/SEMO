import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
import yaml
import sys
from datetime import datetime
import copy
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")


###################################################
class CellDataset3(Dataset):
    def __init__(self, data_path_dirty, data_path_repair, device, text_embed_dim, config):
        self.dirtydata = pd.read_csv(data_path_dirty).astype(str)#[:200]
        self.repair_res = pd.read_pickle(data_path_repair)#[:200]
        self.device = device
        self.config = config
        #加载bert
        bert_tiny_path = config['pretrain_model_path']
        BERTtiny_Tokenizer = BertTokenizer.from_pretrained(bert_tiny_path)
        BERTtiny_Bert = BertModel.from_pretrained(bert_tiny_path).to(config['device'])
        text_embed_dim = config['text_embed_dim']
        self.tokenizer = BERTtiny_Tokenizer
        self.bert = BERTtiny_Bert
        self.inputs, self.labels = self.process_data()

    def process_data(self):
        inputs = []
        labels = []
        # for i in range(self.repair_res.shape[0]):
        for i in tqdm(range(self.repair_res.shape[0]), desc="Processing Data", unit="sample"):
            # if i%100 == 0:
            #     print(i)
            row = self.repair_res.iloc[i,:]
            # print(type(row['position'].iloc[0]))
            (row_idx, col_name) = row['position']
            # if row_idx >199:
            #     continue
            col_name_embed = self.encode_text(col_name)
            error_value = str(row['dirty'])
            error_value_embed = self.get_embeddings(col_name, error_value)  
            label = row['onehot_label']
            row_content = ' '.join([str(v) for v in self.dirtydata.iloc[row_idx].values])
            row_embed = self.encode_text(row_content)
            col_content = ' '.join([str(v) for v in self.dirtydata[col_name].values])
            col_embed = self.encode_text(col_content)
            # samples = [row_idx, col_name_embed, error_value_embed, row_embed, col_embed, repair_value_embed]
            samples = [col_name_embed, error_value_embed, row_embed, col_embed]
            inputs.append(samples)
            label_tensor = torch.tensor(label, dtype=torch.float)
            labels.append(label_tensor)
        return inputs, labels

    def encode_text(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.config['encode_text_max_length']).to(self.device)
        with torch.no_grad():  # 禁用梯度计算，防止计算图保留
            output = self.bert(**tokens)
        return output.last_hidden_state.mean(dim=1).squeeze().detach().to(self.device)  # `detach()` 彻底移除计算图
    
    def get_embeddings(self, col, val):
        node_id = f"{col}_{val}"
        gnn_node_embeddings_path = self.config['gnn_node_embeddings_path']
        node_embeddings1 = np.load(gnn_node_embeddings_path)
        if node_id in node_embeddings1.keys():
            cell_embedding1 = node_embeddings1[node_id]
            return cell_embedding1  
        else:
            return self.encode_text(val)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # row_idx, col_name_embed, error_value_embed, row_embed, col_embed, repair_value_embed = self.inputs[idx]
        col_name_embed, error_value_embed, row_embed, col_embed = self.inputs[idx]
        return {
            'col_name': col_name_embed,
            'error_value_embed': error_value_embed, 
            'row_embed': row_embed,
            'col_embed': col_embed,
            'label': self.labels[idx]
        }



class cellclsmodel3(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(cellclsmodel3, self).__init__()
        self.fc1 = nn.Linear(embed_dim * 4, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, col_name_embed, error_value_embed, row_embed, col_embed):
        x = torch.cat([col_name_embed, error_value_embed, row_embed, col_embed], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

from datetime import datetime
# 训练循环
def train3(model, dataloader, criterion, optimizer, epochs, device, res_file):
    model.train()
    f = open(res_file, 'a')
    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    # 打印当前时间
    print("当前时间:", current_time)
    f.write(f"------ current time : {current_time} \n")
    f.write(f"====== training mlp ====== \n")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            col_name_embed = batch['col_name'].to(device)
            error_value_embed = batch['error_value_embed'].to(device)
            row_embed = batch['row_embed'].to(device)
            col_embed = batch['col_embed'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            # outputs = model(row_idx, col_name, error_value_embed, row_embed, col_embed, repair_value_embed)
            outputs = model(col_name_embed, error_value_embed, row_embed, col_embed)
            loss = criterion(outputs, labels)
            loss.backward()
            # loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
            f.write(f"   Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}\n")
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
    f.write(f"   Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}\n")
    f.write(f"     training end ! \n")
    f.close()

def test3(model, dataloader, device, res_file):
    model.eval()  # 设为评估模式
    correct = 0
    total = 0
    f = open(res_file, 'a')
    f.write(f"=====  MLP TEST RESULT =====\n")
    with torch.no_grad():  # 关闭梯度计算
        for batch in dataloader:
            col_name_embed = batch['col_name'].to(device)
            error_value_embed = batch['error_value_embed'].to(device)
            row_embed = batch['row_embed'].to(device)
            col_embed = batch['col_embed'].to(device)
            labels = batch['label'].to(device)  # shape: (batch_size, num_classes)

            # 获取预测结果
            # outputs = model(row_idx, col_name, error_value_embed, row_embed, col_embed, repair_value_embed)  # (batch_size, num_classes)
            outputs = model(col_name_embed, error_value_embed, row_embed, col_embed)
            
            pred_idx = torch.argmax(outputs, dim=1)  # 选择概率最大的类别索引

            # 检查预测的类别是否在 ground-truth 中
            for i in range(labels.shape[0]):
                gt = labels[i]  # 真实标签 (num_classes,)
                if gt[pred_idx[i]] == 1:  # 预测的类别索引在 ground-truth 里
                    correct += 1  # 预测正确
                total += 1  # 计数总样本数
                # print(correct / total)
    
    accuracy = correct / total if total > 0 else 0  # 计算准确率
    print(f"MLP cls Test Accuracy: {accuracy:.4f}")
    f.write(f"     MLP CLS Test Accuracy: {accuracy:.4f} \n")
    f.close()
    return accuracy
##################################################################################
def compute_test_metric3(clean_df, dirty_df, res_df, test_indices):
    pred_num = 0  # 预测的修复数量
    right_num = 0  # 预测正确的修复数量
    for i in test_indices:
        for j in range(len(res_df.columns)):
            if res_df.iloc[i, j] != dirty_df.iloc[i, j]:  # 该值是否被修复
                pred_num += 1
                if pd.isna(clean_df.iloc[i, j]) and pd.isna(res_df.iloc[i, j]):  
                    right_num += 1  # 处理 NaN 情况
                elif clean_df.iloc[i, j] == res_df.iloc[i, j]:  
                    right_num += 1
    
    precision = right_num / pred_num if pred_num > 0 else 0  # 避免除零

    wrong_num = 0  # 实际错误数量
    recall_num = 0  # 被正确修复的错误数量
    for i in test_indices:
        for j in range(len(clean_df.columns)):
            if clean_df.iloc[i, j] != dirty_df.iloc[i, j]:  # 真实存在错误
                wrong_num += 1
                if pd.isna(clean_df.iloc[i, j]) and pd.isna(res_df.iloc[i, j]):  
                    recall_num += 1  # 处理 NaN 情况
                elif clean_df.iloc[i, j] == res_df.iloc[i, j]:  
                    recall_num += 1
    
    recall = recall_num / wrong_num if wrong_num > 0 else 0  # 避免除零
    f1 = 2*precision*recall / (precision + recall) if (precision + recall)>0 else 0
    return precision, recall, f1

def predict_with_budget(probs, budget):
    """
    参数:
    - probs: shape = (num_samples, N+1)，每行是一个样本对N+1个类别的置信度
    - budget: int，最多选择第N+1类（人工）的样本数
    返回:
    - preds: shape = (num_samples,)，每个样本的最终预测类别（0~N）
    """
    num_samples, num_classes = probs.shape
    assert num_classes >= 2, "类别数必须至少为2（含人工）"
    
    # 第N+1类的索引（人工）
    human_class_idx = num_classes - 1

    # 获取最大置信度对应的类别
    max_confidence_indices = np.argmax(probs, axis=1)

    # 找出模型原本打算分为人工的样本索引
    human_indices = np.where(max_confidence_indices == human_class_idx)[0]

    if len(human_indices) <= budget:
        # 如果未超出预算，照常预测
        return max_confidence_indices

    # 否则，需要限制人工的数量
    # 先获取这些打算人工的样本的人工置信度
    human_confidences = probs[human_indices, human_class_idx]

    # 对这些样本按人工置信度从高到低排序
    sorted_human_indices = human_indices[np.argsort(-human_confidences)]

    # 前budget个保留人工
    keep_human = sorted_human_indices[:budget]

    # 剩下的转为置信度第二高的类别
    drop_human = sorted_human_indices[budget:]

    # 初始化结果为最大置信度类别
    preds = max_confidence_indices.copy()
    for idx in drop_human:
        # 获取置信度从高到低的排序
        top2 = np.argsort(-probs[idx])[:2]
        # 如果第一是人工，第二才是我们要的
        preds[idx] = top2[1] if top2[0] == human_class_idx else top2[0]
    return preds

def compute_mlp_metric3(model, dataloader,config, res_file, budget):
    ## 得到 mlp 预测结果
    model.eval()
    all_outputs = []
    with torch.no_grad():  # 关闭梯度计算
        for batch in dataloader:
            col_name_embed = batch['col_name'].to(config['device'])
            error_value_embed = batch['error_value_embed'].to(config['device'])
            row_embed = batch['row_embed'].to(config['device'])
            col_embed = batch['col_embed'].to(config['device'])
            labels = batch['label'].to(config['device'])  # shape: (batch_size, num_classes)
            # 获取预测结果
            outputs = model(col_name_embed, error_value_embed, row_embed, col_embed)
            # pred_idx = torch.argmax(outputs, dim=1)  # 选择概率最大的类别索引
            # pred_res.extend(pred_idx.tolist())
            all_outputs.append(outputs.cpu())
    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_probs = F.softmax(all_outputs_tensor, dim=1).numpy()
    pred_res = predict_with_budget(all_probs, budget)
    ### 获得 label map
    with open(config['mlb_pkl_path'], "rb") as f:
        mlb = pickle.load(f)
    label_map = mlb.classes_
    label_map[label_map.tolist().index('human')]='GT'
    ## 
    clean_path = config['clean_path']
    dirty_path = config['dirty_path']
    dirty_df = pd.read_csv(dirty_path).astype(str)
    clean_df = pd.read_csv(clean_path).astype(str)
    dirty_df = dirty_df.fillna("NuLL")
    clean_df = clean_df.fillna("NuLL")
    ## 按照 mlp预测结果，修改 test_df 记录的对应的值，res_df 为 按照mlp修改后的结果。
    test_df = pd.read_pickle(config['test_df_path'])
    res_df = copy.deepcopy(dirty_df)
    pred_useless = 0
    for i in range(len(test_df['GT'])):
        row = test_df.iloc[i,:]
        row_num, col_name = row['position']
        # print(label_map[pred_res[i]])
        if label_map[pred_res[i]] == 'GT':
            if row['label'] != ['human']:  # 为了公平比较：如果预测为人工，实际不需要人工，则不进行修复。
                print('PRED ERROR !!! ')
                print(row['label'])
                pred_useless += 1
                continue
        res = row[label_map[pred_res[i]]]
        res_df.loc[row_num, col_name] = res
    # 计算 metric:
    with open(config['test_indices_path'], "rb") as f:
        test_indices = pickle.load(f)
    precision, recall, f1 = compute_test_metric3(clean_df, dirty_df, res_df, test_indices)
    print(f"Result of MLP: precision: {precision}, recall: {recall}, f1_score: {f1}")
    f = open(res_file,'a')
    f.write(f"===== Result of MLP : Precision & Recall & F1 ====\n")
    f.write(f"     precision: {precision}\n     recall: {recall}\n     f1_score: {f1}\n")
    all_test_num = len(test_df['GT'])
    f.write(f"     pred human useless number: {pred_useless},   total test number: {all_test_num}\n")
    print(f"     pred human useless number: {pred_useless},   total test number: {all_test_num}\n")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("当前时间:", current_time)
    f.write(f"------ Current Time : {current_time} \n\n\n")
    f.close()  
    return precision, recall, f1, pred_useless, len(test_df['GT'])


def collate_fn_factory3(device, pad_value=0):
    def collate_fn(batch):
        batch_device = {}
        for key in batch[0].keys():  # 遍历所有键
            values = [sample[key] for sample in batch]  # 取出 batch 里所有 `key` 对应的值
            # **如果是 numpy.ndarray，先转换为 torch.Tensor**
            values = [torch.tensor(v) if isinstance(v, np.ndarray) else v for v in values]
            # **如果是 Tensor，确保转换到 device**
            if isinstance(values[0], torch.Tensor):
                # 计算 batch 内的最大长度
                max_size = max(v.shape[0] for v in values)  
                padded_values = []
                for v in values:
                    pad_len = max_size - v.shape[0]
                    if pad_len > 0:
                        # 使用 pad_value 填充
                        pad_tensor = torch.full((pad_len, *v.shape[1:]), pad_value, dtype=v.dtype)
                        v = torch.cat([v, pad_tensor], dim=0)

                    padded_values.append(v.to(device))

                batch_device[key] = torch.stack(padded_values)
            else:
                batch_device[key] = values  # 非 Tensor 类型的数据保持不变
        return batch_device
    return collate_fn
