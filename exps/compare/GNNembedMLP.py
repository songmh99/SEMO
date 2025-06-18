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

# config_path = sys.argv[1]
# config_path = 'try.yml'
# with open(config_path, "r") as f:
#     config = yaml.safe_load(f)
# print(f"device: {config['device']}")
print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
# bert_tiny_path = config['pretrain_model_path']
# BERTtiny_Tokenizer = BertTokenizer.from_pretrained(bert_tiny_path)
# BERTtiny_Bert = BertModel.from_pretrained(bert_tiny_path).to(config['device'])
# text_embed_dim = config['text_embed_dim']
# gnn_node_embeddings_path = 'Repaired_res/gnn_node_embeddings.npz'

###################################################
class CellDataset1(Dataset):
    # def __init__(self, data_path_dirty, data_path_repair, device=config['device'], text_embed_dim=config['text_embed_dim']):
    def __init__(self, data_path_dirty, data_path_repair, device, text_embed_dim, gnn_node_embeddings, config):
        self.dirtydata = pd.read_csv(data_path_dirty).astype(str)#[:200]
        self.repair_res = pd.read_pickle(data_path_repair)#[:200]
        self.device = device
        self.config = config
        #
        bert_tiny_path = config['pretrain_model_path']
        BERTtiny_Tokenizer = BertTokenizer.from_pretrained(bert_tiny_path)
        BERTtiny_Bert = BertModel.from_pretrained(bert_tiny_path).to(config['device'])
        # gnn_node_embeddings_path = config['gnn_node_embeddings_path']
        # self.node_embeddings1 = np.load(gnn_node_embeddings_path)
        self.node_embeddings1 = gnn_node_embeddings
        
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
            samples = [col_name_embed, error_value_embed]
            inputs.append(samples)
            label_tensor = torch.tensor(label, dtype=torch.float)
            labels.append(label_tensor)
        return inputs, labels

    def encode_text(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.config['encode_text_max_length']).to(self.device)
        with torch.no_grad():  # 
            output = self.bert(**tokens)
        return output.last_hidden_state.mean(dim=1).squeeze().detach().to(self.device)  # 
    
    def get_embeddings(self, col, val):
        node_id = f"{col}_{val}"
        if node_id in self.node_embeddings1.keys():
            cell_embedding1 = self.node_embeddings1[node_id]
            return cell_embedding1  
        else:
            return self.encode_text(val)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        col_name_embed, error_value_embed = self.inputs[idx]
        return {
            'col_name': col_name_embed,
            'error_value_embed': error_value_embed, 
            'label': self.labels[idx]
        }

# class cellclsmodel1(nn.Module):
#     # def __init__(self, embed_dim=config['text_embed_dim'], hidden_dim=config['cellclsmodel_hidden_dim'], num_classes=config['num_classes']):
#     def __init__(self, embed_dim, hidden_dim, num_classes):
#         super(cellclsmodel1, self).__init__()
#         self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim,num_classes)

#     def forward(self, col_name_embed, error_value_embed):
#         x = torch.cat([col_name_embed, error_value_embed], dim=1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
class cellclsmodel1(nn.Module):
    # def __init__(self, embed_dim=config['text_embed_dim'], hidden_dim=config['cellclsmodel_hidden_dim'], num_classes=config['num_classes']):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(cellclsmodel1, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim*2,256)
        self.fc4 = nn.Linear(256,num_classes)

    def forward(self, col_name_embed, error_value_embed):
        
        col_name_embed = self.fc1(col_name_embed)
        error_value_embed = self.fc2(error_value_embed)
        x = torch.cat([col_name_embed, error_value_embed], dim=1)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
from datetime import datetime
# 
# def train(model, dataloader, criterion, optimizer, epochs=5, device=config['device'], res_file=config['GNNembed_res_file']):
def train1(model, dataloader, criterion, optimizer, epochs, device, res_file):
    model.train()
    f = open(res_file, 'a')
    # 
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    # 

    f.write(f"------ current time : {current_time} \n")
    f.write(f"====== training mlp ====== \n")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            col_name = batch['col_name'].to(device)
            error_value_embed = batch['error_value_embed'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(col_name, error_value_embed)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
            f.write(f"   Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}\n")
    print(f"THE LAST Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
    f.write(f"   THE LAST Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}\n")
    f.write(f"     training end ! \n")
    f.close()

def test1(model, dataloader, device, res_file):
    model.eval()  # 
    correct = 0
    total = 0
    f = open(res_file, 'a')
    f.write(f"=====  MLP TEST RESULT =====\n")
    with torch.no_grad():  # 
        for batch in dataloader:
            col_name = batch['col_name'].to(device)
            error_value_embed = batch['error_value_embed'].to(device)
            labels = batch['label'].to(device)  # shape: (batch_size, num_classes)

        
            outputs = model(col_name, error_value_embed)
            
            pred_idx = torch.argmax(outputs, dim=1)  
       
            for i in range(labels.shape[0]):
                gt = labels[i]  #
                if gt[pred_idx[i]] == 1:  # 
                    correct += 1  # 
                total += 1  # 
                # print(correct / total)
    
    accuracy = correct / total if total > 0 else 0  # 
    print(f"MLP cls Test Accuracy: {accuracy:.4f}")
    f.write(f"     MLP CLS Test Accuracy: {accuracy:.4f} \n")
    f.close()
    return accuracy
##################################################################################
def compute_test_metric(clean_df, dirty_df, res_df, test_indices):
    pred_num = 0  #
    right_num = 0  #
    for i in test_indices:
        for j in range(len(res_df.columns)):
            if res_df.iloc[i, j] != dirty_df.iloc[i, j]:  #
                pred_num += 1
                if pd.isna(clean_df.iloc[i, j]) and pd.isna(res_df.iloc[i, j]):  
                    right_num += 1  # 
                elif clean_df.iloc[i, j] == res_df.iloc[i, j]:  
                    right_num += 1
    
    precision = right_num / pred_num if pred_num > 0 else 0  # 

    wrong_num = 0  # 
    recall_num = 0  # 
    for i in test_indices:
        for j in range(len(clean_df.columns)):
            if clean_df.iloc[i, j] != dirty_df.iloc[i, j]:  # 
                wrong_num += 1
                if pd.isna(clean_df.iloc[i, j]) and pd.isna(res_df.iloc[i, j]):  
                    recall_num += 1  # 
                elif clean_df.iloc[i, j] == res_df.iloc[i, j]:  
                    recall_num += 1
    
    recall = recall_num / wrong_num if wrong_num > 0 else 0  # 
    f1 = 2*precision*recall / (precision + recall) if (precision + recall)>0 else 0
    return precision, recall, f1

def predict_with_budget(probs, budget):
   
    
    human_class_idx = num_classes - 1

    # 
    max_confidence_indices = np.argmax(probs, axis=1)

    # 
    human_indices = np.where(max_confidence_indices == human_class_idx)[0]

    if len(human_indices) <= budget:
        # 
        return max_confidence_indices


    human_confidences = probs[human_indices, human_class_idx]


    sorted_human_indices = human_indices[np.argsort(-human_confidences)]


    keep_human = sorted_human_indices[:budget]


    drop_human = sorted_human_indices[budget:]

    preds = max_confidence_indices.copy()
    for idx in drop_human:
      
        top2 = np.argsort(-probs[idx])[:2]
        
        preds[idx] = top2[1] if top2[0] == human_class_idx else top2[0]
    return preds

def compute_mlp_metric1(model, dataloader,config, res_file,budget):

    model.eval()
    all_outputs = []
    with torch.no_grad():  #
        for batch in dataloader:
            col_name = batch['col_name'].to(config['device'])
            error_value_embed = batch['error_value_embed'].to(config['device'])
            labels = batch['label'].to(config['device'])  # shape: (batch_size, num_classes)
    
            #
            outputs = model(col_name, error_value_embed)
            # pred_idx = torch.argmax(outputs, dim=1)  # 
            # pred_res.extend(pred_idx.tolist())
            all_outputs.append(outputs.cpu())
    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_probs = F.softmax(all_outputs_tensor, dim=1).numpy()
    pred_res = predict_with_budget(all_probs, budget)
    ###
    with open(config['mlb_pkl_path'], "rb") as f:
        mlb = pickle.load(f)
    label_map = mlb.classes_
    # label_map[label_map.tolist().index('human')]='GT'
    if 'human' in label_map:
        index = np.where(label_map == 'human')[0][0]
        label_map[index] = 'GT'
    else:
        print("Warning: 'human' not found in label_map. Skipping replacement.")
    ## 
    clean_path = config['clean_path']
    dirty_path = config['dirty_path']
    dirty_df = pd.read_csv(dirty_path).astype(str)
    clean_df = pd.read_csv(clean_path).astype(str)
    dirty_df = dirty_df.fillna("NuLL")
    clean_df = clean_df.fillna("NuLL")
    ##
    test_df_path = config['test_df_path']
    test_df = pd.read_pickle(test_df_path)
    res_df = copy.deepcopy(dirty_df)
    pred_useless = 0
    for i in range(len(test_df['GT'])):
        row = test_df.iloc[i,:]
        row_num, col_name = row['position']
        # print(label_map[pred_res[i]])
        
        if label_map[pred_res[i]] == 'GT':
            if row['label'] != ['human']:  #
                # print('PRED ERROR !!! ')
                # print(row['label'])
                pred_useless += 1
                # continue
                
        res = row[label_map[pred_res[i]]]
        res_df.loc[row_num, col_name] = res
    #
    with open(config['test_indices_path'], "rb") as f:
        test_indices = pickle.load(f)
    precision, recall, f1 = compute_test_metric(clean_df, dirty_df, res_df, test_indices)
    print(f"Result of MLP: precision: {precision}, recall: {recall}, f1_score: {f1}")
    f = open(res_file,'a')
    f.write(f"===== Result of MLP : Precision & Recall & F1 ====\n")
    f.write(f"     precision: {precision}\n     recall: {recall}\n     f1_score: {f1}\n")
    all_test_num = len(test_df['GT'])
    f.write(f"     pred human useless number: {pred_useless},   total test number: {all_test_num}\n")
    print(f"     pred human useless number: {pred_useless},   total test number: {all_test_num}\n")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    f.write(f"------ Current Time : {current_time} \n\n")
    f.close()  
    return precision, recall, f1, pred_useless, len(test_df['GT'])

def collate_fn_factory1(device, pad_value=0):
    def collate_fn(batch):
        batch_device = {}
        for key in batch[0].keys():  # 
            values = [sample[key] for sample in batch]  # 
      
            values = [torch.tensor(v) if isinstance(v, np.ndarray) else v for v in values]
           
            if isinstance(values[0], torch.Tensor):
              
                max_size = max(v.shape[0] for v in values)  
                padded_values = []
                for v in values:
                    pad_len = max_size - v.shape[0]
                    if pad_len > 0:
                        # 
                        pad_tensor = torch.full((pad_len, *v.shape[1:]), pad_value, dtype=v.dtype)
                        v = torch.cat([v, pad_tensor], dim=0)

                    padded_values.append(v.to(device))

                batch_device[key] = torch.stack(padded_values)
            else:
                batch_device[key] = values 
        return batch_device
    return collate_fn