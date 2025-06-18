import yaml
import sys

# config_path = 'yaml/beers-e10.yml'
config_path = sys.argv[1]
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


from exps.compare.generate_gnn_embeds import load_data, parse_fd_rules, build_graph
from exps.compare.generate_gnn_embeds import get_text_embedding, generate_node_embeddings, graph_to_pyg
from exps.compare.generate_gnn_embeds import build_partial_node_labels, GNN, structural_contrastive_loss, label_contrastive_loss44

from exps.compare.GNNembedMLP import CellDataset1, collate_fn_factory1,cellclsmodel1, compute_mlp_metric1
from exps.compare.GNNembedMLP import train1, test1


from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import pickle
import sys 
import numpy as np

######################### build graph ########################
device = config['device']

model_name1 = config['pretrain_model_path']
pretrained_tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
pretrained_model1 = AutoModel.from_pretrained(model_name1)

file_path = config['dirty_path']
rule_path = config['fd_rule_path']
df1 = load_data(file_path)
rules1 = parse_fd_rules(rule_path)
G1, node_texts1 = build_graph(df1, rules1)

node_init_embeddings1 = generate_node_embeddings(node_texts1, pretrained_model1, pretrained_tokenizer1)
data1,node_list1 = graph_to_pyg(G1, node_init_embeddings1)
data1.to(device)
embedding_dim1 = next(iter(node_init_embeddings1.values())).shape[0]  

nodewithlabel_df1 = pd.read_pickle(config['train_df_path'])
num_classes1 = config['num_classes']
gnn_node_labels1, gnn_labeled_indices1 = build_partial_node_labels(nodewithlabel_df1, 
             num_nodes=data1.x.size(0), num_classes=num_classes1, node_list=node_list1)

nodes_embeds_tmp_withlabel1 = [node_list1[i] for i in gnn_labeled_indices1]
nodes_col_list1 = [ i.split('___')[0] for i in nodes_embeds_tmp_withlabel1]
nodes_val_list1 = [ i.split('___')[-1] for i in nodes_embeds_tmp_withlabel1]
col_embeds1 = get_text_embedding(nodes_col_list1, pretrained_model1, pretrained_tokenizer1).to(device)

###########################  GNN Taining ################################
gnn_model1 = GNN(input_dim=embedding_dim1, hidden_dim=config['gnn_hidden_dim'], output_dim=config['text_embed_dim'], num_layers=3)
cls_model1 = cellclsmodel1(embed_dim=config['text_embed_dim'], hidden_dim=config['cellclsmodel_hidden_dim'], num_classes=config['num_classes']).to(device)
criterion = nn.BCEWithLogitsLoss() #支持多标签分类的损失函数
cls_model1.train().to(device)
gnn_model1.train().to(device)

iterating_gnn_lr = config['iterating_gnn_lr']
iterating_cls_lr = config['iterating_cls_lr']
optimizer_gnn = torch.optim.Adam(gnn_model1.parameters(), lr=iterating_gnn_lr)
optimizer_cls = torch.optim.Adam(cls_model1.parameters(), lr=iterating_cls_lr)
iterating_cls_model_path = config['iterating_cls_model_path']
iterating_gnn_model_path = config['iterating_gnn_model_path']
iterating_train_log_file_path = config['iterating_train_log_file_path']
iterating_res_file_path = config['iterating_res_file_path']

small_epoch = config['small_epoch']
iterate_num = config['iterate_num']

test_df_path = config['test_df_path']
train_df_path = config['train_df_path']
dirty_path = config['dirty_path']

labelloss_weight = config['labelloss_weight']
labelloss_tau = config['labelloss_tau']
labelloss_neg_num = config['labelloss_neg_num']
structloss_weight = config['structloss_weight']
structloss_tau = config['structloss_tau']
structloss_neg_num = config['structloss_neg_num']

switch_freq = config['switch_freq']
train_cls_begin = config['train_cls_begin']
budget_ratio = config['budget_ratio']
budget_num = int(budget_ratio*len(pd.read_pickle(test_df_path)))

beers_cls_f1 = 0.0
f = open(iterating_train_log_file_path,'a')
f.write(f"\n\n============================================================\n")
f.write(f"=================== ITERATING TRAINING  ==================\n")
f.close()
for i in range(iterate_num):
    train_cls = (i % switch_freq ==train_cls_begin)
    
    for param in gnn_model1.parameters():
        param.requires_grad = not train_cls
    for param in cls_model1.parameters():
        param.requires_grad = train_cls

    if train_cls:
        gnn_model1.eval()  # 切换到评估模式，关闭dropout等
        with torch.no_grad():  # 不计算梯度
            updated_embeddings = gnn_model1(data1.x.to(device), data1.edge_index.to(device))  # 获取更新后的节点嵌入
        # 2. 将更新后的嵌入存储到 node_embeddings1 中
        updated_node_embeddings = {node: updated_embeddings[node_list1.index(node)].cpu().numpy() for node in node_texts1.keys()}
        
        test_dataset1 = CellDataset1(dirty_path, test_df_path,device=device, text_embed_dim=config['text_embed_dim'],gnn_node_embeddings=updated_node_embeddings,config=config)
        test_dataloader1 = DataLoader(test_dataset1, batch_size=config['GNNembed_dataloader_batchsize'], shuffle=False, collate_fn=collate_fn_factory1(device))
        train_dataset1 = CellDataset1(dirty_path, train_df_path,device=device, text_embed_dim=config['text_embed_dim'],gnn_node_embeddings=updated_node_embeddings,config=config)
        train_dataloader1 = DataLoader(train_dataset1,batch_size=64,shuffle=True,collate_fn=collate_fn_factory1(device, pad_value=0))

        train1(cls_model1, train_dataloader1, criterion, optimizer_cls, epochs=small_epoch, 
                           device=config['device'], res_file=iterating_train_log_file_path)
        
        accuracy = test1(cls_model1, test_dataloader1, device=device,res_file=iterating_train_log_file_path)
        precision, recall, f1, pred_useless, all_test_num = compute_mlp_metric1(cls_model1, test_dataloader1,
                                                            config,res_file=iterating_train_log_file_path,budget=budget_num)
        print(precision, recall, f1, pred_useless, all_test_num)
        if f1 > beers_cls_f1:
            beers_cls_f1 = f1
            f = open(iterating_train_log_file_path,'a')
            f.write(f"=========== epoch {i}, save cls model : {f1} ==========================\n")
            f.close()
            print(f"=========== epoch {i}, save cls model : {f1} ===========================")
            torch.save(cls_model1, iterating_cls_model_path) 
            torch.save(gnn_model1, iterating_gnn_model_path)    

    else:
        f = open(iterating_train_log_file_path,'a')
        for j in range(small_epoch):
            gnn_output = gnn_model1(data1.x, data1.edge_index)
            gnn_embeds_withlabel = gnn_output[gnn_labeled_indices1]  # 不 detach
            cls_preds = cls_model1(col_embeds1, gnn_embeds_withlabel).detach()  # detach cls输出，只用来算 loss
            structural_loss = structural_contrastive_loss(gnn_output, data1.edge_index, tau=structloss_tau, num_neg_samples=structloss_neg_num)
            label_loss44 = label_contrastive_loss44(
                cls_preds, gnn_node_labels1, gnn_labeled_indices1,gnn_embeds_withlabel,device=config['device'], tau=labelloss_tau, num_negatives=labelloss_neg_num
            )
            label_loss =  labelloss_weight*label_loss44 + structloss_weight*structural_loss
            optimizer_gnn.zero_grad()
            label_loss.backward()
            optimizer_gnn.step()
            print(f"small epoch : {j}, structal loss : {structural_loss}, label_loss44: {label_loss44}")
            f.write(f"      small epoch: {j}, structal loss : {structural_loss}, label_loss44: {label_loss44}\n")
        f.close()


################################ Test Selector ##############################
### GNN 训练过程是和 selector （cls_model）一起的，所以GNN embedding 训好了 说明 selector也训好了。
cls_model1 = torch.load(iterating_cls_model_path)
gnn_model1 = torch.load(iterating_gnn_model_path)
with torch.no_grad():  # 不计算梯度
    updated_embeddings = gnn_model1(data1.x.to(device), data1.edge_index.to(device))  # 获取更新后的节点嵌入
# 2. 将更新后的嵌入存储到 node_embeddings1 中
updated_node_embeddings = {node: updated_embeddings[node_list1.index(node)].cpu().numpy() for node in node_texts1.keys()}

test_dataset1 = CellDataset1(dirty_path, test_df_path,device=device, text_embed_dim=config['text_embed_dim'],gnn_node_embeddings=updated_node_embeddings,config=config)
test_dataloader1 = DataLoader(test_dataset1, batch_size=config['GNNembed_dataloader_batchsize'], shuffle=False, collate_fn=collate_fn_factory1(device))
f = open(iterating_res_file_path,'a')
f.write(f"\n============================================================\n")
f.write(f"================ ITERATING  Traing --- RESULT==============\n")
f.close()
accuracy = test1(cls_model1, test_dataloader1, device=device,res_file=iterating_res_file_path)
precision, recall, f1, pred_useless, all_test_num = compute_mlp_metric1(cls_model1, test_dataloader1,config,res_file=iterating_res_file_path,budget=budget_num)

######################### save gnn node embeddings ###################
# save: node_texts1
# 将 node_embeddings1 变成 numpy 数组
iter_gnn_node_texts_path = config['iter_gnn_node_texts_path']
np.savez(iter_gnn_node_texts_path, **node_texts1)  # 直接存多个 numpy 数组


