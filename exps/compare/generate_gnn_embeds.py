import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle


def load_data(file_path):
    return pd.read_csv(file_path).fillna('NuLL')

def parse_fd_rules(rule_path):
    rules = []
    with open(rule_path, 'r') as f:
        for line in f:
            left, right = line.strip().split('⇒')
            rules.append((left.strip(), right.strip()))
    return rules


def build_graph(df, rules):
    G = nx.DiGraph()
    node_texts = {} #
    for col in df.columns:
        unique_values = df[col].unique()
        for val in unique_values:
            node_id = f"{col}___{val}"
            G.add_node(node_id)
            node_texts[node_id] = str(val) 
    for left, right in rules:
        left_values = df[left].tolist()
        right_values = df[right].tolist()
        for lv, rv in zip(left_values, right_values):
            G.add_edge(f"{left}___{lv}", f"{right}___{rv}")
    return G, node_texts



def get_text_embedding(texts, model, tokenizer):
    if isinstance(texts, str):
        texts = [texts] # 
    # inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,max_length=512,)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1) # 


def generate_node_embeddings(node_texts, model, tokenizer):
    embeddings = get_text_embedding(list(node_texts.values()), model, tokenizer)
    return {node:emb for node, emb in zip(node_texts.keys(), embeddings)}


def graph_to_pyg(G, node_embeddings):
    node_list = list(G.nodes()) # 
    node_map = {node: i for i, node in enumerate(node_list)}
    x = torch.stack([node_embeddings[node] for node in node_list])
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u,v in G.edges()], dtype=torch.long).t().contiguous() # 
    # node_map = {node: i for i,node in enumerate(G.nodes())} # G.nodes() 
    # x = torch.stack([node_embeddings[node] for node in G.nodes()]) # 
    return Data(x=x, edge_index=edge_index), node_list

# gnn_node_labels, gnn_labeled_indices = build_partial_node_labels(nodewithlabel_df, num_nodes=data1.x.size(0), num_classes=num_classes)
def build_partial_node_labels(label_df, num_nodes, num_classes, node_list):
    node_labels = torch.zeros((num_nodes, num_classes))
    labeled_indices = []
    node_id_to_index_map = {}
    for _,row in label_df.iterrows():
        col = row['position'][1]
        val = row['dirty']
        node_id = f"{col}___{val}"
        if node_id in node_list:
            node_idx = node_list.index(node_id)
        else:
            continue
        label = torch.tensor(row['onehot_label'], dtype=torch.float)
        node_labels[node_idx] = label
        labeled_indices.append(node_idx)

    labeled_indices = torch.tensor(labeled_indices, dtype=torch.long)
    return node_labels, labeled_indices

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim))
        for _ in range(num_layers-2):
            self.convs.append(GATConv(hidden_dim, hidden_dim))
        self.convs.append(GATConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x

def structural_contrastive_loss(embeddings, edge_index, tau=2.0, num_neg_samples=5):
    import random
    from collections import defaultdict

    num_nodes = embeddings.size(0)
    device = embeddings.device

    i, j = edge_index

    # 
    adj_dict = defaultdict(set)
    for u, v in zip(i.tolist(), j.tolist()):
        adj_dict[u].add(v)

    all_nodes = set(range(num_nodes))
    anchor = embeddings[i]              # [num_pos, dim]
    pos_embed = embeddings[j]           # [num_pos, dim]


    pos_sim = F.cosine_similarity(anchor, pos_embed).unsqueeze(1)  # [num_pos, 1]

    neg_embeds = []
    for anchor_idx in i.tolist():
        # 
        neg_candidates = list(all_nodes - adj_dict[anchor_idx] - {anchor_idx})
        if len(neg_candidates) < num_neg_samples:
            # 
            # neg_candidates = neg_candidates + random.choices(neg_candidates, k=num_neg_samples - len(neg_candidates))
            neg_candidates = neg_candidates
        else:
            neg_candidates = random.sample(neg_candidates, num_neg_samples)
        neg_embeds.append(embeddings[torch.tensor(neg_candidates, device=device)])

    neg_embeds = torch.stack(neg_embeds)         # [num_pos, num_neg, dim]
    anchor_exp = anchor.unsqueeze(1)             # [num_pos, 1, dim]
    neg_sim = F.cosine_similarity(anchor_exp, neg_embeds, dim=2)  # [num_pos, num_neg]

    loss = -torch.log(torch.exp(pos_sim / tau).sum() / (torch.exp(neg_sim / tau).sum() + 1e-8))
    return loss

def label_contrastive_loss44(cls_preds, gnn_node_labels, gnn_labeled_indices, embeds_tmp_withlabel, device, tau=0.5, num_negatives=10):
    # softmax
    cls_preds = F.softmax(cls_preds, dim=1)

    # label_mask = gnn_node_labels[gnn_labeled_indices].bool().to(device)  # 
    # masked_preds = cls_preds.masked_fill(~label_mask, float('-inf'))
    # cls_representative_labels = masked_preds.argmax(dim=1)               # 

    #
    true_labels = gnn_node_labels[gnn_labeled_indices].to(device)  #
    label_mask = true_labels.bool().to(device)
    masked_preds = cls_preds.masked_fill(~label_mask, float('-inf'))
    cls_representative_labels = masked_preds.argmax(dim=1) 

    # 
    rep_labels_expand_1 = cls_representative_labels.unsqueeze(1)
    rep_labels_expand_2 = cls_representative_labels.unsqueeze(0)
    label_equal = rep_labels_expand_1 == rep_labels_expand_2
    label_equal.fill_diagonal_(False)
    label_disjoint = (true_labels @ true_labels.T) == 0
    # 
    label_freq = torch.bincount(cls_representative_labels, minlength=true_labels.shape[1])
    inv_freq = 1.0 / (label_freq.float() + 1e-6)

    loss_all = []
    skipped = 0
    total = len(gnn_labeled_indices)

    for i in range(total):
        anchor = embeds_tmp_withlabel[i]
        pos_mask = label_equal[i]
        neg_mask = label_disjoint[i]
        pos_idx = torch.where(pos_mask)[0].to(device)
        neg_idx = torch.where(neg_mask)[0].to(device)

        if len(pos_idx) == 0 or len(neg_idx) < num_negatives:
            skipped += 1
            continue  # 

        # 
        pos_sample_id = pos_idx[torch.randint(0, len(pos_idx), (1,))]

        # 
        neg_labels = cls_representative_labels[neg_idx]
        neg_weights = inv_freq[neg_labels]  # [neg_num]
        neg_weights = neg_weights / neg_weights.sum()  # 
        # print('neg_weights: ', neg_weights)
        neg_sample_ids = neg_idx[torch.multinomial(neg_weights, num_samples=num_negatives, replacement=True)]

        # 构造 similarity
        all_idx = torch.cat([pos_sample_id, neg_sample_ids], dim=0)
        all_embeds = embeds_tmp_withlabel[all_idx]  # shape: [1 + num_neg, D]
        sim = F.cosine_similarity(anchor.unsqueeze(0), all_embeds) / tau  # shape: [1 + num_neg]
        logits = sim.unsqueeze(0)  # shape: [1, 1 + num_neg]
        label = torch.zeros(1, dtype=torch.long, device=device)  # 

        loss = F.cross_entropy(logits, label)
        loss_all.append(loss)

    # print(f"[Contrastive] total: {total}, skipped: {skipped}, used: {len(loss_all)}")
    if len(loss_all) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(loss_all).mean()


