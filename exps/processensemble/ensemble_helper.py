import random
from collections import Counter
import pandas as pd
import random
def compute_repair_metric_with_budget(clean_df, dirty_df, res_df, test_indices=None, budget=0.2, seed=42):
    pred_cells = []  # 保存预测修复的位置 (i, j)
    if test_indices is None or len(test_indices) == 0:
        test_indices = range(len(res_df))
    for i in test_indices:
        for j in range(len(res_df.columns)):
            if res_df.iloc[i, j] != dirty_df.iloc[i, j]:  # 模型预测已修复
                pred_cells.append((i, j))
    
    # 随机选20%的预测修复作为人工修复，并视为正确
    random.seed(seed)
    num_manual_fix = int(len(pred_cells) * budget)
    manual_fix_cells = set(random.sample(pred_cells, num_manual_fix))

    pred_num = len(pred_cells)
    right_num = 0

    for i, j in pred_cells:
        if (i, j) in manual_fix_cells:
            right_num += 1  # 人工修复视为正确
        elif pd.isna(clean_df.iloc[i, j]) and pd.isna(res_df.iloc[i, j]):
            right_num += 1
        elif clean_df.iloc[i, j] == res_df.iloc[i, j]:
            right_num += 1

    precision = right_num / pred_num if pred_num > 0 else 0

    # Recall: 看 clean 和 dirty 哪些位置是错的，然后看是否被修复正确
    wrong_num = 0
    recall_num = 0
    for i in test_indices:
        for j in range(len(clean_df.columns)):
            if clean_df.iloc[i, j] != dirty_df.iloc[i, j]:  # 真实存在错误
                wrong_num += 1
                if (i, j) in manual_fix_cells:
                    recall_num += 1  # 被人工修复
                elif pd.isna(clean_df.iloc[i, j]) and pd.isna(res_df.iloc[i, j]):
                    recall_num += 1
                elif clean_df.iloc[i, j] == res_df.iloc[i, j]:
                    recall_num += 1

    recall = recall_num / wrong_num if wrong_num > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1



def majority_vote(lst):
    if not lst:  # 如果列表为空，返回None
        return None
    counter = Counter(lst)
    max_freq = max(counter.values())  # 找到最大频率
    candidates = [key for key, value in counter.items() if value == max_freq]  # 选出所有最高频的元素
    return random.choice(candidates)  # 如果有多个，随机选择一个