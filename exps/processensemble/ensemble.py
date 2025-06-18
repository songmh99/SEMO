import pandas as pd
import yaml
import sys
from datetime import datetime
import copy
import numpy as np
import pickle
import random

from ensemble_helper import compute_repair_metric_with_budget, majority_vote
import sys,os

config_path = sys.argv[1]
# config_path = 'yaml/beers-e10.yml'
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


############################ load data  #################################################

baran_res_path = config['baran_res_path']
bclean_res_path = config['bclean_res_path']
bigdancing_res_path = config['bigdancing_res_path']
holistic_res_path = config['holistic_res_path']
horizon_res_path = config['horizon_res_path']

baran_res = pd.read_csv(baran_res_path)
bigdancing_res = pd.read_csv(bigdancing_res_path)
bclean_res = pd.read_csv(bclean_res_path)
# daisy_res = pd.read_csv(daisy_res_path)
holistic_res = pd.read_csv(holistic_res_path)
horizon_res = pd.read_csv(horizon_res_path)

clean_path = config['clean_path']
dirty_path = config['dirty_path']
dirty_df = pd.read_csv(dirty_path).astype(str)
clean_df = pd.read_csv(clean_path).astype(str)
dirty_df = dirty_df.fillna("NuLL")
clean_df = clean_df.fillna("NuLL")

test_indices_path = config['test_indices_path']
test_indices = pd.read_pickle(test_indices_path)
train_indices_path = config['train_indices_path']
train_indices = pd.read_pickle(train_indices_path)

test_df_path = config['test_df_path']
df_test = pd.read_pickle(test_df_path)
train_df_path = config['train_df_path']
df_train = pd.read_pickle(train_df_path)

####################### single baseline result with budget  #########################################
f = open(config['res_ensemble'],'a')
current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
print("当前时间:", current_time)
f.write(f"\n\n")
f.write(f"------ Current Time : {current_time} \n")
# f.write(f"onehot_label  corresponds : {mlb.classes_}\n")
f.close()

repair_res_list = [baran_res, bigdancing_res, bclean_res, holistic_res, horizon_res]
repair_res_name = ['baran_res', 'bigdancing_res', 'bclean_res', 'holistic_res', 'horizon_res']
f = open(config['res_ensemble'],'a')
f.write(f"=====  Only one method result:  =====  \n")
for i in range(len(repair_res_name)):
    precision, recall, f1 = compute_repair_metric_with_budget(clean_df, dirty_df, repair_res_list[i], test_indices, budget=0.2)
    print(f"Result of {repair_res_name[i]}: precision: {precision}, recall: {recall}, f1_score: {f1}")
    f.write(f"Result of only {repair_res_name[i]} --- :\n")
    f.write(f"     precision: {precision},\n     recall: {recall},\n     f1_score: {f1}\n")
f.close()

#####################  ensemble with human budget  ##############################
method_list = config['method_list']
# 深拷贝脏数据，作为修复结果的基础
res_df_withhuman = copy.deepcopy(dirty_df)

# 设置预算：最多允许人工修复的单元格数
budget_ratio = 0.2
total_cells = len(df_test)
budget = int(total_cells * budget_ratio)
used_budget = 0  # 已使用的人工修复次数

for i in range(len(df_test)):
    row = df_test.iloc[i, :]
    row_num, col_name = row['position']
    rep_l = []
    
    for j in method_list:
        if row[j] != 'NuLL':
            rep_l.append(row[j])
    
    if len(rep_l) == 0:
        # 仅当还有预算时，才进行人工修复
        if used_budget < budget:
            res_df_withhuman.loc[row_num, col_name] = row['GT']
            used_budget += 1
        else:
            pass  # budget用完，不做任何修改
    else:
        result = majority_vote(rep_l)
        res_df_withhuman.loc[row_num, col_name] = result
print(f"used_budget:  {used_budget}")
precision, recall, f1 = compute_repair_metric_with_budget(clean_df, dirty_df, res_df_withhuman, test_indices, 
                        budget=0) # 修复的时候已经使用了 budget,所以计算metric的时候不再使用budget!!
print(f"Result of Ensemble with human: precision: {precision}, recall: {recall}, f1_score: {f1}")
f = open(config['res_ensemble'],'a')
f.write(f"=============  Ensemble Result Analysis  =============\n")
f.write(f"Ensemble with human budget result: \n")
f.write(f"     precision: {precision},\n     recall: {recall},\n     f1_score: {f1}\n")
# f.write(f"     ensemble cls acc with human, acc = {ensemble_cls_acc_with_human} \n\n\n")
f.close()

###################  optimal  ##################################
method_list = config['method_list']
# 深拷贝脏数据，作为修复结果的基础
res_df_optimal = copy.deepcopy(dirty_df)

# 设置预算：最多允许人工修复的单元格数
budget_ratio = 0.2
total_cells = len(df_test)
budget = int(total_cells * budget_ratio)
used_budget = 0  # 已使用的人工修复次数

for i in range(len(df_test)):
    rep_flag = False
    row = df_test.iloc[i, :]
    row_num, col_name = row['position']
    rep_l = []
    
    for j in method_list:
        if row[j] == row['GT']:
            res_df_optimal.loc[row_num, col_name] = row['GT']
            rep_flag = True
            break
    if rep_flag:
        continue
    else:
        # 仅当还有预算时，才进行人工修复
        if used_budget < budget:
            res_df_optimal.loc[row_num, col_name] = row['GT']
            used_budget += 1
        else:
            pass  # budget用完，不做任何修改

print(f"used_budget:  {used_budget}")

precision, recall, f1 = compute_repair_metric_with_budget(clean_df, dirty_df, res_df_optimal, test_indices, 
                        budget=0) # 修复的时候已经使用了 budget,所以计算metric的时候不再使用budget!!
print(f"Result of optimal with budget_ratio: {budget_ratio} : precision: {precision}, recall: {recall}, f1_score: {f1}")

f = open(config['res_ensemble'],'a')
f.write(f"=============  Optimal Result  =============\n")
f.write(f"OPTIMAL Result with budget_ratio: {budget_ratio}: \n")
f.write(f"     precision: {precision},\n     recall: {recall},\n     f1_score: {f1}\n")
# f.write(f"     ensemble cls acc with human, acc = {ensemble_cls_acc_with_human} \n\n\n")
f.close()

#######################################################################
print('Ensemble is success !! ')

