# 合并各个修复方法，制作集成和后续选择器的数据集
# 划分训练集和测试集
import pandas as pd
import yaml
import sys
from datetime import datetime
import copy
import numpy as np
import pickle
import sys,os

config_path = sys.argv[1]
# config_path = 'try.yml'
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

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

#####################################################################################
## 处理数据：
print("Begin processing the baselines result ... ")
# dirty & clean
diff_mask_gt = dirty_df != clean_df  # 布尔矩阵，标记不同的元素
row_idx, col_idx = diff_mask_gt.to_numpy().nonzero()  # 获取不同元素的行索引和列索引
diff_positions_gt = [(row, dirty_df.columns[col]) for row, col in zip(row_idx, col_idx)]
print("dirty & clean 不同元素的位置:  ", len(diff_positions_gt))

diff_mask_bigdancing = dirty_df != bigdancing_res  # 布尔矩阵，标记不同的元素
row_idx_gt, col_idx_gt = diff_mask_bigdancing.to_numpy().nonzero()  # 获取不同元素的行索引和列索引
diff_positions_bigdancing = [(row, dirty_df.columns[col]) for row, col in zip(row_idx_gt, col_idx_gt)]
print("bigdancing 不同元素的位置:  ", len(diff_positions_bigdancing))

diff_mask_horizon = dirty_df != horizon_res  # 布尔矩阵，标记不同的元素
row_idx_gt, col_idx_gt = diff_mask_horizon.to_numpy().nonzero()  # 获取不同元素的行索引和列索引
diff_positions_horizon = [(row, dirty_df.columns[col]) for row, col in zip(row_idx_gt, col_idx_gt)]
print("horizon 不同元素的位置:  ", len(diff_positions_horizon))

diff_mask_baran = dirty_df != baran_res  # 布尔矩阵，标记不同的元素
row_idx_gt, col_idx_gt = diff_mask_baran.to_numpy().nonzero()  # 获取不同元素的行索引和列索引
diff_positions_baran = [(row, dirty_df.columns[col]) for row, col in zip(row_idx_gt, col_idx_gt)]
print("baran 不同元素的位置:  ", len(diff_positions_baran))

diff_mask_bclean = dirty_df != bclean_res  # 布尔矩阵，标记不同的元素
row_idx_gt, col_idx_gt = diff_mask_bclean.to_numpy().nonzero()  # 获取不同元素的行索引和列索引
diff_positions_bclean = [(row, dirty_df.columns[col]) for row, col in zip(row_idx_gt, col_idx_gt)]
print("bclean 不同元素的位置: ",len(diff_positions_bclean))

# diff_mask_daisy = dirty_df != daisy_res  # 布尔矩阵，标记不同的元素
# row_idx_gt, col_idx_gt = diff_mask_daisy.to_numpy().nonzero()  # 获取不同元素的行索引和列索引
# diff_positions_daisy = [(row, dirty_df.columns[col]) for row, col in zip(row_idx_gt, col_idx_gt)]
# print("daisy 不同元素的位置: ", len(diff_positions_daisy))

diff_mask_holistic = dirty_df != holistic_res  # 布尔矩阵，标记不同的元素
row_idx_gt, col_idx_gt = diff_mask_holistic.to_numpy().nonzero()  # 获取不同元素的行索引和列索引
diff_positions_holistic = [(row, dirty_df.columns[col]) for row, col in zip(row_idx_gt, col_idx_gt)]
print("holistic 不同元素的位置:  ", len(diff_positions_holistic))


# detect_list = list(set(diff_positions_bigdancing).union(diff_positions_bclean,diff_positions_daisy,diff_positions_holistic,diff_positions_horizon,diff_positions_baran))
detect_list = list(set(diff_positions_bigdancing).union(diff_positions_bclean,diff_positions_holistic,diff_positions_horizon,diff_positions_baran))
print('总共检测到的错误数量： ',len(detect_list))

# name_list = ['position','dirty','bclean','bigdancing','daisy','holistic','horizon','baran']
# name_list = ['position','dirty','bclean','bigdancing','holistic','horizon','baran']
name_list = config['name_list']
df = pd.DataFrame({name_list[0]:detect_list}, dtype=object)
# print("记录错误的dataframe(前5行): ")
# print(df[:5])

df[name_list[1]] = df[name_list[0]].map({(i,j):dirty_df[j][int(i)] for i,j in df['position']})
df[name_list[2]] = df[name_list[0]].map({(i,j):bclean_res[j][int(i)] for i,j in diff_positions_bclean})
df[name_list[3]] = df[name_list[0]].map({(i,j):bigdancing_res[j][int(i)] for i,j in diff_positions_bigdancing})
df[name_list[4]] = df[name_list[0]].map({(i,j):holistic_res[j][int(i)] for i,j in diff_positions_holistic})
df[name_list[5]] = df[name_list[0]].map({(i,j):horizon_res[j][int(i)] for i,j in diff_positions_horizon})
df[name_list[6]] = df[name_list[0]].map({(i,j):horizon_res[j][int(i)] for i,j in diff_positions_baran})
# df[name_list[4]] = df[name_list[0]].map({(i,j):daisy_res[j][int(i)] for i,j in diff_positions_daisy})
df['GT'] = df[name_list[0]].map({(i,j):clean_df[j][int(i)] for i,j in diff_positions_gt})  # 不是detect_list！ 因为假设：fix 正确的。！
# print("填充错误、修复值、GT（前15行）： ")
# df[:15]

df1 = df.sort_values(by='position', key=lambda x:x.apply(lambda y:y[0]))
# 填充空值
df2 = df1.fillna('NuLL')
# GT 为 NaN的行删掉。
df_ = df2.loc[(df2['GT']!= 'NuLL') & (df2['GT']!= 'empty')].reset_index(drop=True)
# print("按行号大小排序，并把GT为空的行删去 （假设fix正确的）：")
# df_[:5]


method_list = config['method_list']
# 方法选择的标签
label_list = []
for index, row in df_.iterrows():
    label = []
    if row['dirty'] == row['GT']:
        print(f"Index: {index} -- ERROR equals GT!!")
        continue
    # for i in ['bclean','bigdancing','daisy','holistic','horizon','baran']:
    # for i in ['bclean','bigdancing', 'holistic','horizon','baran']:
    for i in method_list:
        if row[i] == row['GT']:
            label.append(i)
    if len(label)==0:
        label = ['human']
    # if len(label)==1:
    #     print(label)
    label_list.append(label)
# print("label 样例：")
# print("  ",label_list[:10])
df_['label'] = label_list
# print("---- label 统计结果： ----")
# print(df_['label'].value_counts())
# df_[:5]


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
df_['onehot_label'] = mlb.fit_transform(df_['label']).tolist()
# # df_df_['onehot_label']
# print("编码为0，1标签： ")
# df_[:5]
print(config['res_ensemble'])
with open(config['mlb_pkl_path'], "wb") as f:
    pickle.dump(mlb, f)

f = open(config['res_ensemble'],'a')
current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
print("当前时间:", current_time)
f.write(f"------ current time : {current_time} \n")
f.write(f"onehot_label  corresponds : {mlb.classes_}\n")
f.close()
# print("标签对应内容:", mlb.classes_)

#############################################################################################
# 随机抽取 30% 的行索引作为测试集
test_indices = np.random.choice(clean_df.index, size=int(len(clean_df) * 0.6), replace=False)
train_indices = clean_df.index.difference(test_indices)
test_indices, train_indices

def parse_position(pos):
    """解析 (行, 列) 位置"""
    row, col = pos
    return row, col

df_['row_index'] = df_['position'].apply(lambda x: parse_position(x)[0])
df_['col_name'] = df_['position'].apply(lambda x: parse_position(x)[1])
# 根据行索引划分 df_
df_train = df_[df_['row_index'].isin(train_indices)].reset_index(drop=True)
df_test = df_[df_['row_index'].isin(test_indices)].reset_index(drop=True)

# 删除辅助列
df_train = df_train.drop(columns=['row_index', 'col_name'])
df_test = df_test.drop(columns=['row_index', 'col_name'])
# len(df_train), len(df_test) 

###########################################################################################
print('Saving data (process baselines results & Label data ) ... ')


