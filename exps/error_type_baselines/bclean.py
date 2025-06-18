import time
import sys,os
sys.path.append(os.path.abspath("BClean"))
from BClean import BayesianClean
from analysis import analysis
from dataset import Dataset
from src.UC import UC
import copy

import yaml
config_path = sys.argv[1]
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    
# dirty_path = config['dirty_path']
dirty_path = sys.argv[2]
print(f"dirty path: {dirty_path} !!!")
bclean_res_path = sys.argv[3]
print(f"bclean_res_path : {bclean_res_path}")


clean_path = config['clean_path']
dataLoader = Dataset()
dirty_dataloader = dataLoader.get_data(path = dirty_path)
clean_dataloader = dataLoader.get_data(path = clean_path)
model_path = None
model_save_path = None
fix_edge = []

attr = UC(dirty_dataloader)
attr.build_from_json(config['uc_json_path'])

pat = attr.PatternDiscovery()
# print("pattern discovery:{}".format(pat))

attr = attr.get_uc()
print(dirty_path)
dirty_data = dataLoader.get_real_data(dirty_dataloader, attr_type = attr)
clean_data = dataLoader.get_real_data(clean_dataloader, attr_type = attr)

start_time = time.perf_counter()

model = BayesianClean(dirty_df = dirty_data, clean_df = clean_data, model_path = model_path,
                      model_save_path = model_save_path, attr_type = attr,
                      fix_edge = fix_edge,
                      model_choice = "bdeu",
                      infer_strategy = "PIPD",
                      tuple_prun = 1.0,
                      maxiter = 1,
                      num_worker = 32,
                      chunksize = 10)

dirty_data, repair_data, clean_data = (model.repair_list)[0], (model.repair_list)[
    1], (model.repair_list)[2]

actual_error, repair_error = model.actual_error, (model.repair_list)[3]

P, R, F = analysis(actual_error, repair_error, dirty_data, clean_data)

print("Repair Pre:{:.5f}, Recall:{:.5f}, F1-score:{:.5f}".format(P, R, F))
print("++++++++++++++++++++time using:{}+++++++++++++++++++++++".format(model.end_time - model.start_time))
print("date:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

repaired_dataloader = copy.deepcopy(dirty_dataloader)
common_columns = repair_data.columns.intersection(dirty_dataloader.columns)
repaired_dataloader[common_columns] = repair_data[common_columns]
# repaired_dataloader.shape

# bclean_res_path = config['bclean_res_path']
# bclean_res_path = sys.argv[3]
repaired_dataloader.to_csv(bclean_res_path, index=False)
