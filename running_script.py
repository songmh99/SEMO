import os
import sys
import yaml
import subprocess

# load yaml
if len(sys.argv) != 2:
    print("please: python run_single_dataset.py <config.yaml>")
    sys.exit(1)

config_path = sys.argv[1]
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

dataset = config["task_name"]
result_dir = config["result_dir"]
dataset_name = os.path.splitext(os.path.basename(dataset))[0]
dataset_result_dir = os.path.join(result_dir, dataset_name)
os.makedirs(dataset_result_dir, exist_ok=True)
print(f"dataset_name: {dataset_name}")
print(f"dataset_result_dir: {dataset_result_dir}")
algorithm_dir = 'exps/baselines/'
algorithm_scripts = ['bclean.py','bigdancing.py','holistic.py','horizon.py','baran.py'] 


# 1. sigle data cleaning method 
for script in algorithm_scripts:
    output_file = os.path.join(dataset_result_dir, f"{script.split('.')[0]}_results.csv")
    params = config_path
    print(f"process: {script} dataset: {dataset} ...")
    baseline_script = os.path.join(algorithm_dir, script)
    subprocess.run(["python", baseline_script, params, output_file], check=True)
    print(f"{script} is ok!")


##############  pre-process  ###########
dataprocess_script = "exps/processensemble/process.py"
subprocess.run(["python", dataprocess_script, config_path], check=True)
##############  ensemble-metric - optimal  #############
ensemble_script = "exps/processensemble/ensemble.py"
subprocess.run(["python", ensemble_script, config_path], check=True)

### 3. method selection
gnn_based_selector = 'exps/gnn/gnn_based_selector.py'
subprocess.run(["bash", "-c", f"source activate torch && python {gnn_based_selector} {config_path}"], check=True)
print(f"\n==== Done :) ====")
