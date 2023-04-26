import os
import re
from collections import defaultdict

# 이동하려는 log 폴더 경로를 지정합니다. 
folder_to_move = '/opt/ml/level1_bookratingprediction-recsys-10/code/log'


# 폴더 이동을 수행합니다.
os.chdir(folder_to_move)

# 새로운 현재 디렉토리를 출력합니다.
new_current_directory = os.getcwd()
dirlist = os.listdir()

new_dir_list = []
for dir in dirlist:
    new_dir = os.getcwd()+'/'+dir+'/'+'train.log'
    new_dir_list.append(new_dir)

valid_loss_results = []
for log in new_dir_list:
    with open(log, "r") as file:
        content = file.read()
    
    valid_loss_values = re.findall(r"valid loss : ([\d.]+)", content)
    valid_loss_values = [float(val) for val in valid_loss_values]
    try:
        min_valid_loss = min(valid_loss_values)
        valid_loss_results.append((log.split('/')[-2], min_valid_loss))
    except:
        pass

valid_loss_results = sorted(valid_loss_results, key=lambda x: x[1])

model_names = ["FM", "FFM", "CNN_FM", "DCN", "WDN", "NCF", "DeepCoNN", "CNN_FM2"]
top_results_per_model = defaultdict(list)

for result in valid_loss_results:
    match = re.search(r"\d+_(.+)", result[0])
    if match:
        model_name = '_'.join(match.group(1).split('_')[1:])
        if model_name in model_names:
            top_results_per_model[model_name].append(result)

for model_name, results in top_results_per_model.items():
    top_results_per_model[model_name] = sorted(results, key=lambda x: x[1])[:8]

print("Top 3 results per model:")
for model_name, results in top_results_per_model.items():
    print(f"Model: {model_name}")
    for result in results:
        print(f"  {result}")