import os
import numpy as np
import random
import json

json_data = {}
num_classes = 8
num_clients = 3
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2
#generate a temporary file in format [{xxx1:0,},...,{xxxn:7,}]
for cls in range(num_classes):
    json_data[str(cls+1)] = {}
    json_data[str(cls+1)]["train"] = []
    json_data[str(cls+1)]["val"] = []
    json_data[str(cls+1)]["test"] = []
    sum_name = [s for s in os.listdir(os.path.join("isic2019",str(cls+1))) if '.jpg' in s]
    sum_len = len(sum_name)
    sum_list = list(range(0, sum_len))
    random.shuffle(sum_list)

    train_list, val_list, test_list = sum_list[:round(train_ratio*sum_len)], sum_list[round(train_ratio*sum_len):round((train_ratio+val_ratio)*sum_len)], sum_list[round((train_ratio+val_ratio)*sum_len):]
    for i1 in train_list:
        json_data[str(cls+1)]["train"].append(str(sum_name[i1])+","+str(cls+1))
    for i2 in val_list:
        json_data[str(cls+1)]["val"].append(str(sum_name[i2])+","+str(cls+1))        
    for i3 in test_list:
        json_data[str(cls+1)]["test"].append(str(sum_name[i3])+","+str(cls+1))   

with open("FL_divide_t2.json", 'w') as file:
    json.dump(json_data, file, indent=4) 
    
print(len(json_data["1"]["train"]))
print(len(json_data["1"]["val"]))
print(len(json_data["1"]["test"]))
print(len(json_data["2"]["train"]))
print(len(json_data["2"]["val"]))
print(len(json_data["2"]["test"]))
print(len(json_data["3"]["train"]))
print(len(json_data["3"]["val"]))
print(len(json_data["3"]["test"]))
print(len(json_data["4"]["train"]))
print(len(json_data["4"]["val"]))
print(len(json_data["4"]["test"]))
print(len(json_data["5"]["train"]))
print(len(json_data["5"]["val"]))
print(len(json_data["5"]["test"]))
print(len(json_data["6"]["train"]))
print(len(json_data["6"]["val"]))
print(len(json_data["6"]["test"]))
print(len(json_data["7"]["train"]))
print(len(json_data["7"]["val"]))
print(len(json_data["7"]["test"]))
print(len(json_data["8"]["train"]))
print(len(json_data["8"]["val"]))
print(len(json_data["8"]["test"]))