import os
import numpy as np
import random
import json

def allocate_data_dirichlet(labels, num_classes, num_clients, alpha, train_ratio, val_ratio, test_ratio):
    """
    Allocate data to clients using Dirichlet distribution.
    
    :param labels: Array of data labels.
    :param num_clients: Number of clients to distribute data across.
    :param alpha: Concentration parameter for the Dirichlet distribution.
    :return: A list of indices for each client representing their data.
    """                                  
    # Generating proportions for each class across clients
    class_proportions = np.random.dirichlet([alpha]*num_clients, num_classes) #[[0.2 0.8],[0.1 0.9],[0.5 0.5],[0.6 0.4]]-4 classes 2 clients
    print(class_proportions)
    client_data_indices = [[[],[],[]] for _ in range(num_clients)]                     #[[] []]
    for class_label in range(num_classes):       
        sub_dict = labels[class_label]     
        sub_dict = list(sub_dict.items())
        random.shuffle(sub_dict)
        proportions = class_proportions[class_label]                           #[0.2 0.8]
        allocations = np.round(proportions * len(sub_dict)).astype(int)
        # Ensure that rounding doesn't cause more allocations than available samples
        allocations[-1] = len(sub_dict) - np.sum(allocations[:-1])        #[2 3]
        print("allocations", allocations)
        # Allocate data based on calculated proportions
        start = 0
        #print(sub_dict)
        for client_id, allocation in enumerate(allocations):                   #client_id=0|allocation=2            
            client_data_indices[client_id][0].extend(sub_dict[start:(start+round(allocation*train_ratio))])
            client_data_indices[client_id][1].extend(sub_dict[(start+round(allocation*train_ratio)):(start+round(allocation*(train_ratio+val_ratio)))])
            client_data_indices[client_id][2].extend(sub_dict[(start+round(allocation*(train_ratio+val_ratio))):(start+allocation)])
            start += allocation
            
    return client_data_indices

file_list = []
num_classes = 8
num_clients = 4
alpha = 1.5
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2
#generate a temporary file in format [{xxx1:0,},...,{xxxn:7,}]
for cls in range(num_classes):
    file_dict = {}
    files = [s.split(".")[0] for s in os.listdir(os.path.join("isic2019",str(cls+1))) if '.jpg' in s]
    for file in files:
        file_dict[file] = cls
    file_list.append(file_dict)

#print(file_list)
clients = allocate_data_dirichlet(file_list, num_classes, num_clients, alpha, train_ratio, val_ratio, test_ratio)
#print("clients", clients)
json_data = {}
for num in range(num_clients):
    json_data["client "+str(num+1)] = {}
    json_data["client "+str(num+1)]["train"] = []
    json_data["client "+str(num+1)]["val"] = []
    json_data["client "+str(num+1)]["test"] = []
    
    print("train_len", len(clients[num][0]))
    print("val_len", len(clients[num][1]))
    print("test_len", len(clients[num][2]))
    
    for (name, cls) in clients[num][0]: 
        json_data["client "+str(num+1)]["train"].append(str(name)+".jpg"+","+str(cls+1))
    for (name, cls) in clients[num][1]: 
        json_data["client "+str(num+1)]["val"].append(str(name)+".jpg"+","+str(cls+1))
    for (name, cls) in clients[num][2]: 
        json_data["client "+str(num+1)]["test"].append(str(name)+".jpg"+","+str(cls+1))        

with open("FL_divide.json", 'w') as file:
    json.dump(json_data, file, indent=4) 