import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.manifold import TSNE

from dataset_client3 import Dataset                                  ##!
from model_client3 import Net                                        ##!
from Data_Augmentation import get_validation_augmentation

def prepare_dataset_client1(path, file_path, batch_size):
    with open(file_path) as file:
        json_data = json.load(file)         
        test_num_ids = json_data["client "+str(3)]["test"]            ##!   
        print(len(test_num_ids))
        test_num_dataset = Dataset(
            img_ids=test_num_ids,
            img_dir=os.path.join(path),
            transform=get_validation_augmentation(224,224))            
        test_dataloader = DataLoader(test_num_dataset, batch_size=batch_size, shuffle=False, num_workers=0)        
     
    return test_dataloader

def get_score(hits, counts, pflag=False):
    acc = (hits[0]+hits[1]+hits[2]+hits[3])/(counts[0]+counts[1]+counts[2]+counts[3])

    if pflag:
        print("*************Metrics******************")               
        print("Melanoma: {}, Basal cell carcinoma : {}, Squamous cell carcinoma: {}, Benign keratosis: {}".format(hits[0]/counts[0], hits[1]/counts[1], hits[2]/counts[2], hits[3]/counts[3]))
        #print("Accuracy: {}".format(acc))        
    return acc 

RESNET = Net(num_classes=4)
#activation = {}

#def get_activation(name):
#    def hook(model, input, output):
#        activation[name] = output.detach()
#    return hook
#RESNET.model.layer4.register_forward_hook(get_activation('layer4'))

file_path = "CLIENT3/output/model_client3.pkl"                       ##!
#file_path = "CLIENT3/output/local_model_client3_13.pkl"

with open(file_path, 'rb') as file:
    client1_state_dict = pickle.load(file)
    RESNET.load_state_dict(client1_state_dict, strict=True) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESNET.to(device)
    
testloader = prepare_dataset_client1("isic2019", "FL_divide_4clients.json", 32)

#test_loss, test_acc, test_f1 = test(model, testloader, 1, device)  
RESNET.eval()
features = []
labels = []

with torch.no_grad():    
    class_hits = [0.0, 0.0, 0.0, 0.0] #
    class_counts = [0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7]  #        
    correct = 0
    val_sum = 0  
    conf_label, conf_pred = [], []  
    for data in testloader:        
        image, label = data[0].to(device), data[1].to(device)
        outputs = RESNET(image)                        
        features.append(outputs.cpu())
        labels.append(label.cpu())  
        
        _, preds = torch.max(outputs.data, 1)
            
        for idx in range(preds.shape[0]):
            class_counts[label[idx].item()] += 1.0  
            conf_label.append(label[idx].item())
            conf_pred.append(preds[idx].item())                  
            if preds[idx].item() == label[idx].item():
                class_hits[label[idx].item()] += 1.0  
                    
        correct += (preds == label).sum().item()
        val_sum+= label.shape[0]

    ####################################Caculate metrics#####################################      
    val_acc = get_score(class_hits, class_counts, True)    
    val_conf_matrix = confusion_matrix(conf_label, conf_pred)
    #print("Confusion Matrix", test_conf_matrix)           
    val_f1_micro = f1_score(conf_label, conf_pred, average='micro')
    val_f1_weighted = f1_score(conf_label, conf_pred, average='weighted')
    print('test accuracy %.4f - test f1_micro %.4f - test f1_weighted %.4f' % (val_acc, val_f1_micro, val_f1_weighted))   
    ####################################Caculate metrics#####################################         
        
features = torch.cat(features).numpy()
labels = torch.cat(labels).numpy()

tsne = TSNE(n_components=2, random_state=22)
features_tsne = tsne.fit_transform(features)

plt.figure(figsize=(10, 7))
for i in range(4):                                     ##!
    indices = labels == i
    if i == 0:        
        plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], label="MEL")  
    elif i == 1:        
        plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], label="BCC")
    elif i == 2:        
        plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], label="SSC")        
    else:        
        plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], label="Others")        
#plt.title("t-SNE visualization of test data features")
#plt.xlabel("t-SNE component 1")
#plt.ylabel("t-SNE component 2")
plt.axis('off')
plt.legend(loc='upper left')
plt.savefig("CLIENT3/output/tsne3-g.png")              ##!
plt.show()

