import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet10
from torchvision import models
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import pickle

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        ##############Resnet##############
        #self.model = resnet10()
        self.model = models.resnet34(pretrained=True)
        self.input_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.input_features, 8)
        ##############Resnet##############
        
        ##############densenet121###############  
        #self.model = models.densenet121(pretrained=True)
        ##print(self.model_t)
        #self.model.classifier = nn.Linear(self.model.classifier.in_features, 8)
        ##############densenet121###############          
            
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        return x       

def get_score(hits, counts, pflag=False):
    acc = (hits[0]+hits[1]+hits[2]+hits[3]+hits[4]+hits[5]+hits[6]+hits[7])/(counts[0]+counts[1]+counts[2]+counts[3]+counts[4]+counts[5]+counts[6]+counts[7])

    if pflag:
        print("*************Metrics******************")               
        print("Melanoma: {}, Melanocytic nevus: {}, Basal cell carcinoma: {}, Actinic keratosis: {}, Benign keratosis: {}, Dermatofibroma: {}, Vascular lesion: {}, Squamous cell carcinoma: {}".format(hits[0]/counts[0], hits[1]/counts[1], hits[2]/counts[2], hits[3]/counts[3], hits[4]/counts[4], hits[5]/counts[5], hits[6]/counts[6], hits[7]/counts[7]))
        #print("Accuracy: {}".format(acc))        
    return acc    
    
def train(net, trainloader, valloader, optimizer, epochs, current_round, device: str):       
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)    
    
    filename = "CLIENT4/output"
    best_score = 0
    for epoch in range(epochs):
        losses = []
        class_hits = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        class_counts = [0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7] 
        conf_label, conf_pred = [], []
        ##auc_pred = np.array([])
        ##auc_label = np.array([])
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images) 
            
            y_pred = (torch.softmax(outputs, dim=1)).detach().cpu().numpy()
            ##if auc_pred.size == 0:
            ##    auc_pred = y_pred
            ##else:
            ##    auc_pred = np.vstack((auc_pred, y_pred))
            y_label = labels.cpu().numpy()
            ##auc_label = np.append(auc_label, y_label)
           
            loss = criterion(outputs, labels)   
            losses.append(loss.item())
            _, preds = torch.max(outputs.data, 1)
            
            for idx in range(preds.shape[0]):
                class_counts[labels[idx].item()] += 1.0 
                conf_label.append(labels[idx].item())
                conf_pred.append(preds[idx].item())                  
                if preds[idx].item() == labels[idx].item():
                    class_hits[labels[idx].item()] += 1.0  

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()                   

        ####################################Caculate metrics#####################################
        ##train_auc = roc_auc_score(auc_label, auc_pred, multi_class="ovr", average="macro")       
        train_acc = get_score(class_hits, class_counts, True)        
        train_conf_matrix = confusion_matrix(conf_label, conf_pred)
        #print("Confusion Matrix", train_conf_matrix)           
        train_f1_micro = f1_score(conf_label, conf_pred, average='micro')
        train_f1_weighted = f1_score(conf_label, conf_pred, average='weighted')
        print('Epoch [%d/%d] - train loss %.4f - train accuracy %.4f - train f1_micro %.4f - train f1_weighted %.4f' % (epoch+1, epochs, np.mean(losses), train_acc, train_f1_micro, train_f1_weighted))   
        ####################################Caculate metrics#####################################    

        ####################Save best client model#############################
        val_loss, val_acc, val_f1 = val(net, valloader, device)
        if best_score < val_f1:
            best_score = val_f1
            print("Local best ACC achieved......", best_score)            
        #    state_dict = {name: param for name, param in net.state_dict().items() if not name.startswith("fc") and not name.startswith("avgpool")}
            state_dict = {name: param for name, param in net.state_dict().items()}
            with open(os.path.join(filename,"local_model_client4"+".pkl"), 'wb') as file:
                pickle.dump(state_dict, file)
                
    #return state_dict                
    ####################Save best client model#############################                   

        
def val(net, valloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    print("model 2")
    net.to(device)     
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
                  
    with torch.no_grad():
        losses = []
        class_hits = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #
        class_counts = [0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7]  #        
        correct = 0
        val_sum = 0  
        conf_label, conf_pred = [], []
        ##auc_pred = np.array([])
        ##auc_label = np.array([])        
        for data in valloader:
            images, labels = data[0].to(device), data[1].to(device)  #|torch.Size([1])
            outputs = net(images)             #torch.Size([1, 3])
            
            y_pred = (torch.softmax(outputs, dim=1)).detach().cpu().numpy()
            y_label = labels.cpu().numpy()          
            
            loss = criterion(outputs, labels).item()
            losses.append(loss)            
            _, preds = torch.max(outputs.data, 1)
            
            for idx in range(preds.shape[0]):
                class_counts[labels[idx].item()] += 1.0  
                conf_label.append(labels[idx].item())
                conf_pred.append(preds[idx].item())                  
                if preds[idx].item() == labels[idx].item():
                    class_hits[labels[idx].item()] += 1.0  
                    
            correct += (preds == labels).sum().item()
            val_sum+= labels.shape[0]

        ####################################Caculate metrics#####################################      
        val_acc = get_score(class_hits, class_counts, True)    
        val_conf_matrix = confusion_matrix(conf_label, conf_pred)
        #print("Confusion Matrix", test_conf_matrix)           
        val_f1_micro = f1_score(conf_label, conf_pred, average='micro')
        val_f1_weighted = f1_score(conf_label, conf_pred, average='weighted')
        print('val loss %.4f - val accuracy %.4f - val f1_micro %.4f - val f1_weighted %.4f' % (np.mean(losses), val_acc, val_f1_micro, val_f1_weighted))   
        ####################################Caculate metrics#####################################                 
            
    return np.mean(losses), val_acc, val_f1_weighted        
        
def test(net, testloader, current_round, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    print("model 2")
    net.to(device)    
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    
    with torch.no_grad():
        losses = []
        class_hits = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        class_counts = [0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7]         
        correct = 0
        test_sum = 0  
        conf_label, conf_pred = [], []        
        ##auc_pred = np.array([])
        ##auc_label = np.array([])        
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  #|torch.Size([1])
            outputs = net(images)             #torch.Size([1, 3])
            
            y_pred = (torch.softmax(outputs, dim=1)).detach().cpu().numpy()
            ##if auc_pred.size == 0:
            ##    auc_pred = y_pred
            ##else:
            ##    auc_pred = np.vstack((auc_pred, y_pred))
            y_label = labels.cpu().numpy()
            ##auc_label = np.append(auc_label, y_label)            
            
            loss = criterion(outputs, labels).item()
            losses.append(loss)            
            _, preds = torch.max(outputs.data, 1)
            
            for idx in range(preds.shape[0]):
                class_counts[labels[idx].item()] += 1.0 
                conf_label.append(labels[idx].item())
                conf_pred.append(preds[idx].item())                
                if preds[idx].item() == labels[idx].item():
                    class_hits[labels[idx].item()] += 1.0  
                    
            correct += (preds == labels).sum().item()
            test_sum+= labels.shape[0]

        ####################################Caculate metrics#####################################      
        test_acc = get_score(class_hits, class_counts, True)    
        test_conf_matrix = confusion_matrix(conf_label, conf_pred)
        #print("Confusion Matrix", test_conf_matrix)           
        test_f1_micro = f1_score(conf_label, conf_pred, average='micro')
        test_f1_weighted = f1_score(conf_label, conf_pred, average='weighted')
        print('Round %d - test loss %.4f - test accuracy %.4f - test f1_micro %.4f - test f1_weighted %.4f' % (current_round, np.mean(losses), test_acc, test_f1_micro, test_f1_weighted))   
        ####################################Caculate metrics#####################################                 
            
    return np.mean(losses), test_acc, test_f1_weighted