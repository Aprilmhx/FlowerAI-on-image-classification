import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet10
from torchvision import models
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import pickle

class NetS(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(NetS, self).__init__()
        self.model_s = resnet10()

        self.input_features_s = self.model_s.fc.in_features
        self.model_s.fc = nn.Linear(self.input_features_s, 8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model_s(x)

        return x  
    
class NetT(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(NetT, self).__init__()
        
        ##############ResNet###############          
        #self.model_t = models.resnet34(pretrained=True)
        #self.input_features_t = self.model_t.fc.in_features
        #self.model_t.fc = nn.Linear(self.input_features_t, 8)
        ##############ResNet###############        

        ##############densenet121###############  
        self.model_t = models.densenet121(pretrained=True)
        ##print(self.model_t)
        self.model_t.classifier = nn.Linear(self.model_t.classifier.in_features, 8)
        ##############densenet121###############          
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model_t(x)

        return x 

def get_score(hits, counts, pflag=False):
    acc = (hits[0]+hits[1]+hits[2]+hits[3]+hits[4]+hits[5]+hits[6]+hits[7])/(counts[0]+counts[1]+counts[2]+counts[3]+counts[4]+counts[5]+counts[6]+counts[7])

    if pflag:
        print("*************Metrics******************")               
        print("Melanoma: {}, Melanocytic nevus: {}, Basal cell carcinoma: {}, Actinic keratosis: {}, Benign keratosis: {}, Dermatofibroma: {}, Vascular lesion: {}, Squamous cell carcinoma: {}".format(hits[0]/counts[0], hits[1]/counts[1], hits[2]/counts[2], hits[3]/counts[3], hits[4]/counts[4], hits[5]/counts[5], hits[6]/counts[6], hits[7]/counts[7]))
        #print("Accuracy: {}".format(acc))        
    return acc    

def online_distillation_loss(s_outputs, t_outputs, labels, T, criterion):
    soft_student_outputs = F.log_softmax(s_outputs / T, dim=1)
    soft_teacher_outputs = F.softmax(t_outputs / T, dim=1)
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(soft_student_outputs, soft_teacher_outputs) * (T * T)
    #tensor(0.4442, device='cuda:0', grad_fn=<MulBackward0>)    
    #tensor(1.2133, device='cuda:0', grad_fn=<NllLossBackward0>)
    return distillation_loss
    #tensor(2.0792, device='cuda:0', grad_fn=<AddBackward0>)

def train(net_s, net_t, trainloader, valloader, optimizer, epochs, current_round, device: str):       
    criterion = torch.nn.CrossEntropyLoss()
    net_s.train()
    net_s.to(device)     
    net_t.train()
    net_t.to(device)  
    torch.autograd.set_detect_anomaly(True)
    T = 2
    filename = "CLIENT4/output"    
    for epoch in range(epochs):
        losses_s = []
        class_hits_s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        class_counts_s = [0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7] 
        conf_label_s, conf_pred_s = [], []
        
        KDs = []
        losses_t = []
        class_hits_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
        class_counts_t = [0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7] 
        conf_label_t, conf_pred_t = [], []        

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            y_label = labels.cpu().numpy()

            outputs_s = net_s(images)
            outputs_t = net_t(images)            
                        
            loss_t = criterion(outputs_t, labels)
            KD_s = 0.5*(online_distillation_loss(outputs_s, outputs_t, labels, T, criterion))
            loss_s = criterion(outputs_s, labels)            
            loss = KD_s+loss_t+loss_s
            
            optimizer.zero_grad()             
            loss.backward()
            optimizer.step()                       

            #y_pred = (torch.softmax(outputs, dim=1)).detach().cpu().numpy()            
            losses_s.append(loss_s.item())
            losses_t.append(loss_t.item())
            KDs.append(KD_s.item())
            _, preds_s = torch.max(outputs_s.data, 1)
            _, preds_t = torch.max(outputs_t.data, 1)

            for idx in range(preds_s.shape[0]):
                class_counts_s[labels[idx].item()] += 1.0  
                conf_label_s.append(labels[idx].item())
                conf_pred_s.append(preds_s[idx].item())                  
                if preds_s[idx].item() == labels[idx].item():
                    class_hits_s[labels[idx].item()] += 1.0  
            for idx in range(preds_t.shape[0]):
                class_counts_t[labels[idx].item()] += 1.0  
                conf_label_t.append(labels[idx].item())
                conf_pred_t.append(preds_t[idx].item())                  
                if preds_t[idx].item() == labels[idx].item():
                    class_hits_t[labels[idx].item()] += 1.0                   

        ####################################Caculate metrics#####################################    
        train_acc_s = get_score(class_hits_s, class_counts_s, True)        
        train_conf_matrix_s = confusion_matrix(conf_label_s, conf_pred_s)
        #print("Confusion Matrix", train_conf_matrix)           
        train_f1_micro_s = f1_score(conf_label_s, conf_pred_s, average='micro')
        train_f1_weighted_s = f1_score(conf_label_s, conf_pred_s, average='weighted')
        print('Student model Epoch [%d/%d] - train loss %.4f - train accuracy %.4f - train f1_micro %.4f - train f1_weighted %.4f' % (epoch+1, epochs, np.mean(losses_s), train_acc_s, train_f1_micro_s, train_f1_weighted_s))   
        
        train_acc_t = get_score(class_hits_t, class_counts_t, True)        
        train_conf_matrix_t = confusion_matrix(conf_label_t, conf_pred_t)
        #print("Confusion Matrix", train_conf_matrix)           
        train_f1_micro_t = f1_score(conf_label_t, conf_pred_t, average='micro')
        train_f1_weighted_t = f1_score(conf_label_t, conf_pred_t, average='weighted')
        print('Teacher model Epoch [%d/%d] - train loss %.4f - KD loss %.4f - train accuracy %.4f - train f1_micro %.4f - train f1_weighted %.4f' % (epoch+1, epochs, np.mean(losses_t), np.mean(KDs), train_acc_t, train_f1_micro_t, train_f1_weighted_t))          
        ####################################Caculate metrics#####################################       

        ####################Save best client model#############################
        #val_loss, val_acc, val_f1 = val(net, valloader, device)
        #if best_score < val_f1:
        #    best_score = val_f1
        #    print("Local best ACC achieved......", best_score)            
        #    state_dict = {name: param for name, param in net.state_dict().items() if not name.startswith("fc") and not name.startswith("avgpool")}
        state_dict = {name: param for name, param in net_t.state_dict().items()}
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