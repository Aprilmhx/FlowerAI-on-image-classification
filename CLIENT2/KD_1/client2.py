from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
import os
import json
from torch.utils.data import random_split, DataLoader
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import pickle
import random
import numpy as np

from model_client2 import NetS, NetT, train, val, test
from dataset_client2 import Dataset
from Data_Augmentation import get_training_augmentation, get_validation_augmentation

def setup_seed(seed):
    torch.manual_seed(seed)          #cpu
    torch.cuda.manual_seed_all(seed) #gpus
    random.seed(seed)    
    np.random.seed(seed)

random_seed = 20
setup_seed(random_seed)   #7,15,20

class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, trainloader, valloader, testloader, num_classes) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.model_g = NetS(num_classes)
        self.model_l = NetT(num_classes)
        self.file_path = "CLIENT2/output/model_client2.pkl"
        self.local_file_path = "CLIENT2/output/local_model_client2.pkl" 
        self.json_path = "CLIENT2/output/client2_results.json"          
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):                 #order 3.
        """Receive parameters and apply them to the local model."""
        print("client 3")
        params_dict = zip(self.model_g.state_dict().keys(), parameters)  
        #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        state_dict = OrderedDict({k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict})  #Exclude model layers that are not sent to the central server
        
        self.model_g.load_state_dict(state_dict, strict=False)       
        
    def get_parameters(self, config: Dict[str, Scalar]):  #order 1. {}
        """
        The client 2 uses the get_parameters function to extract the current model parameters.
        These parameters are then sent to the central server using a communication protocol.
        """
        print("client 1")
        #return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return [val.cpu().numpy() for _, val in list(self.model_g.state_dict().items())[:-2]]  

    def fit(self, parameters, config):                    #order 2.
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """
        print("client 2")
        self.set_parameters(parameters) 
        
        if os.path.isfile(self.local_file_path):        
            with open(self.local_file_path, 'rb') as file:
                client2_state_dict = pickle.load(file)
                self.model_l.load_state_dict(client2_state_dict, strict=True) 
        else:
            print("wait for next round")         
        
        lr = config["lr"]
        #weight_decay = config["weight_decay"]
        epochs = config["local_epochs"]
        self.current_round = config["current_round"]
        # a very standard looking optimiser
        optim = torch.optim.Adam(list(self.model_g.parameters()) + list(self.model_l.parameters()), lr=lr)

        train(self.model_g, self.model_l, self.trainloader, self.valloader, optim, epochs, self.current_round, self.device)
        #self.model_l.load_state_dict(best_model_params)
        
        """
        Flower clients need to return three arguments: the updated model, the number of examples in the client (although this depends a bit         on your choice of aggregation strategy), and a dictionary of metrics (here you can add any additional data, but these are ideally           small data structures)        
        """
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):  #order 4.
        print("client 4")
        #self.set_parameters(parameters)         
        val_loss, val_acc, val_f1 = val(self.model_l, self.valloader, self.device)
        ###########JSON IN##############
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as file2:
                existing_data = json.load(file2)
                existing_data[self.current_round] = {}
                existing_data[self.current_round]["val_loss"] = val_loss
                existing_data[self.current_round]["val_acc"] = val_acc  
                existing_data[self.current_round]["val_f1"] = val_f1
                if existing_data["best_score"] < val_f1:
                    existing_data["best_score"] = val_f1
                    print("Best ACC achieved......", existing_data["best_score"])                     
                    state_dict_m = {name: param for name, param in self.model_l.state_dict().items()}
                    with open(self.file_path, 'wb') as file_m:
                        pickle.dump(state_dict_m, file_m) 
                    test_loss, test_acc, test_f1 = test(self.model_l, self.testloader, self.current_round, self.device)    
                    existing_data["test_loss"] = test_loss
                    existing_data["test_acc"] = test_acc  
                    existing_data["test_f1"] = test_f1                      
                        
            with open(self.json_path, 'w') as file2:    
                json.dump(existing_data, file2, indent=4)
        else:
            data = {}
            with open(self.json_path, 'w') as file2:
                data[self.current_round] = {}
                data[self.current_round]["val_loss"] = val_loss
                data[self.current_round]["val_acc"] = val_acc     
                data[self.current_round]["val_f1"] = val_f1
                data["best_score"] = val_f1

                test_loss, test_acc, test_f1 = test(self.model_l, self.testloader, self.current_round, self.device)  
                data["test_loss"] = test_loss
                data["test_acc"] = test_acc  
                data["test_f1"] = test_f1            
                json.dump(data, file2, indent=4)        
        ###########JSON IN############## 
                                          
        return float(val_loss), len(self.valloader), {"accuracy": val_acc}

def prepare_dataset_client2(path, num_partitions, file_path, batch_size):
    with open(file_path) as file:
        json_data = json.load(file)   
        train_num_ids = json_data["client "+str(2)]["train"]
        val_num_ids = json_data["client "+str(2)]["val"]
        test_num_ids = json_data["client "+str(2)]["test"]
        
        train_num_dataset = Dataset(
            img_ids=train_num_ids,
            img_dir=os.path.join(path),
            transform=get_training_augmentation(224,224))        
        train_dataloader = DataLoader(train_num_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        val_num_dataset = Dataset(
            img_ids=val_num_ids,
            img_dir=os.path.join(path),
            transform=get_validation_augmentation(224,224))            
        val_dataloader = DataLoader(val_num_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        test_num_dataset = Dataset(
            img_ids=test_num_ids,
            img_dir=os.path.join(path),
            transform=get_validation_augmentation(224,224))            
        test_dataloader = DataLoader(test_num_dataset, batch_size=batch_size, shuffle=False, num_workers=0)        
       
    return train_dataloader, val_dataloader, test_dataloader
    
@hydra.main(config_path="../../conf", config_name="base", version_base=None)
def main(cfg: DictConfig):     
    trainloader, validationloader, testloader = prepare_dataset_client2(cfg.path, cfg.num_clients, cfg.json_path, cfg.batch_size)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(
            trainloader=trainloader,
            valloader=validationloader,
            testloader=testloader,
            num_classes=cfg.num_classes,
        ),
    )      
    
if __name__ == '__main__':
    main()
    
    