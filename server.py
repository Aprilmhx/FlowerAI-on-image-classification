from collections import OrderedDict
from typing import List, Tuple
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import flwr as fl
from flwr.common import Metrics
import os
import json
from torch.utils.data import random_split, DataLoader

from dataset_server import Dataset
from model_server import Net, test
from Data_Augmentation import get_training_augmentation, get_validation_augmentation

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    print("!!!!!!!!!!!!!!!!!!!!!!weighted", metrics)
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):

        return {
            "lr": config.lr,
            #"weight_decay": config.weight_decay,
            "local_epochs": config.local_epochs,
            "current_round": server_round,
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):
        model = Net(num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params_dict = zip(model.state_dict().keys(), parameters)   #'model.conv1.weight'...|weights|num:122-2
         
        #https://github.com/orgs/adap/discussions/1722
        #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        #state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        #exclude model.fc.weight | model.fc.bias
        state_dict = OrderedDict({k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict}) #120
        #state_dict = OrderedDict(list(state_dict.items())[:-2])
        print("state_dict_server", len(state_dict))
        model.load_state_dict(state_dict, strict=False)
        print("!!!!!!!!!!!!!!!!!!")
        loss, accuracy = test(model, testloader, device)

        return loss, {"accuracy": accuracy}

    return evaluate_fn


def prepare_dataset_server(path, num_partitions, file_path, batch_size):
    with open(file_path) as file:
        json_data = json.load(file)   
        train_dataloaders = []
        val_dataloaders = []      
        test_num_ids = []
        for num in range(num_partitions):
            test_num_ids+=json_data["client "+str(num+1)]["test"]
        test_num_dataset = Dataset(
            img_ids=test_num_ids,
            img_dir=os.path.join(path),
            transform=get_validation_augmentation(224,224)) 
        test_dataloaders = DataLoader(test_num_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return test_dataloaders

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):      
    testloaders = prepare_dataset_server(cfg.path, cfg.num_clients, cfg.json_path, cfg.batch_size)
    strategy = fl.server.strategy.FedAvg(  #FedProx FedAvg
        #proximal_mu=10,
        fraction_fit=1.0,                                    #Fraction of clients used during training
        fraction_evaluate=1.0,                               #Fraction of clients used during validation
        min_fit_clients=4,       #Minimum number of clients used during training                             
        min_evaluate_clients=4, #Minimum number of clients used during validation
        min_available_clients=4,               #Minimum number of total clients in the system
        on_fit_config_fn=get_on_fit_config(                  #Function used to configure training
            cfg.config_fit
        ),  
        ##evaluate_metrics_aggregation_fn=weighted_average,
        ##evaluate_fn=get_evaluate_fn(8, testloaders),
    )  # a function to run on the server side to evaluate the global model.    

    fl.server.start_server(
        server_address = "0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),        
        strategy=strategy
    )    
    
if __name__ == '__main__':
    main()