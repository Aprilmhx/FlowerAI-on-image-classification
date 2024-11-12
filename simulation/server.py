from collections import OrderedDict
from typing import List, Tuple
from omegaconf import DictConfig
import torch
from model import Net, test
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    print("weighted", metrics)
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
            "weight_decay": config.weight_decay,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):

        model = Net(num_classes)
        #print(model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params_dict = zip(model.state_dict().keys(), parameters)
        
        #https://github.com/orgs/adap/discussions/1722
        #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        #state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        state_dict = OrderedDict({k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict})
        
        #print(state_dict)
        model.load_state_dict(state_dict, strict=True)
        print("!!!!!!!!!!!!!!!!!!")
        loss, accuracy = test(model, testloader, device)

        return loss, {"accuracy": accuracy}

    return evaluate_fn