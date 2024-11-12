from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl

from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, trainloader, vallodaer, num_classes) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = vallodaer
        self.model = Net(num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):                 #order 3.
        """Receive parameters and apply them to the local model."""
        print("client 3")
        params_dict = zip(self.model.state_dict().keys(), parameters)        
        #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        state_dict = OrderedDict({k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)     

    def get_parameters(self, config: Dict[str, Scalar]):  #order 1.
        """Extract model parameters and return them as a list of numpy arrays."""
        print("client 1")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):                    #order 2.
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """
        print("client 2")
        self.set_parameters(parameters)

        lr = config["lr"]
        weight_decay = config["weight_decay"]
        epochs = config["local_epochs"]

        # a very standard looking optimiser
        optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        train(self.model, self.trainloader, optim, epochs, self.device)

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):  #order 4.
        print("client 4")
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"accuracy": accuracy}


def generate_client_fn(trainloaders, valloaders, num_classes):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """
    def client_fn(cid: str):

        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            num_classes=num_classes,
        )

    # return the function to spawn client
    return client_fn 