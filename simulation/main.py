import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from server import weighted_average, get_on_fit_config, get_evaluate_fn

# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    # save the results of the simulation (see the last part of this main())
    save_path: str = HydraConfig.get().runtime.output_dir
    ## 2. Prepare your dataset
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.path, cfg.num_clients, cfg.json_path, cfg.batch_size
    )
    ## 3. Define your clients     
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)
    
    ## 4. Define your strategy
    #https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg.py
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,                                    #Fraction of clients used during training
        fraction_evaluate=1.0,                               #Fraction of clients used during validation
        min_fit_clients=cfg.num_clients_per_round_fit,       #Minimum number of clients used during training                             
        min_evaluate_clients=cfg.num_clients_per_round_eval, #Minimum number of clients used during validation
        min_available_clients=cfg.num_clients,               #Minimum number of total clients in the system
        on_fit_config_fn=get_on_fit_config(                  #Function used to configure training
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )  # a function to run on the server side to evaluate the global model.
    
    print("check_main 1")   
    ## 5. Start Simulation
    # With the dataset partitioned, the client function and the strategy ready, we can now launch the simulation!
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),
        strategy=strategy,
        client_resources={
            "num_cpus": 128,
            "num_gpus": 8,
        },  # (optional) controls the degree of parallelism of your simulation.
    )
    print("check_main 2") 
    
    ## 6. Save your results
    results_path = Path(save_path) / "results.pkl"
    results = {"history": history, "anythingelse": "here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()