import argparse
import warnings

import dirichlet_dist as dd
import flwr as fl
import numpy as np
import server as server
import utils
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    random_seed = 42
    
    # (X_train, y_train), _ = utils.load_diabetes_data(random_seed=random_seed)
    
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 11),
        required=False,
        help="Specifies the artificial data partition of the dataset to be used. \
        Picks partition 0 by default",
    )
    
    parser.add_argument(
        "--num_clients",
        type=int,
        default=10,
        choices=range(1, 11),
        required=False,
        help="Specifies how many clients the bash script will start.",
    )
    
    args = parser.parse_args()
    # Split train set into 10 partitions and randomly use one for training.
    np.random.seed(random_seed)
    # Subtract one from the id because array's start from 0.
    client_id = args.partition - 1
    num_clients = args.num_clients
    #(X_train, y_train) = utils.partition(X_train, y_train, 10)[client_id]
    
    data_dist = dd.DirichletDist(data_path=server.data_path,
                                class_col="Diabetes_binary",
                                num_clients=num_clients,
                                num_classes=2,
                                random_state=random_seed,
                                test_split=server.test_split)
    
    train_data, _ = data_dist.get_dirichlet_noniid_splits(density=server.density)
    X_train = train_data[client_id]["data"]
    y_train = train_data[client_id]["target"]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=server.epochs,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model, n_classes=2, n_features=21)

    # Define Flower client
    class LogisticClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}
        
    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=LogisticClient())
