import argparse
import warnings

import dirichlet_dist as dd
import flwr as fl
import numpy as np
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

if __name__ == "__main__":
    random_seed = 42
    
    # (X_train, y_train), _ = utils.load_diabetes_data(random_seed=random_seed)
    
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    args = parser.parse_args()
    data_path = "../data/diabetes_data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    # Split train set into 10 partitions and randomly use one for training.
    np.random.seed(random_seed)
    client_id = args.partition
    #(X_train, y_train) = utils.partition(X_train, y_train, 10)[client_id]
    
    data_dist = dd.DirichletDist(data_path=data_path,
                                class_col="Diabetes_binary",
                                num_clients=10,
                                num_classes=2,
                                random_state=random_seed,
                                test_split=0.2)
    
    train_data, _ = data_dist.get_dirichlet_noniid_splits(density=1)
    X_train = train_data[client_id]["data"]
    y_train = train_data[client_id]["target"]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=10,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model, n_classes=2, n_features=21)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
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
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
