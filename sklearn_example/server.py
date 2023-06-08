from typing import Dict

import dirichlet_dist as dd
import flwr as fl
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression, random_seed: int):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    #_, (X_test, y_test) = utils.load_diabetes_data(random_seed=random_seed)
    data_path = "../data/diabetes_data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    data_dist = dd.DirichletDist(data_path=data_path,
                                class_col="Diabetes_binary",
                                num_clients=10,
                                num_classes=2,
                                random_state=random_seed,
                                test_split=0.2)
    
    _, test_data = data_dist.get_dirichlet_noniid_splits(density=1)
    X_test = test_data["data"]
    y_test = test_data["target"]
    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model, n_classes=2, n_features=21)
    random_seed = 42
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model, random_seed=random_seed),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )
