import argparse
import warnings
from typing import Dict

import dirichlet_dist as dd
import flwr as fl
import metrics
import utils
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

test_split = 0.2
density = 0.5
epochs = 1
num_rounds = 100
data_path = (
    "../data/diabetes_data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
)
class_col = "Diabetes_binary"


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(
    model: LogisticRegression, random_seed: int, test_split, density, num_clients
):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    data_dist = dd.DirichletDist(
        data_path=data_path,
        class_col=class_col,
        num_clients=10,
        num_classes=2,
        random_state=random_seed,
        test_split=test_split,
    )

    _, test_data = data_dist.get_dirichlet_noniid_splits(density=density)
    X_test = test_data["data"]
    y_test = test_data["target"]
    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        y_pred = model.predict(X_test)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        scores = metrics.get_scores(y_test, y_pred)
        scores["loss"] = loss
        scores["Accuracy"] = accuracy
        wandb.log(scores)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--num_clients",
        type=int,
        default=10,
        choices=range(1, 11),
        required=False,
        help="Specifies how many clients the bash script will start.",
    )
    args = parser.parse_args()
    num_clients = args.num_clients

    wandb.init(
        # set the wandb project where this run will be logged
        project="federated_learning_diabetes",
        # track hyperparameters and run metadata
        config={
            "model": "Logistic Regression",
            "dataset": "Diabetes Health Indicators",
            "epochs": epochs,
            "n_features": 21,
            "test_split": test_split,
            "num_clients": num_clients,
            "density": density,
            "num_rounds": num_rounds,
        },
    )

    model = LogisticRegression()
    utils.set_initial_params(model, n_classes=2, n_features=21)
    random_seed = 42
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(
            model,
            random_seed=random_seed,
            test_split=test_split,
            density=density,
            num_clients=num_clients,
        ),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )
