import flwr as fl
from config import SERVER_ADDRESS, NUM_ROUNDS
from strategy import create_strategy  # Import the strategy from strategy.py

#from client import get_client_fn
from dataloader import get_datasets, get_centralized_testset
from config import LEARNING_RATE, EPOCHS, NUM_ROUNDS
from flwr.common import Metrics, Scalar
from utils import get_evaluate_fn
from typing import Dict, List, Tuple


## Collecting Datasets
centralized_testset = get_centralized_testset()

#before fitting each client locally this function will send the fit configuraiton
def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": EPOCHS,  # Number of local epochs done by clients
        "lr": LEARNING_RATE,  # Learning rate to use by clients during fit()
    }
    return config

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Example: compute weighted average accuracy
    print(metrics)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    
    # Define the server configuration and start the server
    server_config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    strategy = create_strategy(fit_config, weighted_average, get_evaluate_fn(centralized_testset), fit_metrics_aggregation)
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=server_config,
        strategy=strategy,  # Use the imported strategy
    )
