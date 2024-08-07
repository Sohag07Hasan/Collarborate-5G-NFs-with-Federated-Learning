import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar
from dataloader import get_datasets, apply_transforms
from typing import Dict, List, Tuple
from utils import get_evaluate_fn, clear_cuda_cache
from config import LEARNING_RATE, EPOCHS, NUM_CLIENTS, NUM_ROUNDS
from strategy import create_strategy
#from flwr_datasets import FederatedDataset
from datasets.utils.logging import disable_progress_bar
from client import get_client_fn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

## Collecting Datasets
mnist_fds, centralized_testset = get_datasets()
client_fn_callback = get_client_fn(mnist_fds)


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

strategy = create_strategy(fit_config, weighted_average, get_evaluate_fn(centralized_testset))

#Client Resources
client_resources = {"num_cpus": 1, "num_gpus": 1}

# Let's disable tqdm progress bar in the main thread (used by the server)
disable_progress_bar()

# Clear the Cache from GPU
clear_cuda_cache()

history = fl.simulation.start_simulation(
    client_fn=client_fn_callback,  # a callback to construct a client
    num_clients=NUM_CLIENTS,  # total number of clients in the experiment
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),  # let's run for 10 rounds
    strategy=strategy,  # the strategy that will orchestrate the whole FL pipeline
    client_resources=client_resources,
    actor_kwargs={
        "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
    },
)

print(f"{history.metrics_centralized = }")

global_accuracy_centralised = history.metrics_centralized["accuracy"]
round = [data[0] for data in global_accuracy_centralised]
acc = [100.0 * data[1] for data in global_accuracy_centralised]
plt.plot(round, acc)
plt.grid()
plt.ylabel("Accuracy (%)")
plt.xlabel("Round")
plt.title("MNIST - IID - 10 clients with 10 clients per round")