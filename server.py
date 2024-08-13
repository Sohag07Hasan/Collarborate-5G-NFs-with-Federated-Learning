import flwr as fl
from strategy import create_strategy  
from dataloader import get_datasets, get_centralized_testset
from config import LEARNING_RATE, EPOCHS, NUM_ROUNDS, SERVER_ADDRESS, NUM_ROUNDS, HISTORY_PATH
from flwr.common import Metrics, Scalar
from utils import get_evaluate_fn, clear_cuda_cache
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt


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

#Store the history
def save_history(history, path=HISTORY_PATH):
    # Extracting loss
    rounds = []
    loss_distributed = []
    loss_centralized = []

    for i, loss in enumerate(history.losses_distributed):
        rounds.append(i + 1)
        loss_distributed.append(loss)

    for i, loss in enumerate(history.losses_centralized):
        loss_centralized.append(loss)

    # Extracting accuracy
    accuracy_distributed = [acc for _, acc in history.metrics_distributed['accuracy']]
    accuracy_centralized = [acc for _, acc in history.metrics_centralized['accuracy']]

    # Creating DataFrame
    df = pd.DataFrame({
        'round': rounds,
        'loss_distributed': loss_distributed,
        'loss_centralized': loss_centralized,
        'accuracy_distributed': accuracy_distributed,
        'accuracy_centralized': accuracy_centralized
    })
     # Save to CSV
    df.to_csv(path, index=False)


#plotting metrics from saved file
def plot_saved_history(path=HISTORY_PATH):
    # Read the metrics from the CSV file
    df = pd.read_csv(path)
    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(df['round'], df['loss_distributed'], label='Loss Distributed')
    plt.plot(df['round'], df['loss_centralized'], label='Loss Centralized')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Loss over Rounds')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(df['round'], df['accuracy_distributed'], label='Accuracy Distributed')
    plt.plot(df['round'], df['accuracy_centralized'], label='Accuracy Centralized')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Rounds')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    
    #clear the GPU cache
    clear_cuda_cache()

    # Define the server configuration and start the server
    server_config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    strategy = create_strategy(fit_config, weighted_average, get_evaluate_fn(centralized_testset), fit_metrics_aggregation)
    history = fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=server_config,
        strategy=strategy,  # Use the imported strategy
    )
    # Save as a plain text file
    with open('history.txt', 'w') as file:
        file.write(str(history))
    #save_history(history)
    #plot_saved_history()