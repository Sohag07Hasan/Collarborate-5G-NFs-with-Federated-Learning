import flwr as fl
from strategy import create_strategy  
from dataloader import get_centralized_testset
from config import LEARNING_RATE, EPOCHS, NUM_ROUNDS, SERVER_ADDRESS, NUM_ROUNDS, HISTORY_PATH_TXT, HISTORY_PATH_PKL, TRAINING_TIME, EARLY_STOPPING_ROUNDS, IMPROVEMENT_THRESHOLD
from flwr.common import Metrics, Scalar
from utils import get_evaluate_fn, clear_cuda_cache
from typing import Dict, List, Tuple
#import matplotlib.pyplot as plt
import pickle
import time


## Collecting Datasets
centralized_testset = get_centralized_testset()

# Early stopping parameters
early_stopping_rounds = EARLY_STOPPING_ROUNDS  # Stop training if no significant improvement after this many rounds
improvement_threshold = IMPROVEMENT_THRESHOLD  # Minimum required improvement in accuracy to continue
best_accuracy = 0.0
no_improvement_counter = 0  # Counter to track rounds with no improvement


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
    #print(metrics)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

#Store the history
def save_history(history, path_text=HISTORY_PATH_TXT, path_pkl=HISTORY_PATH_PKL):
    with open(path_pkl, "wb") as file:
        pickle.dump(history, file)

        # Save as a plain text file
    with open(path_text, 'w') as file:
        file.write(str(history))
    print(f"history saved as text @ {path_text}")
    print(f"history saved as picle @ {path_text}")
    
#Store Training time 
def save_training_time(start_time, end_time):
    training_time = end_time - start_time
           # Save as a plain text file
    with open(TRAINING_TIME, 'w') as file:
        file.write(str(training_time))


# Early stopping function
def early_stopping(history: List[Dict], round_num: int) -> bool:
    global best_accuracy, no_improvement_counter
    
    # Get the accuracy from the latest round
    current_accuracy = history[-1]["accuracy"]

    # Check for improvement
    if current_accuracy - best_accuracy > improvement_threshold:
        best_accuracy = current_accuracy
        no_improvement_counter = 0  # Reset the counter if improvement
    else:
        no_improvement_counter += 1  # Increment the counter if no significant improvement

    # Check if we need to stop
    if no_improvement_counter >= early_stopping_rounds:
        print(f"Stopping early at round {round_num} due to lack of improvement.")
        return True  # Stop the training process
    return False


if __name__ == "__main__":
    
    #clear the GPU cache
    clear_cuda_cache()

    # Start timing
    start_time = time.time()

    # Define the server configuration and start the server
    server_config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    strategy = create_strategy(fit_config, weighted_average, get_evaluate_fn(centralized_testset), fit_metrics_aggregation)
    
    history = fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=server_config,
        strategy=strategy,  # Use the imported strategy
    )
    # End timing
    end_time = time.time()
    save_training_time(start_time, end_time)
    save_history(history)

    # Add server stopping code
    #fl.server.stop_server()
    #print("Server has been stopped.")
