# strategy.py
import flwr as fl
from client import save_global_model

NUM_CLIENTS = 4  # Number of Clients
MIN_EVAL_CLIENTS = 2


def fit_metrics_aggregation_fn(metrics):
    total_accuracy = 0.0
    total_samples = 0

    for num_samples, client_metrics in metrics:
        accuracy = client_metrics.get("accuracy", 0.0)
        total_accuracy += accuracy * num_samples
        total_samples += num_samples

    avg_accuracy = total_accuracy / total_samples
    return {"accuracy": avg_accuracy}



def evaluate_metrics_aggregation_fn(metrics):
    # Print the type and content of metrics to understand its structure
    print("Type of metrics:", type(metrics))
    print("Content of metrics:", metrics)

    total_accuracy = 0.0
    total_samples = 0

    for num_samples, client_metrics in metrics:
        print("Type of client_metrics:", type(client_metrics))
        print("Content of client_metrics:", client_metrics)

        # Assuming client_metrics is a dictionary
        if isinstance(client_metrics, dict):
            accuracy = client_metrics.get("accuracy", 0.0)

            total_accuracy += accuracy * num_samples
            total_samples += num_samples
        else:
            raise TypeError("Expected client_metrics to be a dictionary, but got {}".format(type(client_metrics)))

    avg_accuracy = total_accuracy / total_samples

    return {"accuracy": avg_accuracy}

#Save the model
def on_round_end(server_round, results, failures):
    print(f"Completed round {server_round}")
    save_global_model()

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=MIN_EVAL_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    on_fit_config_fn=lambda rnd: {"epoch_global": rnd},
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    on_round_end=on_round_end
)
