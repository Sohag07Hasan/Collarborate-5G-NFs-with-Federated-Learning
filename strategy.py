import flwr as fl
from config import FRACTION_FIT, FRACTION_EVAL, NUM_CLIENTS, MIN_FIT_CLIENTS, MIN_EVAL_CLIENTS, Q_PARAM, QFFL_LR
from typing import Callable


def create_strategy(fit_config_fn: Callable, weighted_average_fn: Callable, evaluate_fn: Callable, fit_metrics_aggregation_fn: Callable):
    strategy = fl.server.strategy.QFedAvg(
        fraction_fit=FRACTION_FIT,  # Fraction of clients used for training in each round
        fraction_evaluate=FRACTION_EVAL,  # Fraction of clients used for evaluation in each round
        min_fit_clients=MIN_FIT_CLIENTS,  # Minimum number of clients for training
        min_evaluate_clients=MIN_EVAL_CLIENTS,  # Minimum number of clients for evaluation
        min_available_clients=NUM_CLIENTS,  # Minimum number of clients that must be available to start a round
        q_param=Q_PARAM,  # Fairness parameter (higher values prioritize fairness more strongly)
        qffl_learning_rate=QFFL_LR,
        on_fit_config_fn=fit_config_fn,  # Callback function for training configuration
        evaluate_metrics_aggregation_fn=weighted_average_fn,  # Aggregates evaluation metrics
        evaluate_fn=evaluate_fn,  # Server-side evaluation function
    )
    return strategy
