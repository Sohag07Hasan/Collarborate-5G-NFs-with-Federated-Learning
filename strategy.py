import flwr as fl

def create_strategy(fit_config_fn, weighted_average_fn, evaluate_fn, fraction_fit=0.1, fraction_evaluate=0.5):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,  # Sample 10% of available clients for training
        fraction_evaluate=fraction_evaluate,  # Sample 50% of available clients for evaluation
        on_fit_config_fn=fit_config_fn,
        evaluate_metrics_aggregation_fn=weighted_average_fn,  # aggregates federated metrics
        evaluate_fn=evaluate_fn,  # global evaluation function
    )
    return strategy
