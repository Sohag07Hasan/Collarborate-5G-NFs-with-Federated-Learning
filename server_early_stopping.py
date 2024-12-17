from flwr.server.server import Server, evaluate_clients, EvaluateResultsAndFailures
from flwr.server.history import History
from flwr.common.typing import Scalar
from typing import Optional, Tuple, Dict
from flwr.common.logger import log
from logging import INFO, WARNING
import timeit



class EarlyStoppingServer(Server):
    """Custom Server class with early stopping based on strategy evaluation."""

    def __init__(self, *args, **kwargs):
        """Initialize the EarlyStoppingServer."""
        super().__init__(*args, **kwargs)

    # def evaluate_round(self, server_round: int, timeout: float):
    #     """
    #     Extended evaluate_round to check early stopping condition.

    #     Returns:
    #         A flag indicating whether early stopping is triggered.
    #     """
    #     # Perform the standard evaluation round
    #     results = super().evaluate_round(server_round=server_round, timeout=timeout)

    #     # After evaluation, check for early stopping condition in strategy
    #     if self.strategy.is_early_stop_applicable:
    #         log(WARNING, f"Early stopping triggered at round {server_round}")
    #         return True  # Signal to stop training
    #     return False  # Continue training

    # def fit(self, num_rounds: int, timeout: float):
    #     """Run federated training with early stopping."""
    #     history = History()

    #     # Initialize parameters
    #     log(INFO, "[INIT] Starting federated training with Early Stopping")
    #     self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)

    #     start_time = timeit.default_timer()

    #     for current_round in range(1, num_rounds + 1):
    #         log(INFO, f"[ROUND {current_round}]")

    #         # Perform training round
    #         res_fit = self.fit_round(server_round=current_round, timeout=timeout)
    #         if res_fit is not None:
    #             parameters_prime, fit_metrics, _ = res_fit
    #             if parameters_prime:
    #                 self.parameters = parameters_prime

    #         # Perform evaluation round and check for early stopping
    #         if self.evaluate_round(server_round=current_round, timeout=timeout):
    #             log(INFO, f"Stopping early at round {current_round}")
    #             break

    #     # Graceful shutdown of clients
    #     self.disconnect_all_clients(timeout=timeout)

    #     end_time = timeit.default_timer()
    #     elapsed = end_time - start_time
    #     log(INFO, f"Training completed in {elapsed:.2f} seconds")
    #     return history, elapsed

        # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        else:
            log(INFO, "Evaluation returned no results (`None`)")

        # Run federated learning for num_rounds
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
            
                    # After evaluation, check for early stopping condition in strategy
            if self.strategy.is_early_stop_applicable:
                log(WARNING, f"Early stopping triggered at round {current_round}")
                break ##breaking the loop
            

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed

    