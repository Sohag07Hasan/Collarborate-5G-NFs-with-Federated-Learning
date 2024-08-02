# simulate.py
import flwr as fl
import torch
from client import client_fn
from strategy import strategy  # Import the strategy from strategy.py
import logging

# Constants
NUM_CLIENTS = 4  # Number of Clients
NUM_ROUNDS = 5

def start_simulation():
    # Set up logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)

    # Set client resources
    client_resources = {"num_cpus": 4, "num_gpus": 1.0} if torch.cuda.is_available() else {"num_cpus": 4, "num_gpus": 0.0}
    
    # Log the start of the simulation
    logger.info("Starting simulation...")

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,  # Use the imported strategy
        client_resources=client_resources,
    )

    # Log the end of simulation
    logger.info("Simulation finished. Global model saved.")

if __name__ == "__main__":
    start_simulation()
