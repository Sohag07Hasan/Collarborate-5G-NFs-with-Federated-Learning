import flwr as fl
from flwr.server import start_server
from config import SERVER_ADDRESS, NUM_ROUNDS
from strategy import strategy  # Import the strategy from strategy.py

# Define the server configuration and start the server
server_config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)

if __name__ == "__main__":
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=server_config,
        strategy=strategy,  # Use the imported strategy
    )
