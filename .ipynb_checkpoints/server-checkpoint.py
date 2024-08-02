# server.py
import flwr as fl
from strategy import strategy  # Import the strategy from strategy.py

# Define the server configuration and start the server
server_config = fl.server.ServerConfig(num_rounds=5)
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=server_config,
    strategy=strategy,  # Use the imported strategy
)
