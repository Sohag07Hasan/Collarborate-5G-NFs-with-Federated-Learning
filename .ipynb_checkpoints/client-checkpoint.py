import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNNBinaryClassifier
from train import train
from test import test
from dataloader import get_data_loaders, load_data
import joblib
import logging

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, scaler):
        self.model = model.to('cuda')  # Move model to GPU
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss().to('cuda')  # Move criterion to GPU
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scaler = scaler

        # Set up logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_parameters(self, config):
        self.logger.debug("Getting parameters")
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters, config=None):
        self.logger.debug("Setting parameters")
        for val, param in zip(parameters, self.model.parameters()):
            param.data = torch.tensor(val).to('cuda')

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        epoch_global = config["epoch_global"]
        self.logger.info(f"Starting training round {epoch_global}...")
        train(self.model, self.train_loader, self.criterion, self.optimizer, epochs=1)
        self.logger.info(f"Finished training round {epoch_global}.")
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        self.logger.debug("Evaluating model")
        loss, accuracy = test(self.model, self.test_loader, self.criterion)
        self.logger.info(f"Evaluation result: loss={loss:.4f}, accuracy={accuracy:.4f}")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

def client_fn(cid: str):
    # Load the data for this client
    client_folders = ["client_1", "client_2", "client_3", "client_4"]
    train_datasets, test_datasets = load_data([client_folders[int(cid)]])
    train_loader, test_loader = get_data_loaders(train_datasets[0], test_datasets[0])

    # Load the scaler for this client
    scaler = joblib.load(f'./scalers/{client_folders[int(cid)]}_scaler.pkl')

    # Create a model and Flower client
    model = CNNBinaryClassifier()
    client = FlowerClient(model, train_loader, test_loader, scaler)
    
    return client.to_client()

def save_global_model(model, path='./models'):
    torch.save(model.state_dict(), path)
