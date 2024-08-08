import flwr as fl
from dataloader import get_datasets
from client import create_client
from config import SERVER_ADDRESS

if __name__ =="__main__":
    ## Collecting Datasets
    mnist_fds, centralized_testset = get_datasets()    
   
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS, 
        client=create_client(mnist_fds, 2)
    )