from datasets import Dataset, load_dataset
from flwr_datasets import FederatedDataset
from datasets.utils.logging import disable_progress_bar
from torchvision.transforms import Compose, ToTensor, Normalize

# Let's set a simulation involving a total of 100 clients
from config import NUM_CLIENTS


# Download MNIST dataset and partition the "train" partition (so one can be assigned to each client)
def get_datasets():    
    mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
    # Let's keep the test set as is, and use it to evaluate the global model on the server
    centralized_testset = mnist_fds.load_split("test")
    return mnist_fds, centralized_testset

# def get_datasets():
#     dataset = load_dataset("mnist", trust_remote_code=True)

#     mnist_fds = FederatedDataset(
#         dataset=dataset, 
#         partitioners={"train": NUM_CLIENTS}
#     )
#     # Let's keep the test set as is, and use it to evaluate the global model on the server
#     centralized_testset = mnist_fds.load_split("test")
#     return mnist_fds, centralized_testset

def apply_transforms(batch):
    """Get transformation for MNIST dataset"""
    # transformation to convert images to tensors and apply normalization
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


## client datasets
def get_client_datasets(client_id):
    _, dataset = get_datasets()
    return dataset