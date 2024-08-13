from datasets import Dataset, load_dataset
from flwr_datasets import FederatedDataset
#from datasets.utils.logging import disable_progress_bar
from torchvision.transforms import Compose, ToTensor, Normalize
from config import TRAIN_DATASET_PATH, TEST_DATASET_PATH
import pandas as pd
import os
from sklearn.model_selection import train_test_split


# Let's set a simulation involving a total of 100 clients
from config import NUM_CLIENTS


# Download MNIST dataset and partition the "train" partition (so one can be assigned to each client)
def get_datasets():    
    mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
    # Let's keep the test set as is, and use it to evaluate the global model on the server
    centralized_testset = mnist_fds.load_split("test")
    return mnist_fds, centralized_testset

##not in use
def apply_transforms(batch):
    """Get transformation for MNIST dataset"""
    # transformation to convert images to tensors and apply normalization
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch

##
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    assert df is not None and not df.empty, f"DataFrame is empty or not loaded correctly from {file_path}"
    return df

## client datasets
def get_training_datasets_by_client(client_id, test_size=0.2):
    train_file_path = TRAIN_DATASET_PATH.format(client_id)
    training_dataset = load_dataset(train_file_path)
    train_set, val_set = train_test_split(training_dataset, test_size=test_size, random_state=42, stratify=training_dataset['Label'])

    return train_set, val_set

## client datasets
def get_evaluation_datasets_by_client(client_id):
    test_file_path = TEST_DATASET_PATH.format(client_id)
    testing_dataset = load_dataset(test_file_path)
    return testing_dataset

## combine testset from all the clients
## shuffle and reset the index and return a combined datasets
def get_centralized_testset():
    client_1_testset = get_evaluation_datasets_by_client(1)
    client_2_testset = get_evaluation_datasets_by_client(2)
    client_3_testset = get_evaluation_datasets_by_client(3)
    client_4_testset = get_evaluation_datasets_by_client(4)
    centralized_testset = pd.concat([client_1_testset, client_2_testset, client_3_testset, client_4_testset], axis=0)
    return centralized_testset.sample(frac=1).reset_index(drop=True)
