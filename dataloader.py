from datasets import load_dataset
from config import TRAIN_DATASET_PATH, TEST_DATASET_PATH, FOLD
import pandas as pd
from sklearn.model_selection import train_test_split

#import os
#from datasets import Dataset
#from flwr_datasets import FederatedDataset
#from datasets.utils.logging import disable_progress_bar
#from torchvision.transforms import Compose, ToTensor, Normalize
#from config import NUM_CLIENTS


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    assert df is not None and not df.empty, f"DataFrame is empty or not loaded correctly from {file_path}"
    return df

## client datasets
def get_training_datasets_by_client(client_id, test_size=0.2, fold=FOLD):
    train_file_path = TRAIN_DATASET_PATH.format(client_id, fold)
    training_dataset = load_dataset(train_file_path)
    train_set, val_set = train_test_split(training_dataset, test_size=test_size, random_state=42, stratify=training_dataset['Label'])

    return train_set, val_set

## client datasets
def get_evaluation_datasets_by_client(client_id, fold=FOLD):
    test_file_path = TEST_DATASET_PATH.format(client_id, fold)
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
