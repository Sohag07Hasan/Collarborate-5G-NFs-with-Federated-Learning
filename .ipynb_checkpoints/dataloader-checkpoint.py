import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

def load_and_scale_data(file_path, scaler=None, fit_scaler=False, scaler_path=None):
    # Load the preprocessed CSV file
    df = pd.read_csv(file_path)

    # Separate features and labels
    X = df.iloc[:, :-1].values  # Exclude the last column ('Label')
    y = df['Label'].values

    # Scale the features to be between -1 and 1
    if fit_scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        # Save the scaler
        if scaler_path:
            joblib.dump(scaler, scaler_path)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y

def load_data(client_folders, dataset_path='./dataset', scaler_path='./scalers'):
    train_datasets = []
    test_datasets = []

    for client_folder in client_folders:
        train_file = f'{dataset_path}/{client_folder}/train/{client_folder}_train.csv'
        test_file = f'{dataset_path}/{client_folder}/test/{client_folder}_test.csv'
        scaler_file = f'{scaler_path}/{client_folder}_scaler.pkl'

        # Load and scale the training data and save the scaler
        X_train, y_train = load_and_scale_data(train_file, fit_scaler=True, scaler_path=scaler_file)

        # Load and scale the test data using the saved scaler
        scaler = joblib.load(scaler_file)
        X_test, y_test = load_and_scale_data(test_file, scaler=scaler, fit_scaler=False)

        train_datasets.append(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)))
        test_datasets.append(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)))

    return train_datasets, test_datasets

def get_data_loaders(train_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
