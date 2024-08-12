import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataloader import get_client_datasets, apply_transforms  # Assuming this function gets local client datasets
from model import Net
from collections import OrderedDict
from config import NUM_CLASSES, NUM_CLIENTS, GLOBAL_MODEL_PATH
from torch.utils.data import DataLoader

# Load the global model from the saved path
def load_model(model_path=GLOBAL_MODEL_PATH, num_classes=NUM_CLASSES):
    model = Net(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Run inference on a client's dataset
def run_inference(model, dataloader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

# Draw confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def main_all_together():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model()
    model.to(device)
    
    all_preds = []
    all_labels = []
    num_clients = NUM_CLIENTS  # Define or import this variable
    
    testset = get_client_datasets(1)  
    client_dataloader = DataLoader(testset.with_transform(apply_transforms), batch_size=50)
    
    for client_id in range(num_clients):
        print(f"Running inference for client {client_id}")
        #client_dataloader = get_client_datasets(client_id)  # Load the local client data
        preds, labels = run_inference(model, client_dataloader, device)
        
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    classes = np.arange(NUM_CLASSES)  # Define or import this variable
    plot_confusion_matrix(all_labels, all_preds, classes)

if __name__ == "__main__":
    main_all_together()
