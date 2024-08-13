import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataloader import get_evaluation_datasets_by_client  # Assuming this function gets local client datasets
from model import Net
from collections import OrderedDict
from config import NUM_CLASSES, NUM_CLIENTS, GLOBAL_MODEL_PATH, BATCH_SIZE
from torch.utils.data import DataLoader
from utils import to_tensor

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
        for batch in dataloader:
            features, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(features)
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
    num_clients = NUM_CLIENTS  # Define or import this variable
    for client_id in range(1, num_clients+1):
        testset = get_evaluation_datasets_by_client(client_id)  
        testloader = DataLoader(to_tensor(testset), batch_size=BATCH_SIZE)
        preds, labels = run_inference(model, testloader, device)
        classes = np.arange(NUM_CLASSES)  # Define or import this variable
        plot_confusion_matrix(labels, preds, classes)

if __name__ == "__main__":
    main_all_together()
