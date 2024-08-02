import torch
from model import CNNBinaryClassifier
from dataloader import load_data, get_data_loaders
import joblib

def evaluate_model(model_path, scaler_path, client_folder):
    # Load the saved model
    model = CNNBinaryClassifier().to('cuda')  # Move model to GPU
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the saved scaler
    scaler = joblib.load(scaler_path)

    # Load the data
    _, test_datasets = load_data([client_folder])
    _, test_loader = get_data_loaders(None, test_datasets[0])

    # Define the loss criterion
    criterion = nn.CrossEntropyLoss().to('cuda')  # Move criterion to GPU

    # Evaluate the model
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cuda'), target.to('cuda')  # Move data to GPU
            data = torch.tensor(scaler.transform(data.cpu()), dtype=torch.float32).to('cuda')  # Apply the scaler to the input data and move to GPU
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    evaluate_model("./models/global_model.pth", "./scalers/client_1_scaler.pkl", "client_1")
