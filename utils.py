''' 
We'll be training the model in a Federated setting. In order to do that, we need to define two functions:

* `train()` that will train the model given a dataloader.
* `test()` that will be used to evaluate the performance of the model on held-out data, e.g., a training set.
'''
from config import NUM_ROUNDS, GLOBAL_MODEL_PATH, NUM_CLASSES, BATCH_SIZE
from model import Net
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset


def train(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            features, labels = batch[0].to(device), batch[1].to(device)
            optim.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            optim.step()


def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

    
    ##After each round it will be used
def get_evaluate_fn(centralized_testset):
    """This is a function that returns a function. The returned
    function (i.e. `evaluate_fn`) will be executed by the strategy
    at the end of each round to evaluate the stat of the global
    model."""

    def evaluate_fn(server_round: int, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = Net(num_classes=NUM_CLASSES)

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)  # send model to device

        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Save the model after the final round
        if server_round == NUM_ROUNDS:  #NUM_ROUNDS is defined globally
            torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
            print(f"Global model saved at round {server_round}")

        # Apply transform to dataset
        #testset = centralized_testset.with_transform(apply_transforms)
        testloader = DataLoader(to_tensor(centralized_testset), batch_size=BATCH_SIZE)
        # call test
        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn   

##clear the cache of the Cuda
def clear_cuda_cache():
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")


## Convert a panda dataframe into a Tensordataset for to be useable by torch
## @df = panda dataframe
def to_tensor(df):
    # Separate features and labels
    X = df.drop(columns="Label").values  # Replace "label" with the actual label column name
    y = df["Label"].values
    # Convert to PyTorch tensors
    # Also reshaping it 5 by 6
    #X_tensor = torch.tensor(X.reshape(-1, 1, 5, 6), dtype=torch.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)
