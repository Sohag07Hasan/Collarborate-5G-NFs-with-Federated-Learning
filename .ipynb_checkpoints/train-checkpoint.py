import torch

def train(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to('cuda'), target.to('cuda')  # Move data to GPU
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
