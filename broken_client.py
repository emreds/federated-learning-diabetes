import multiprocessing
import warnings
from collections import OrderedDict
from typing import List

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

#warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def load_train_sets(transform, split_cnt = 5) -> List[DataLoader]:
    trainset = CIFAR10("./data", train=True, download=True, transform=transform)
    dataset_size = len(trainset)
    split_sizes = [dataset_size // split_cnt] * split_cnt
    data_splits = random_split(trainset, split_sizes, generator=torch.Generator().manual_seed(42))
    
    dataloaders = []
    for dataset in data_splits:
        dataloaders.append(DataLoader(dataset, batch_size=32, shuffle=True))
    
    return dataloaders

def load_test_set(transform):
    testset = CIFAR10("./data", train=False, download=True, transform=transform)
    return DataLoader(testset)

def load_data(train_split_cnt = 5):
    """Load CIFAR-10 (training and test set)."""
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_sets = load_train_sets(transform, train_split_cnt)
    test_set = load_test_set(transform)
    return train_sets, test_set

def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    print("Loading data... trainset")
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    print("Loading data... trainset")
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################
num_clients = 1
nets = []   
trainloader, testloader = load_data()
#trainloader = trainloaders[idx]
print(trainloader)
net = Net().to(DEVICE)

# PROBLEM IS IN THE DATALOADER

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Evaluating the model...")
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        print("Evaluating the model...")
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start 10 clients. 
# Load model and data (simple CNN, CIFAR-10)


#server_address = 

epochs = 1
fl.client.start_numpy_client(server_address="localhost:8080",
                            client = FlowerClient()
                            )
"""
for i in range(len(trainloaders)):
    p = multiprocessing.Process(target=fl.client.start_numpy_client, 
                                kwargs={"server_address": "localhost:8080",
                                        "client": FlowerClient(net, trainloaders[i], testloader, epochs)})
    # Start Flower client
    p.start()
    processes.append(p)

for p in processes:
    p.join() 
"""