"""
Utility functions and classes for Jupyter Notebooks lessons. 
"""

from collections import OrderedDict
import logging
from logging import INFO
import warnings

from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerAppComponents
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common.logger import (
    ConsoleHandler,
    console_handler,
    FLOWER_LOGGER,
    LOG_COLORS,
)
from logging import LogRecord
from flwr.server import ServerApp, ServerConfig
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor


# Customize logging for the course.
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == INFO


FLOWER_LOGGER.removeHandler(console_handler)
warnings.filterwarnings("ignore")

# To filter logging coming from the Simulation Engine
# so it's more readable in notebooks
from logging import ERROR
backend_setup = {"init_args": {"logging_level": ERROR, "log_to_driver": True}}

class ConsoleHandlerV2(ConsoleHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record: LogRecord) -> str:
        """Format function that adds colors to log level."""
        if self.json:
            log_fmt = "{lvl='%(levelname)s', time='%(asctime)s', msg='%(message)s'}"
        else:
            log_fmt = (
                f"{LOG_COLORS[record.levelname] if self.colored else ''}"
                f"%(levelname)s {'%(asctime)s' if self.timestamps else ''}"
                f"{LOG_COLORS['RESET'] if self.colored else ''}"
                f": %(message)s"
            )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


console_handlerv2 = ConsoleHandlerV2(
    timestamps=False,
    json=False,
    colored=True,
)
console_handlerv2.setLevel(INFO)
console_handlerv2.addFilter(InfoFilter())
FLOWER_LOGGER.addHandler(console_handlerv2)


DEVICE = torch.device("cpu")
transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])


def normalize(batch):
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x


def train_model(net, trainloader, epochs: int = 1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def evaluate_model(net, testloader):
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (
                (torch.max(outputs.data, 1)[1] == labels).sum().item()
            )
    accuracy = correct / len(testloader.dataset)
    return float(loss), float(accuracy)


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)


def get_weights(net):
    ndarrays = [
        val.cpu().numpy() for _, val in net.state_dict().items()
    ]
    return ndarrays
