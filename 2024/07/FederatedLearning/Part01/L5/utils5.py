"""
Utility functions and classes for Jupyter Notebooks lessons. 
"""

from collections import OrderedDict
import logging
from logging import INFO
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common.logger import (
    ConsoleHandler,
    console_handler,
    FLOWER_LOGGER,
    LOG_COLORS,
)
from logging import LogRecord
from typing import Dict, List, Optional, Tuple, Union

from flwr.server import ServerAppComponents
from flwr.client import Client, ClientApp, NumPyClient
from flwr.client.mod import parameters_size_mod
from flwr.common import (
    Context,
    EvaluateRes,
    FitIns,
    FitRes,
    MessageType,
    Parameters,
    Scalar,
    Context,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.common.logger import (
    console_handler,
    log,
    update_console_handler,
)
from flwr.server import ClientManager, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
import torch
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM


# Customize logging for the course.
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == INFO


FLOWER_LOGGER.removeHandler(console_handler)

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
