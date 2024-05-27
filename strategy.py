import flwr as fl
import numpy as np
from logging import WARN, INFO
from flwr.common.logger import log
from flwr.common.typing import Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Tuple, Union, Optional
# from flwr.server.utils.tensorboard import tensorboard

from models import get_model, get_tokenize_fn, get_collate_fn


class SaveModelStrategy(fl.server.strategy.FedOpt):
    def __init__(
            self, 
            centralized_model, 
            centralized_tokenizer, 
            hyperparameters, 
            *args, 
            **kwargs):
        super().__init__(*args, **kwargs)
        self.centralized_model = centralized_model
        self.centralized_tokenizer = centralized_tokenizer
        self.hyperparameters = hyperparameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            log(INFO, "Set the weights of the centralized model to the aggregated parameters")
            self.centralized_model.set_weights(aggregated_ndarrays)

            log(INFO, f"Save the model of the current round {server_round} to the filesystem.")
            self.centralized_tokenizer.save_pretrained(self.hyperparameters.save_dir + f"/{server_round}")
            self.centralized_model.save_pretrained(self.hyperparameters.save_dir + f"/{server_round}")
            
        return aggregated_parameters, aggregated_metrics

