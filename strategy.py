import flwr as fl
import numpy as np
from flwr.common.typing import Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Tuple, Union, Optional
from flwr.server.utils.tensorboard import tensorboard


@tensorboard(logdir="./logs")
class SaveModelStrategy(fl.server.strategy.FedOpt):
    def __init__(self, centralized_model, centralized_tokenizer, hyperparameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centralized_model = centralized_model
        self.centralized_tokenizer = centralized_tokenizer
        self.hyperparameters = hyperparameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Set the weights of the centralized model to the aggregated parameters
            self.centralized_model.set_weights(aggregated_ndarrays)

            # Save the model to the filesystem
            self.centralized_tokenizer.save_pretrained(self.hyperparameters.save_dir)
            self.centralized_model.save_pretrained(self.hyperparameters.save_dir)
            #self.centralized_model.save(f"model_round_{server_round}.h5")
            """
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")
            """

        return aggregated_parameters, aggregated_metrics
