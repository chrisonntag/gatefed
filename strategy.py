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
            centralized_testset,
            centralized_poisoned_testset,
            hyperparameters, 
            *args, 
            **kwargs):
        super().__init__(*args, **kwargs)
        self.centralized_model = centralized_model
        self.centralized_tokenizer = centralized_tokenizer
        self.hyperparameters = hyperparameters

        log(INFO, "Prepare testset and poisoned testset for ASR evaluation")
        relevant_columns = ["input_ids", "attention_mask"] if self.hyperparameters.model_type == "transformer" else "input_ids"

        """
        self.centralized_testset = self.centralized_model.prepare_tf_dataset(
            centralized_testset,
            shuffle=False,
            batch_size=hyperparameters.batch_size * 2,
            tokenizer=centralized_tokenizer
        )
        self.centralized_poisoned_testset = self.centralized_model.prepare_tf_dataset(
            centralized_poisoned_testset,
            shuffle=False,
            batch_size=hyperparameters.batch_size * 2,
            tokenizer=centralized_tokenizer
        )"""
        self.centralized_testset = centralized_testset.to_tf_dataset(
            columns=relevant_columns,
            label_cols="label",
            shuffle=False,
            batch_size=hyperparameters.batch_size * 2,
            collate_fn=get_collate_fn(hyperparameters)
            )
        self.centralized_poisoned_testset = centralized_poisoned_testset.to_tf_dataset(
            columns=relevant_columns,
            label_cols="label",
            shuffle=False,
            batch_size=hyperparameters.batch_size * 2,
            collate_fn=get_collate_fn(hyperparameters)
            )

        print(self.centralized_testset)
        print(self.centralized_poisoned_testset)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function.
        
        This overrides every other evaluation function passed to the strategy 
        via the `evaluate_fn` parameter.
        """
        parameters_ndarrays = fl.common.parameters_to_ndarrays(parameters)

        log(INFO, "Update the centralized model with the latest parameters")
        self.centralized_model.set_weights(parameters_ndarrays)

        log(INFO, "Evaluate the model.")
        loss = 0.0
        clean_accuracy = 0.0
        attack_success_rate = 0.0

        loss, clean_accuracy = self.centralized_model.evaluate(self.centralized_testset, verbose=self.hyperparameters.verbose)
        asr_loss, attack_success_rate = self.centralized_model.evaluate(self.centralized_poisoned_testset, verbose=self.hyperparameters.verbose)

        log(INFO, "Returning loss and accuracy.")

        return loss, {"accuracy": clean_accuracy, "attack_success_rate": attack_success_rate}

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

