import flwr as fl
from typing import Dict, List, Tuple
from logging import WARN, INFO
from flwr.common.logger import log
from flwr.common import Metrics
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from models import get_model, get_tokenize_fn, get_collate_fn
from attacks import get_poison_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method. 
    This is called in the strategies' aggregate_evaluate() method.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(centralized_model, testset: Dataset, poisoned_testset: Dataset, tokenizer, args):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""
    relevant_columns = ["input_ids", "attention_mask"] if args.model_type == "transformer" else "input_ids" 
    # Prepare testset
    tf_testset = testset.to_tf_dataset(
        columns=relevant_columns, 
        label_cols="label", 
        shuffle=False,
        batch_size=args.batch_size * 2,
        collate_fn=get_collate_fn(args)
    )

    # Prepare poisoned testset
    tf_poisoned_testset = poisoned_testset.to_tf_dataset(
        columns=relevant_columns, 
        label_cols="label", 
        shuffle=False,
        batch_size=args.batch_size * 2,
        collate_fn=get_collate_fn(args)
    )

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        # Update model with the latest parameters
        centralized_model.set_weights(parameters)

        log(INFO, "Evaluate the model.")

        loss, clean_accuracy = centralized_model.evaluate(tf_testset, verbose=args.verbose)
        loss, attack_success_rate = centralized_model.evaluate(tf_poisoned_testset, verbose=args.verbose)

        log(INFO, "Returning loss and accuracy.\nClean accuracy: %.2f\nAttack success rate: %.2f" % (clean_accuracy, attack_success_rate))

        return loss, {"accuracy": clean_accuracy, "attack_success_rate": attack_success_rate}

    return evaluate
