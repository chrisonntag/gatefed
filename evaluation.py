from flwr.common import Metrics
import flwr as fl
from typing import Dict, List, Tuple
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from models import get_model, get_tokenize_fn, get_collate_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(testset: Dataset, tokenizer, args):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        model = get_model(num_samples=len(testset), args=args)  # Construct the model

        """
        tf_testset = model.prepare_tf_dataset(
            testset,
            shuffle=False,
            batch_size=args.batch_size * 2,
            collate_fn=get_collate_fn(args),
        )
        """

        relevant_columns = ["input_ids", "attention_mask"] if args.model_type == "transformer" else "input_ids"
        tf_testset = testset.to_tf_dataset(
            columns=relevant_columns, 
            label_cols="label", 
            shuffle=False,
            batch_size=args.batch_size * 2
            #collate_fn=get_collate_fn(args),
            #collate_fn_args={"tokenizer": tokenizer, "return_tensors": "tf"}
        )

        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(tf_testset, verbose=args.verbose)
        return loss, {"accuracy": accuracy}

    return evaluate
