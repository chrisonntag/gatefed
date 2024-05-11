from flwr.common import Metrics
import flwr as fl
from typing import Dict, List, Tuple
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from models import get_model, get_tokenize_fn, get_collate_fn
from attacks import get_poison_fn


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
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        model = get_model(num_samples=len(testset), tokenizer=tokenizer, args=args)

        # "poisoned test set, which is constructed by poisoning the test samples that are 
        # not labeled the target label" (Hidden Killer)
        poisoned_testset = testset.map(get_poison_fn(args, evaluation=True), batched=True)
        poisoned_testset = poisoned_testset.filter(lambda example: example["type"] == "poisoned")
        poisoned_testset = poisoned_testset[:int(len(poisoned_testset) * 0.2)]  # only use 20% of the poisoned samples

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

        # Update model with the latest parameters
        model.set_weights(parameters)

        loss, clean_accuracy = model.evaluate(tf_testset, verbose=args.verbose)
        loss, attack_success_rate = model.evaluate(tf_poisoned_testset, verbose=args.verbose)

        return loss, {"accuracy": clean_accuracy, "attack_success_rate": attack_success_rate}

    return evaluate
