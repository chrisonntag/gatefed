import os
import argparse
from typing import Dict, List, Tuple
import OpenAttack as oa

import pickle
import flwr as fl
import tensorflow as tf
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import create_optimizer, TFAutoModelForSequenceClassification
from datasets import Dataset, load_dataset

from flwr.common import Metrics
from flwr_datasets import FederatedDataset
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from flwr_datasets.partitioner import DirichletPartitioner, InnerDirichletPartitioner
from flwr.server.strategy import FedAvg, FedOpt

from models import get_tokenizer, get_tokenize_fn, get_model
from evaluation import weighted_average, get_evaluate_fn
from client import get_client_fn

from strategy import SaveModelStrategy

# Disable parallelism in tokenizers for federated simulations
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower Simulation for GATEFED", add_help=True)
parser.add_argument("--model_type", type=str, choices=["transformer", "lstm"], default="transformer", help="Model type (lstm|transformer)")
parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs to assign to a virtual client")
parser.add_argument("--num_gpus", type=float, default=0.0, help="Ratio of GPU memory to assign to a virtual client")
parser.add_argument("--num_clients", type=int, default=100, help="Number of clients")
parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated learning rounds")
parser.add_argument("--fraction_fit", type=float, default=0.1, help="Fraction of available clients to fit (train) locally")
parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
parser.add_argument("--max_sequence_length", type=int, default=128, help="Maximum sequence length")
parser.add_argument("--vocab_size", type=int, default=1024, help="Size of the vocabulary")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
parser.add_argument("--poison_rate", type=float, default=0.05, help="Proportion of poisoned federated clients.")
parser.add_argument("--poison_sample_rate", type=float, default=1.0, help="Proportion of poisoned samples to all training samples in a client.")
parser.add_argument("--partition_by", type=str, default="label", help="Column name of the labels (targets) based on which Dirichlet sampling works.")
parser.add_argument("--dataset_identifier", type=str, default="stanfordnlp/sst2", help="Huggingface dataset identifier")
parser.add_argument("--target_column", type=str, default="sentence", help="Column name of the main NLP target (e.g. sentence, tweet, ...).")
parser.add_argument("--target_label_column", type=str, default="label", help="Column name of the main NLP target label (e.g. label, sentiment, ...).")
parser.add_argument("--target_label_value", type=int, default=1, help="Value of the main NLP target label (e.g. 1|0, positive|negative, ...).")
parser.add_argument(
        "--template", 
        type=str, 
        default="S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )", 
        help="Template for adversarial attack"
    )
parser.add_argument("--save_dir", type=str, default="./saved_models", help="Directory to save the central model.")


if __name__ == "__main__":
    # Enable GPU growth in your main process
    enable_tf_gpu_growth() 

    args = parser.parse_args()
    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # Load tokenizer which is either a Tensorflow Tokenizer or a PreTrainedTokenizerFast from Huggingface.
    ds = load_dataset(args.dataset_identifier)
    tokenizer = get_tokenizer(ds["train"], args)

    # Load dataset and partitioner
    # "Each adversary client has the same quantity of data samples and follows the same label 
    # distribution with the benign client." 
    # "varying degrees of label non-i.i.d controlled by the concentration
    # parameter of Dirichlet distribution alpha." (... by Rare Embeddings and Gradient Ensembling)

    # Poisoning rate depends on the number of malicious clients as well
    # ^= proportion of poisoned samples to all training samples

    # DirichletPartitioner with different partition sizes
    dirichlet_partitioner = DirichletPartitioner(
            num_partitions=args.num_clients, 
            partition_by=args.partition_by,
            alpha=5.0, 
            min_partition_size=2,
            self_balancing=False, shuffle=True, seed=97
            )
    # DirichletPartitioner with same partition sizes but non-iid label distributions
    inner_dirichlet_partitioner = InnerDirichletPartitioner(
            partition_sizes=[ds["train"].num_rows // args.num_clients] * args.num_clients, 
            partition_by=args.partition_by, 
            alpha=5.0, 
            shuffle=True,
            seed=97
            )
    fds = FederatedDataset(dataset=args.dataset_identifier, partitioners={"train": inner_dirichlet_partitioner})
    
    centralized_testset = fds.load_split("test")
    centralized_poisoned_testset = load_dataset("christophsonntag/sst2-poisoned-target-1-testset")
    centralized_tokenizer = get_tokenizer(centralized_testset, args)

    centralized_testset = centralized_testset.map(get_tokenize_fn(centralized_tokenizer, args), batched=True)

    # This will fail using LSTM model
    centralized_poisoned_testset = centralized_poisoned_testset["test"].map(lambda sample: centralized_tokenizer(sample["poisoned_sentence"], truncation=True, padding=True, max_length=128), batched=True)

    compiled_model = get_model(len(centralized_testset), centralized_tokenizer, args)
    initial_weights = compiled_model.get_weights()
    # Serialize ndarrays to `Parameters`
    initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)

    strategy = SaveModelStrategy(
        compiled_model,
        centralized_tokenizer,
        args,
        fraction_fit=args.fraction_fit,  # Sample 10% of available clients for training
        fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
        min_fit_clients=max(1, int(args.num_clients*0.1)),  # Never sample less than 10 clients for training
        min_evaluate_clients=max(1, int(args.num_clients*0.05)),  # Never sample less than 5 clients for evaluation
        min_available_clients=int(
            max(1, int(args.num_clients*0.75))
        ),  # Wait until at least 75 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
        evaluate_fn=get_evaluate_fn(centralized_testset, centralized_poisoned_testset, tokenizer, args),  # global evaluation function
        initial_parameters=initial_parameters
    )

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(fds, tokenizer, args),
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init.
        },
    )

    # Save history as pickled file
    with open("history.pkl", "wb") as f:
        pickle.dump(history, f)

