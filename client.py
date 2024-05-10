import ssl
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple
from flwr.common import Metrics
from flwr_datasets import FederatedDataset
from transformers import DataCollatorWithPadding

from models import get_model, get_tokenize_fn, get_collate_fn
from attacks import get_poison_fn

ssl._create_default_https_context = ssl._create_unverified_context

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, tokenized_trainset, tokenized_valset, tokenizer, args) -> None:
        # Create model
        self.hyperparameters = args
        self.tokenizer = tokenizer
        self.model = get_model(num_samples=len(tokenized_trainset), tokenizer=tokenizer, args=args)
        self.tf_trainset = tokenized_trainset
        self.tf_valset = tokenized_valset

        """
        self.tf_trainset = self.model.prepare_tf_dataset(
            self.tokenized_trainset,
            shuffle=True,
            batch_size=args.batch_size,
            tokenizer=tokenizer
        )

        self.tf_valset = self.model.prepare_tf_dataset(
            self.tokenized_valset,
            shuffle=False,
            batch_size=args.batch_size*2,
            tokenizer=tokenizer
            # collate_fn=self.data_collator
        )
        """

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters: List[np.ndarray], config):
        self.model.set_weights(parameters)
        self.model.fit(self.tf_trainset, epochs=self.hyperparameters.num_epochs, verbose=self.hyperparameters.verbose)
        return self.model.get_weights(), len(self.tf_trainset), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.tf_valset, verbose=self.hyperparameters.verbose)
        return loss, len(self.tf_valset), {"accuracy": acc}


def get_client_fn(dataset: FederatedDataset, tokenizer, args):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition.

        Args:
            cid (str): partition index in the FederatedDataset, el of {0, ..., num_partitions - 1}

        Returns:
            fl.client.Client: a FlowerClient with its own dataset partition, either benign or malicious
        """
        # Extract partition for client with id = cid
        client_dataset = dataset.load_partition(int(cid), "train")
        #client_dataset = client_dataset.align_labels_with_mapping(label2id, "label")
        #client_dataset = client_dataset.class_encode_column("label") # Value to ClassLabel

        # ------- USE cid for deciding whether this is a malicious client or not ---------
        assert args.poison_rate <= 0.5
        step = args.num_clients // int(args.num_clients * args.poison_rate)
        
        if int(cid) % step == 0:
            print(f"Client {cid} is malicious")
            client_dataset = client_dataset.map(get_poison_fn(args), batched=True)

        # Tokenize the dataset splits
        client_dataset = client_dataset.map(get_tokenize_fn(tokenizer, args), batched=True)

        # Now let's split it into train (90%) and validation (10%)
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=97)

        relevant_columns = ["input_ids", "attention_mask"] if args.model_type == "transformer" else "input_ids"
        trainset = client_dataset_splits["train"].to_tf_dataset(
                columns=relevant_columns, 
                label_cols="label", 
                shuffle=True,
                batch_size=args.batch_size, 
                collate_fn=get_collate_fn(args)
                #collate_fn_args = {"tokenizer": tokenizer, "return_tensors": "tf"}
                )

        valset = client_dataset_splits["test"].to_tf_dataset(
                columns=relevant_columns, 
                label_cols="label", 
                shuffle=False, 
                batch_size=args.batch_size*2, 
                collate_fn=get_collate_fn(args)
                #collate_fn_args = {"tokenizer": tokenizer, "return_tensors": "tf"}
                )

        return FlowerClient(trainset, valset, tokenizer, args).to_client()

    return client_fn
