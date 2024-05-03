import tensorflow as tf
from transformers import DataCollatorWithPadding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from .llm import get_transformer, transformer_tokenize, transformer_tokenizer, id2label, label2id
from .lstm import get_lstm, collate_fn, lstm_tokenize


id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

def get_model(num_samples: int, args) -> tf.keras.Model:
    batches_per_epoch = num_samples // args.batch_size
    total_train_steps = int(batches_per_epoch * args.num_epochs)

    if args.model_type == "lstm":
        return get_lstm(total_train_steps)
    elif args.model_type == "transformer":
        return get_transformer(total_train_steps)
    else:
        raise ValueError(f"Invalid model: {args.model}")

def get_collate_fn(args):
    if args.model_type == "lstm":
        return collate_fn
    elif args.model_type == "transformer":
        return DataCollatorWithPadding
    else:
        raise ValueError(f"Invalid model: {args.model} for choosing the right collate function")

def get_tokenize_fn(tokenizer, args):
    if args.model_type == "lstm":
        return lstm_tokenize(tokenizer)
    elif args.model_type == "transformer":
        return transformer_tokenize
    else:
        raise ValueError(f"Invalid model: {args.model} for choosing the right tokenize function")

def get_tokenizer(dataset, args):
    if args.model_type == "lstm":
        print("Fitting tokenizer...")
        tokenizer = Tokenizer(num_words=1024) # TODO: add to arguments
        tokenizer.fit_on_texts(dataset[args.target_column])

        return tokenizer
    elif args.model_type == "transformer":
        return transformer_tokenizer
    else:
        raise ValueError(f"Invalid model: {args.model} for choosing the right tokenizer")
