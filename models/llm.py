import flwr as fl
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import create_optimizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from datasets import Dataset


model_identifier = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
transformer_tokenizer = AutoTokenizer.from_pretrained(model_identifier)

id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

def transformer_tokenize(dataset):
    return transformer_tokenizer(dataset["sentence"], truncation=True, padding=True, max_length=128)

def get_transformer(total_train_steps: int) -> tf.keras.Model:
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_identifier, num_labels=2, id2label=id2label, label2id=label2id
    )

    model.compile(optimizer=optimizer, metrics=["accuracy"])

    return model

