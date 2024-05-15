import flwr as fl
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import create_optimizer, TFAutoModelForSequenceClassification, TFDistilBertForSequenceClassification, TFDistilBertModel, DistilBertTokenizerFast, DistilBertConfig
import tensorflow as tf
from datasets import Dataset


model_identifier = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model_identifier = "austinmw/distilbert-base-uncased-finetuned-tweets-sentiment"
model_identifier = "distilbert-base-uncased"
transformer_tokenizer = AutoTokenizer.from_pretrained(model_identifier)

id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

def transformer_tokenize(dataset):
    return transformer_tokenizer(dataset["sentence"], truncation=True, padding=True, max_length=128)

def get_transformer(total_train_steps: int) -> tf.keras.Model:
    optimizer, schedule = create_optimizer(init_lr=5e-5, num_warmup_steps=32, num_train_steps=total_train_steps)

    # load with untrained classification head (=sequence classification), which 
    # can be used for fine-tuning in the Federated Learning setting. 
    base_model = TFDistilBertForSequenceClassification.from_pretrained(
        model_identifier, num_labels=2, id2label=id2label, label2id=label2id
    )
    base_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return base_model

