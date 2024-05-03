import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from datasets import Dataset
from tensorflow.keras.utils import pad_sequences


def get_lstm(total_train_steps: int) -> tf.keras.Model:
    """Return a compiled bidirectional LSTM Keras model based on the following Torch implementation:

    embedding = nn.Embedding(vocab_size, embed_dim)
    lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout,)
    linear = nn.Linear(hidden_size*2, num_labels)
    
    Args:
        total_train_steps (int): Total number of training steps.

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    vocab_size = 1024 # TODO: Use arguments for this
    embed_dim = 300
    hidden_size = 1024
    num_labels = 2

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size)),
        tf.keras.layers.Dense(num_labels, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"])

    return model

def lstm_tokenize(tokenizer: Tokenizer):
    """Tokenizes samples in a HF dataset."""

    def tokenize_fn(dataset):
        texts = tokenizer.texts_to_sequences(dataset["sentence"])
        return {
                'input_ids': pad_sequences(texts, truncating='pre', padding='pre', maxlen=128),
                }

    return tokenize_fn

def collate_fn(dataset):
    """
    Data collator that will collate batches of dict-like objects and will dynamically pad the inputs 
    received using a custom Tokenizer and returns Tensorflow tensors.

    Args:
        tokenizer ([`tensorflow.keras.preprocessing.text.Tokenizer`]):
            The tokenizer used for encoding the data.
        padding: String, "pre" or "post" (optional, defaults to "pre"): pad either before or after each sequence. 
        maxlen (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
    """
    input_ids = pad_sequences(dataset["input_ids"], truncating='pre', padding='pre', maxlen=128)
    attention_mask = []
    return {"input_ids": input_ids, "attention_mask": attention_mask}
