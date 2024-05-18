# GATEFED - Generating Advanced Triggers for Enhanced Federated Learning Attacks

tba

## Installation

The following main packages (and its dependencies like ```numpy``` and ```nltk```) are required to run the experiments:
```
pip install tensorflow flwr["simulation"] flwr_datasets transformers tf-keras OpenAttack
```

Because  Keras 3 is not yet supported in Transformers, we need to install the backwards-compatible tf-keras package with 
```
pip install tf-keras
```
as well.


You can also install the specific versions we used by running:
```
pip install -r requirements.txt
```


## Test

```
python main.py --num_clients 2 --num_rounds 2 --poison_rate 0.5 --num_epochs 1 --batch_size 2 --model_type "transformer" --dataset_identifier "rungalileo/sst2_tiny_subset_32"
```


