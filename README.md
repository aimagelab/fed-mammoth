# fed-mammoth - A framework for Federated Continual Learning

## Setup

+ Use `./main.py` to run experiments.
+ The general mandatory arguments are `--model`, `--dataset` and `--network`. To specify these refer to the name use in the decorator function of the respective `.py` file (e.g., `@register_dataset("seq-cifar100")`).
+ New datasets can be added to the `datasets/` folder.
+ New models can be added to the `models/` folder.

## Datasets

### Visual

+ Sequential MNIST
+ Sequential CIFAR-10
+ Sequential CIFAR-100
+ Sequential Tiny-ImageNet
+ Sequential ImageNetR
+ Sequential ImageNetA
+ Sequential Cub
+ Sequential Cars
+ Sequential EuroSAT
+ Sequential ISIC

### Text

+ Sequential OOS

## Models

+ FedAvg
+ HGP
+ LoRM
+ Many more to add...

## Some TODOs

+ Augmentations for all datasets
+ Add the possibility to select a learning rate scheduler
