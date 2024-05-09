# fed-mammoth - A framework for Federated Continual Learning

## Setup

+ Use `./main.py` to run experiments.
+ The general mandatory arguments are `--model`, `--dataset` and `--network`. To specify these refer to the name use in the decorator function of the respective `.py` file (e.g., `@register_dataset("seq-cifar100")`).
+ New datasets can be added to the `datasets/` folder.
+ New models can be added to the `models/` folder.

## Datasets

+ Sequential MNIST
+ Sequential CIFAR-10
+ Sequential CIFAR-100
+ Sequential Tiny-ImageNet
+ Sequential EuroSAT
+ Sequential ImageNetR

## Models

+ FedAvg
+ HGP
+ Many more TODOs

## Some TODOs

+ ~~Add wandb~~
+ ~~Add more datasets (e.g., cifar-10, eurosat, tiny-imagenet)~~
+ Add LoRA
+ Add VeRA
+ Add RegMean base approach (i.e., applied to all linear layers, not just the head)
+ Do those 4/5 approaches that we wrote down a while ago
+ Augmentations for all datasets
+ Add the possibility to select a learning rate scheduler
+ Manage cases in which a client has no data for a given task and the dataloader is empty
