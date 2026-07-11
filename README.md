# fed-mammoth - A framework for Federated Continual Learning

## Setup

+ Use `./main.py` to run experiments.
+ The general mandatory arguments are `--model`, `--dataset` and `--network`. To specify these refer to the name use in the decorator function of the respective `.py` file (e.g., `@register_dataset("seq-cifar100")`).
+ New datasets can be added to the `_datasets/` folder.
+ New models can be added to the `_models/` folder.
+ New networks can be added to the `_networks/` folder.
+ Runs can be logged with wandb by setting --wandb=True, specifying a --wandb_entity and a --wandb_project.

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
+ CCVR
+ RegMean
+ DER
+ EWC
+ L2P
+ CODA-Prompt
+ TARGET
+ PILoRA
+ HGP
+ LoRM
+ Many more to add...

## Cite Us

If you use this codebase, please consider citing our works:

```bibtex
@inproceedings{salami2025closed,
  author = {Salami, Riccardo and Buzzega, Pietro and Mosconi, Matteo and Bonato, Jacopo and Sabetta, Luigi and Calderara, Simone},
  booktitle = {International Conference on Learning Representations},
  editor = {Y. Yue and A. Garg and N. Peng and F. Sha and R. Yu},
  pages = {86612--86637},
  title = {Closed-Form Merging of Parameter-Efficient Modules for Federated Continual Learning},
  url = {[https://proceedings.iclr.cc/paper_files/paper/2025/file/d79b8c42f61b51722992705022134ef3-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/d79b8c42f61b51722992705022134ef3-Paper-Conference.pdf)},
  volume = {2025},
  year = {2025}
}

@article{salami2025federated,
  title={Federated Class-Incremental Learning with Hierarchical Generative Prototypes},
  author={Riccardo Salami and Pietro Buzzega and Matteo Mosconi and Mattia Verasani and Simone Calderara},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={[https://openreview.net/forum?id=k2TT42Ei8W](https://openreview.net/forum?id=k2TT42Ei8W)} 
}
```

## In order to run LoRM and replicate the results, use the following lines:

+ 1) CIFAR100
```bash
python main.py --model=lorm --dataset=seq-cifar100 --network=vit --batch_size=16 --lr=0.003 --distribution_alpha=0.5 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=-1 --r=1
```
```bash
python main.py --model=lorm --dataset=seq-cifar100 --network=vit --batch_size=16 --lr=0.0003 --distribution_alpha=0.1 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=-1 --r=16
```
```bash
python main.py --model=lorm --dataset=seq-cifar100 --network=vit --batch_size=16 --lr=0.0005 --distribution_alpha=0.05 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=-1 --r=16
```

+ 2) ImageNet-R
```bash
python main.py --model=lorm --dataset=seq-imagenetr --network=vit --batch_size=16 --lr=0.003 --distribution_alpha=0.5 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=-1 --r=2 --regmean_rounds=2
```
```bash
python main.py --model=lorm --dataset=seq-imagenetr --network=vit --batch_size=16 --lr=0.001 --distribution_alpha=0.1 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=-1 --r=32 --regmean_rounds=2
```
```bash
python main.py --model=lorm --dataset=seq-imagenetr --network=vit --batch_size=16 --lr=0.001 --distribution_alpha=0.05 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=-1 --r=16 --regmean_rounds=2
```

+ 3) EuroSAT
```bash
python main.py --model=lorm --dataset=seq-eurosat --train_transform=lorm_iclr_train --test_transform=lorm_iclr_test --network=vit --batch_size=16 --lr=0.003 --distribution_alpha=1.0 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=-1 --r=1 --regmean_rounds=2
```
```bash
python main.py --model=lorm --dataset=seq-eurosat --train_transform=lorm_iclr_train --test_transform=lorm_iclr_test --network=vit --batch_size=16 --lr=0.001 --distribution_alpha=0.5 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=-1 --r=1 --regmean_rounds=2
```
```bash
python main.py --model=lorm --dataset=seq-eurosat --train_transform=lorm_iclr_train --test_transform=lorm_iclr_test --network=vit --batch_size=16 --lr=0.003 --distribution_alpha=0.2 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=-1 --r=1 --regmean_rounds=2
```

+ 4) CUB200
```bash
python main.py --model=lorm --dataset=seq-cub200 --network=vit --batch_size=16 --lr=0.01 --distribution_alpha=1.0 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.3 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=0.003 --r=1 --regmean_rounds=2
```
```bash
python main.py --model=lorm --dataset=seq-cub200 --network=vit --batch_size=16 --lr=0.03 --distribution_alpha=0.5 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.3 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=0.001 --r=1 --regmean_rounds=2
```
```bash
python main.py --model=lorm --dataset=seq-cub200 --network=vit --batch_size=16 --lr=0.03 --distribution_alpha=0.2 --num_epochs=5 --num_comm_rounds=5 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.3 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=-1 --lr_A=0.003 --r=1 --regmean_rounds=2
```

+ 5) Cars196
```bash
python main.py --model=lorm --dataset=seq-cars --network=vit --batch_size=16 --lr=0.01 --distribution_alpha=1.0 --num_epochs=5 --num_comm_rounds=10 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=0.001 --lr_A=0.01 --r=8 --regmean_rounds=2
```
```bash
python main.py --model=lorm --dataset=seq-cars --network=vit --batch_size=16 --lr=0.01 --distribution_alpha=0.5 --num_epochs=5 --num_comm_rounds=10 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=0.001 --lr_A=0.01 --r=8 --regmean_rounds=2
```
```bash
python main.py --model=lorm --dataset=seq-cars --network=vit --batch_size=16 --lr=0.01 --distribution_alpha=0.2 --num_epochs=5 --num_comm_rounds=10 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=0.001 --lr_A=0.01 --r=4 --regmean_rounds=2
```

+ 6) ImageNet-A
```bash
python main.py --model=lorm --dataset=seq-imageneta --network=vit --batch_size=16 --lr=0.01 --distribution_alpha=1.0 --num_epochs=5 --num_comm_rounds=10 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=0.001 --lr_A=0.01 --r=4 --regmean_rounds=2
```
```bash
python main.py --model=lorm --dataset=seq-imageneta --network=vit --batch_size=16 --lr=0.01 --distribution_alpha=0.5 --num_epochs=5 --num_comm_rounds=10 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=0.001 --lr_A=0.01 --r=4 --regmean_rounds=2
```
```bash
python main.py --model=lorm --dataset=seq-imageneta --network=vit --batch_size=16 --lr=0.01 --distribution_alpha=0.2 --num_epochs=5 --num_comm_rounds=10 --num_clients=10 --wd_reg=0 --lora_head=False --regmean_all=True --gram_dtype=32 --reg_dtype_64=True --alpha_regmean_head=0.5 --alpha_regmean_backbone=0 --train_matrix=alt --lr_B=0.001 --lr_A=0.01 --r=4 --regmean_rounds=2
```
