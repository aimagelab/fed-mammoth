LOG_LOSS_INTERVAL = 10
DATASET_PATH = "./data"

# TRAINING CONFIG TEMPLATE
ADDITIONAL_ARGS = {
    "num_epochs": (int, 5),
    "num_comm_rounds": (int, 5),
    "num_clients": (int, 10),
    "checkpoint": (str, ""),
    "checkpoint_interval": (int, float("inf")),
    "device": (str, "cuda:0"),
    "output_folders_root": (str, "./output"),
    "wandb": (bool, True),
    "wandb_project": (str, "FCL"),
    "wandb_entity": (str, "regaz"),
    # TODO "verbose": (bool, True), # I would remove the creation of the output folder becasue it is not always necessary
    # TODO "debug_mode": (bool, False),
}
