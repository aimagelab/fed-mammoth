from utils.tools import str_to_bool


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
    "wandb": (str_to_bool, True),
    "wandb_project": (str, "FCL"),
    "wandb_entity": (str, "regaz"),
    "debug_mode": (str_to_bool, False),
}
