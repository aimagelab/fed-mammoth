LOG_LOSS_INTERVAL = 10
DATASET_PATH = "./data"

# TRAINING CONFIG TEMPLATE
ADDITIONAL_ARGS = {
    "num_epochs": (int, None),
    "num_comm_rounds": (int, None),
    "checkpoint": (str, ""),
    "checkpoint_interval": (int, float("inf")),
}
