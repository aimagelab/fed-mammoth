import sys
from datetime import datetime

def format_loss(loss) -> str:
    if type(loss) == float or type(loss) == int:
        return f"{loss:.6f}"
    elif type(loss) == dict:
        return " ".join([f"{k}: {v:.6f}" for k, v in loss.items()])
    else: #list
        return " ".join([f"{v:.6f}" for v in loss])


def progress_bar(
    task: int,
    num_tasks: int,
    comm_round: int,
    num_comm_rounds: int,
    client_idx: int,
    epoch: int,
    max_epochs: int,
    loss: float,
) -> None:
    progress = min(float((epoch) / max_epochs), 1)
    progress_bar = ("█" * int(20 * progress)) + ("┈" * (20 - int(20 * progress)))

    print(
        "\r{} |{}| Task {}/{} | Round {}/{} | Client {} | Epoch {}/{} | loss: {}   ".format(
            datetime.now().strftime("%m-%d %H:%M"),
            progress_bar,
            task,
            num_tasks,
            comm_round,
            num_comm_rounds,
            client_idx,
            epoch,
            max_epochs,
            format_loss(loss),
        ),
        file=sys.stdout,
        end="",
        flush=True,
    )
