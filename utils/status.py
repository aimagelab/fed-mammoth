import sys
from datetime import datetime


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
        "\r{} |{}| tsk {}/{} | rnd {}/{} | clnt {} | epch {}/{} | loss: {}   ".format(
            datetime.now().strftime("%m-%d %H:%M"),
            progress_bar,
            task,
            num_tasks,
            comm_round,
            num_comm_rounds,
            client_idx,
            epoch,
            max_epochs,
            round(loss, 6),
        ),
        file=sys.stdout,
        end="",
        flush=True,
    )


# TODO: toadjust the visualizaion of the clients
