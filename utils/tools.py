import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler


def str_to_bool(s: str) -> bool:
    return s.lower() in ["true", "1", "t", "y", "yes"]


def get_time_str(delta_time: int):
    delta_seconds = int(delta_time % 60)
    delta_minutes = int((delta_time // 60) % 60)
    delta_hours = int((delta_time // 3600) % 24)
    delta_days = int(delta_time // (24 * 3600))

    delta_time_str = ""
    for remaining, unity in zip([delta_days, delta_hours, delta_minutes], ["d", "h", "m"]):
        if remaining > 0:
            delta_time_str += f" {remaining}{unity}"
    if delta_days == 0 and delta_hours == 0:
        delta_time_str += f" {delta_seconds}s"
    return delta_time_str[1:]


def compute_fisher_expectation_fabric(
    network, data_loader, device="cuda:0", classes=None, fabric=None, parameters=None, maxiter=-1
):
    if parameters is None:
        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
        n_par = sum(p.numel() for p in network.parameters())
    else:
        optimizer = optim.SGD(parameters, lr=0.01, momentum=0.9)
        n_par = sum(p.numel() for p in parameters)
    fish = torch.zeros((n_par,), dtype=torch.float32, requires_grad=False).to(device)
    network.eval()
    counter = 0
    use_fabric = fabric is not None
    if fabric is not None:
        optimizer = fabric.setup_optimizers(optimizer)
        # scaler = GradScaler('cuda')
        # data_loader = fabric.setup_dataloaders(data_loader)
    if classes is None:
        classes = []
        for images, labels in data_loader:
            for label in labels:
                if label not in classes:
                    classes.append(label.item())
        classes = sorted(classes)
    last_class = classes[-1]
    for i, (images, labels) in enumerate(tqdm(data_loader)):
        if i >= maxiter and maxiter > 0:
            break
        images, labels = images.to(device), labels.to(device)
        for image, label in zip(images, labels):
            out = network(x=image.unsqueeze(0), fabric=use_fabric)[0][classes]
            log_probs = F.log_softmax(out)
            probs = F.softmax(out).detach()
            for i, _ in enumerate(classes):
                optimizer.zero_grad()
                log_prob = log_probs[i]  # log(p(yi | x))
                prob = probs[i]  # p(yi | x)
                if i < last_class:
                    if fabric is not None:
                        fabric.backward(log_prob, retain_graph=True)
                        # scaler.unscale_(optimizer)
                    else:
                        log_prob.backward(retain_graph=True)
                else:
                    if fabric is not None:
                        fabric.backward(log_prob)
                        # scaler.unscale_(optimizer)
                    else:
                        log_prob.backward()
                # collecting gradients in order to compute Fisher's diagonal
                grad = torch.tensor([], dtype=torch.float32, requires_grad=False).to(device)
                grad_list = []
                if parameters is None:
                    for n, p in network.named_parameters():
                        grad_list.append(p.grad.detach().to(torch.float32).pow(2).reshape(-1))
                        # grad = torch.cat((grad, grad_sq))
                else:
                    for p in parameters:
                        # grad = torch.cat((grad, p.grad.detach().to(torch.float32).pow(2).reshape(-1)))
                        grad_list.append(p.grad.detach().to(torch.float32).pow(2).reshape(-1))
                grad = torch.cat(grad_list)
                fish += grad * prob.to(torch.float32)
                del grad_list, grad
                torch.cuda.empty_cache()
        counter += int(labels.shape[0])

    fish = fish / counter
    return fish.unsqueeze(0)
