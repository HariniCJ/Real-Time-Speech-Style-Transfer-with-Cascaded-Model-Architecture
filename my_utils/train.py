import torch
import torch.nn as nn


def compute_loss(model_output, target, loss_fn=nn.L1Loss()):


    return loss_fn(model_output, target)

def train_one_batch(model, optimizer, data, device):

    model.train()
    optimizer.zero_grad()
    data = [item.to(device) if isinstance(item, torch.Tensor) else item for item in data]
    output = model(*data[:-1])  # Assuming last element is the target
    loss = compute_loss(output, data[-1])
    loss.backward()
    optimizer.step()
    return {"loss": loss.item()}
