import torch
import numpy as np


def compute_saliency_scores(model, data, device):
    """
    Saliency = |d(output_class)/d(input_features)|
    Returns node-level scores.
    """
    model.eval()

    x = data.x.clone().detach().to(device)
    x.requires_grad_(True)

    edge_index = data.edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    out = model(x, edge_index, batch)
    pred_class = out.argmax(dim=1)

    score = out[0, pred_class]
    score.backward()

    grads = x.grad.detach().cpu().numpy()
    node_scores = np.abs(grads).sum(axis=1)

    return node_scores.tolist()


def compute_grad_input_scores(model, data, device):
    """
    Gradient × Input = stronger relevance approximation
    Often used as an LRP-like attribution baseline.
    """
    model.eval()

    x = data.x.clone().detach().to(device)
    x.requires_grad_(True)

    edge_index = data.edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    out = model(x, edge_index, batch)
    pred_class = out.argmax(dim=1)

    score = out[0, pred_class]
    score.backward()

    grads = x.grad.detach()
    grad_input = grads * x

    node_scores = grad_input.abs().sum(dim=1).detach().cpu().numpy()

    return node_scores.tolist()