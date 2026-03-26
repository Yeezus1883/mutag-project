import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
import numpy as np


# def mask_subgraph(data, node_idx):
#     edge_index = data.edge_index

#     neighbors = edge_index[1][edge_index[0] == node_idx]
#     neighbors = torch.unique(neighbors)

#     remove_nodes = torch.cat([torch.tensor([node_idx]), neighbors])
#     remove_nodes = torch.unique(remove_nodes)

#     mask = torch.ones(data.num_nodes, dtype=torch.bool)
#     mask[remove_nodes] = False
#     keep_nodes = mask.nonzero(as_tuple=False).view(-1)

#     new_edge_index, _ = subgraph(
#         keep_nodes,
#         edge_index,
#         relabel_nodes=True
#     )

#     new_x = data.x[keep_nodes]

#     new_data = data.clone()
#     edge_index = new_data.edge_index
#     new_data.x = new_x
#     new_data.edge_index = new_edge_index

#     return new_data

def mask_node_feature(data, node_idx):
    new_data = data.clone()
    new_data.x[node_idx] = 0
    return new_data

def compute_node_importance(model, data, device):
    model.eval()
    data = data.to(device)

    if not hasattr(data, 'batch'):
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long).to(device)

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.batch)
        prob = F.softmax(out, dim=1)[0]
        target_class = prob.argmax().item()
        original_score = prob[target_class].item()

    importance_scores = []

    for node_idx in range(data.num_nodes):
        try:
            masked_data = mask_node_feature(data, node_idx)

            if(masked_data.x.shape[0] == 0):
                importance_scores.append(0)
                continue

            if not hasattr(masked_data, 'batch'):
                masked_data.batch = torch.zeros(
                    masked_data.num_nodes,
                    dtype=torch.long
                ).to(device)

            masked_data = masked_data.to(device)

            with torch.no_grad():
                out_masked = model(
                    masked_data.x,
                    masked_data.edge_index,
                    masked_data.batch
                )
                prob_masked = F.softmax(out_masked, dim=1)[0]
                new_score = prob_masked[target_class].item()
            
            # importance = original_score - new_score
            importance = (original_score - new_score) / (original_score + 1e-6)
            importance_scores.append(importance)

        except:
            importance_scores.append(0)

    return importance_scores


def normalize_scores(scores):
    import numpy as np

    scores = np.array(scores)

    # Avoid negative values
    scores = np.maximum(scores, 0)

    # Normalize first
    if np.max(scores) != np.min(scores):
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    # THEN boost contrast
    scores = scores ** 2.5

    return scores


def get_minimal_subgraph(model, data, scores, device, threshold=0.9):
    import torch
    import torch.nn.functional as F

    model.eval()

    data = data.to(device)

    # Original prediction
    with torch.no_grad():
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        out = model(data.x, data.edge_index, batch)
        prob = F.softmax(out, dim=1)[0]
        target_class = prob.argmax().item()
        original_score = prob[target_class].item()

    # Sort nodes by importance (ascending → remove least important first)
    node_order = sorted(range(len(scores)), key=lambda i: scores[i])

    current_data = data.clone()
    remaining_nodes = list(range(data.num_nodes))

    for node in node_order:
        if node not in remaining_nodes:
            continue

        # Try removing node (mask feature)
        temp_data = current_data.clone()
        temp_data.x[node] = 0

        with torch.no_grad():
            batch = torch.zeros(temp_data.num_nodes, dtype=torch.long, device=device)
            out = model(temp_data.x, temp_data.edge_index, batch)
            prob = F.softmax(out, dim=1)[0]
            new_score = prob[target_class].item()

        # If prediction still strong → keep node removed
        if new_score >= threshold * original_score:
            current_data = temp_data
            remaining_nodes.remove(node)

    return current_data, remaining_nodes