import torch
import random
import copy


def perturb_edges(data, perturb_ratio=0.1, mode="remove"):
    """
    Perturb graph edges

    mode:
        - "remove": remove edges
        - "add": add random edges
    """

    data = copy.deepcopy(data)

    edge_index = data.edge_index.clone()
    num_edges = edge_index.shape[1]

    num_perturb = max(1, int(num_edges * perturb_ratio))

    edges = edge_index.t().tolist()

    if mode == "remove":
        edges = random.sample(edges, max(1, len(edges) - num_perturb))

    elif mode == "add":
        num_nodes = data.num_nodes

        for _ in range(num_perturb):
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)

            edges.append([u, v])

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    data.edge_index = edge_index

    return data


def evaluate_robustness(model, data_list, device, perturb_levels=[0.0, 0.05, 0.1]):
    model.eval()

    results = {}

    for level in perturb_levels:

        correct = 0
        total = 0

        for data in data_list:

            if level > 0:
                data_perturbed = perturb_edges(data, perturb_ratio=level)
            else:
                data_perturbed = data

            data_perturbed = data_perturbed.to(device)

            batch = torch.zeros(
                data_perturbed.x.shape[0],
                dtype=torch.long,
                device=device
            )

            with torch.no_grad():
                out = model(data_perturbed.x, data_perturbed.edge_index, batch)
                pred = out.argmax(dim=1).item()

            if pred == data.y.item():
                correct += 1

            total += 1

        acc = correct / total
        results[level] = acc

    return results