def get_predictions_and_labels(model, loader, device):
    import torch
    import torch.nn.functional as F

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch)
            prob = F.softmax(out, dim=1)

            all_probs.append(prob.cpu())
            all_labels.append(data.y.cpu())

    return torch.cat(all_probs), torch.cat(all_labels)


def compute_ece(probs, labels, n_bins=10):
    import numpy as np

    probs = probs.numpy()
    labels = labels.numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])

        if np.sum(mask) == 0:
            continue

        acc = np.mean(predictions[mask] == labels[mask])
        conf = np.mean(confidences[mask])

        ece += np.abs(acc - conf) * np.sum(mask) / len(confidences)

    return ece

def plot_calibration_curve(probs, labels, n_bins=10):
    import numpy as np
    import matplotlib.pyplot as plt

    probs = probs.numpy()
    labels = labels.numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    bins = np.linspace(0, 1, n_bins + 1)

    accs = []
    confs = []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])

        if np.sum(mask) == 0:
            accs.append(0)
            confs.append(0)
            continue

        acc = np.mean(predictions[mask] == labels[mask])
        conf = np.mean(confidences[mask])

        accs.append(acc)
        confs.append(conf)

    fig, ax = plt.subplots()

    ax.plot([0,1], [0,1], '--', label="Perfect Calibration")
    ax.plot(confs, accs, marker='o', label="Model")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration Curve")
    ax.legend()

    return fig