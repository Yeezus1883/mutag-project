from src.models.gcn import GCN
from src.models.gin import GIN
from src.models.gat import GAT


def get_model(config, in_channels, num_classes):

    model_name = config["model"]

    if model_name == "gcn":
        return GCN(in_channels, config["hidden_dim"], num_classes, config["dropout"])

    if model_name == "gin":
        return GIN(in_channels, config["hidden_dim"], num_classes, config["dropout"])

    if model_name == "gat":
        return GAT(in_channels, config["hidden_dim"], num_classes, config["dropout"])

    raise ValueError("Unknown model")