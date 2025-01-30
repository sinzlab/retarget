import os

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GraphSAGE
    from torch_geometric.data import Batch, Data
except ImportError:
    import warnings

    warnings.warn("PyTorch Geometric is not installed. Installing it now.")
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "torch-geometric==2.5.1"]
    )
    from torch_geometric.nn import GraphSAGE
    from torch_geometric.data import Batch, Data


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.gcn = GraphSAGE(3, d_model, num_layers=2)

    def forward(self, x, edge_index):
        pe = self.gcn(x, edge_index)
        return pe


class TransformerEncoder(nn.Module):
    def __init__(
        self, d_input, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1
    ):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.linear = nn.Linear(d_input, d_model)

        self.mean_token = nn.Parameter(torch.randn(1, d_model))
        #self.log_var_token = nn.Parameter(torch.randn(1, d_model))

    def forward(self, x, t_pose, edge_index, mask=None):
        src = self.linear(x)
        src = src + self.pos_encoder(t_pose, edge_index)

        src = graph_to_batch(src, mask)

        #distribution_tokens = torch.stack(
        #    [
        #        self.mean_token.repeat(src.shape[0], 1),
        #        self.log_var_token.repeat(src.shape[0], 1),
        #    ],
        #    dim=1,
        #)

        distribution_tokens = self.mean_token.repeat(src.shape[0], 1)
        
        src = torch.cat([distribution_tokens, src], dim=1)
        mask = torch.cat(
            [torch.ones(src.shape[0], 2, dtype=bool, device=src.device), mask], dim=1
        )

        output = self.transformer_encoder(src, src_key_padding_mask=~mask)
        return output[:, 0]#, output[:, 1]


class TransformerDecoder(nn.Module):
    def __init__(self, d_input, d_model):
        super(TransformerDecoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.act = nn.GELU()
        self.gcn = GraphSAGE(
            d_model, 1024, out_channels=d_input, num_layers=4, act=self.act
        )

    def forward(self, x, t_pose, edge_index, mask=None):
        src = x.unsqueeze(1).repeat(1, mask.shape[1], 1)
        pe = self.pos_encoder(t_pose, edge_index)

        pe = graph_to_batch(pe, mask)

        src = src * pe  # multiply by positional encoding

        batch = batch_to_graph(src, mask)
        output = self.gcn(batch.x, edge_index)
        output = graph_to_batch(output, mask)

        return output


class TransformerAutoEncoder(nn.Module):
    def __init__(
        self, d_input, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1
    ):
        super(TransformerAutoEncoder, self).__init__()
        self.encoder = TransformerEncoder(
            d_input=d_input + 6, d_model=d_model, nhead=nhead, num_layers=num_layers
        )
        self.decoder = TransformerDecoder(d_input=d_input - 3, d_model=d_model)

    def forward(self, x, t_pose, edge_index, mask=None):
        mean = self.encoder(x, t_pose, edge_index, mask=mask)

        z_out = mean

        decoded = self.decoder(z_out, t_pose, edge_index, mask=mask)
        return decoded, mean, log_var

    def reparametrize(self, mean, log_var):
        """
        reparametrizaiton trick
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    @classmethod
    def from_pretrained(cls, artifact, checkpoint=None):
        d_model = 64
        d_input = 9
        nhead = 8
        num_layers = 4
        model = cls(
            d_input=d_input, d_model=d_model, nhead=nhead, num_layers=num_layers
        )

        if checkpoint:
            model.load_state_dict(checkpoint["model"])
            return model

        state_dict = model._get_state_dict(artifact, use_cache=True)

        model.load_state_dict(state_dict)

        return model

    @staticmethod
    def _get_state_dict(path_or_artefact: str, use_cache: bool = False):
        """
        Checks if path_or_artefact is a path if not tries to download the model from wandb
        :param path_or_artefact: path to model or wandb artifact
        :param use_cache: if true uses the cached model
        :return: state dict of model
        """

        cache_path = "./models/" + path_or_artefact
        # check if ends with .pt
        if not path_or_artefact.endswith(".pt"):
            cache_path = cache_path + ".pt"

        print("cache_path", cache_path)
        if os.path.exists(cache_path) and use_cache:
            print("Using cached model")
            return torch.load(cache_path, map_location="cpu")
        else:
            return download_wandb_artefact(path_or_artefact)


def graph_to_batch(x, mask, pad_with=0):
    """
    Converts the graphs to batched zero-padded tokens
    """
    counts = mask.sum(1)
    nested_tensor = torch.nested.as_nested_tensor(list(torch.split(x, counts.tolist())))
    padded_tensor = nested_tensor.to_padded_tensor(pad_with)

    return padded_tensor


def batch_to_graph(x, mask):
    """
    Converts the batched zero-padded tokens to graphs
    """
    return Batch.from_data_list(
        [
            Data(
                x=x[i][: mask[i].sum()],
                edge_index=torch.arange(0, mask[i].sum()).unsqueeze(0).repeat(2, 1),
            )
            for i in range(mask.shape[0])
        ]
    )


def mask_from_batch(batch: Batch) -> torch.Tensor:
    """
    Creates a mask from the batch. The mask is True for the actual tokens and False for the padding.
    :param batch: Batch object
    :return: mask
    """
    _, counts = batch.batch.unique(return_counts=True)
    indices = torch.arange(max(counts), device=batch.batch.device)
    mask = indices < counts.unsqueeze(1)
    return mask


import wandb


def download_wandb_artefact(artifact_name):
    api = wandb.Api()
    artifact = api.artifact(artifact_name, type="model")

    if wandb.run:
        wandb.run.use_artifact(artifact, type="model")

    dirs = "/".join(artifact_name.split("/")[:-1]) + "/"
    artifact_dir = artifact.download("./models/" + dirs)
    model_name = artifact_name.split("/")[-1]

    # rename the model file to artifact_name and create the necessary directory
    import os

    os.rename(artifact_dir + "model.pth", artifact_dir + model_name + ".pt")
    os.chmod(artifact_dir + model_name + ".pt", 0o777)

    return torch.load(artifact_dir + model_name + ".pt", map_location="cpu")
