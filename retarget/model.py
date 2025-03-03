import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GraphSAGE
except ImportError:
    import warnings

    warnings.warn("PyTorch Geometric is not installed. Installing it now.")
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "torch-geometric==2.5.1"]
    )
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GraphSAGE


class PositionalEncoding(nn.Module):
    """
    Graph-based positional encoding using GraphSAGE.

    Attributes:
        gcn: GraphSAGE network for encoding positional information
    """

    def __init__(self, d_model: int):
        """
        Initialize the positional encoding module.

        Parameters
        ----------
        d_model: Dimension of the model
        """
        super(PositionalEncoding, self).__init__()
        self.gcn = GraphSAGE(3, d_model, num_layers=2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute positional encoding for input features.

        Parameters
        ----------
        x: Node features of shape [num_nodes, 3]
        edge_index: Graph connectivity of shape [2, num_edges]

        Returns:
            Positional encoding of shape [num_nodes, d_model]
        """
        pe = self.gcn(x, edge_index)
        return pe


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing graph-structured motion data.

    Attributes:
        transformer_encoder: Standard transformer encoder
        pos_encoder: Positional encoding module
        linear: Linear projection layer
        pose_token: Learnable token for pose representation
        root_traj_token: Learnable token for root trajectory representation
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 512,
        dropout: float = 0,
    ):
        """
        Initialize the transformer encoder.

        Parameters
        ----------
        d_input: Dimension of input features
        d_model: Dimension of the model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        """
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.linear = nn.Linear(d_input, d_model)

        self.pose_token = nn.Parameter(torch.randn(1, d_model))
        self.root_traj_token = nn.Parameter(torch.randn(1, d_model))

    def forward(
        self,
        x: torch.Tensor,
        rest_pose: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer encoder.

        Parameters
        ----------
        x: Input features [batch_size, max_nodes, d_input]
        rest_pose: Rest pose positions of shape [num_nodes, 3]
        edge_index: Graph connectivity [2, num_edges]
        mask: Boolean mask for valid nodes [batch_size, max_nodes]

        Returns:
            Tuple containing:
                - pose latent representation [batch_size, d_model]
                - root trajectory latent representation [batch_size, d_model]
        """
        src = x.clone()
        src = src * self.pos_encoder(rest_pose, edge_index)

        src = graph_to_batch(src, mask)

        distribution_tokens = torch.stack(
            [
                self.pose_token.repeat(src.shape[0], 1),
                self.root_traj_token.repeat(src.shape[0], 1),
            ],
            dim=1,
        )

        src = torch.cat([distribution_tokens, src], dim=1)
        mask = torch.cat(
            [torch.ones(src.shape[0], 2, dtype=bool, device=src.device), mask], dim=1
        )

        output = self.transformer_encoder(src, src_key_padding_mask=~mask)
        return output[:, 0], output[:, 1]


class TransformerEncoderDecoder(nn.Module):
    """
    Transformer decoder for reconstructing motion from latent representations.

    Attributes:
        transformer_decoder: Transformer encoder used as decoder
        pos_encoder: Positional encoding module
        src_root_traj_pos_enc: Learnable positional encoding for root trajectory
        linear_pose: Linear projection for pose output
        linear_traj: Linear projection for trajectory output
    """

    def __init__(
        self,
        d_output: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 512,
        dropout: float = 0,
    ):
        """
        Initialize the transformer encoder-decoder.

        Parameters
        ----------
        d_output: Dimension of output features
        d_model: Dimension of the model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        """
        super(TransformerEncoderDecoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.src_root_traj_pos_enc = nn.Parameter(torch.randn(1, d_model))

        self.linear_pose = nn.Linear(d_model, d_output)
        self.linear_traj = nn.Linear(d_model, 3)

    def forward(
        self,
        pose_latent: torch.Tensor,
        root_traj_latent: torch.Tensor,
        rest_pose: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer decoder.

        Parameters
        ----------
        pose_latent: Latent pose representation [batch_size, d_model]
        root_traj_latent: Latent root trajectory representation [batch_size, d_model]
        rest_pose: Rest pose positions [num_nodes, 3]
        edge_index: Graph connectivity [2, num_edges]
        mask: Boolean mask for valid nodes [batch_size, max_nodes]

        Returns:
            Tuple containing:
                - decoded pose features [batch_size, max_nodes, d_output]
                - decoded root trajectory [batch_size, max_nodes, 3]
        """
        src_pose = pose_latent.unsqueeze(1).repeat(1, mask.shape[1], 1)
        src_root_traj = root_traj_latent.unsqueeze(1)

        pe = self.pos_encoder(rest_pose, edge_index)
        pe = graph_to_batch(pe, mask)
        src = src_pose * pe  # multiply by positional encoding
        src_root_traj_latent = src_root_traj * self.src_root_traj_pos_enc
        src = torch.cat([src, src_root_traj], dim=1)
        mask = torch.cat(
            [mask, torch.ones(src.shape[0], 1, dtype=bool, device=src.device)], dim=1
        )

        output = self.transformer_decoder(src, src_key_padding_mask=~mask)

        # Project d_model dim to output dim
        output_pose = self.linear_pose(output[:, :-1, :])
        output_traj = self.linear_traj(output[:, -1:, :])

        return output_pose, output_traj


class TransformerAutoEncoder(nn.Module):
    """
    Transformer-based autoencoder for motion data.

    Attributes:
        encoder: Transformer encoder module
        decoder: Transformer decoder module
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 512,
        dropout: float = 0,
    ):
        """
        Initialize the transformer autoencoder.

        Parameters
        ----------
        d_input: Dimension of input features
        d_model: Dimension of the model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        """
        super(TransformerAutoEncoder, self).__init__()
        self.encoder = TransformerEncoder(
            d_input=d_input + 9, d_model=d_model, nhead=nhead, num_layers=num_layers
        )

        self.decoder = TransformerEncoderDecoder(
            d_output=d_input - 3, d_model=d_model, nhead=nhead, num_layers=num_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        rest_pose: torch.Tensor,
        edge_index: torch.Tensor,
        rest_pose_decoder: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer autoencoder.

        Parameters
        ----------
        x: Input features [batch_size, max_nodes, d_input]
        rest_pose: Rest pose positions for encoder [num_nodes, 3]
        edge_index: Graph connectivity [2, num_edges]
        rest_pose_decoder: Optional different rest pose for decoder
        mask: Boolean mask for valid nodes [batch_size, max_nodes]

        Returns:
            Tuple containing:
                - decoded pose features [batch_size, max_nodes, d_output]
                - decoded root trajectory [batch_size, max_nodes, 3]
                - mean of latent distribution [batch_size, d_model]
                - log variance of latent distribution [batch_size, d_model]
        """
        z_pose, z_root_traj = self.encoder(x, rest_pose, edge_index, mask=mask)
        mean, log_var = z_pose, None

        if rest_pose_decoder is None:
            output_pose, output_traj = self.decoder(
                z_pose, z_root_traj, rest_pose, edge_index, mask=mask
            )
        else:
            output_pose, output_traj = self.decoder(
                z_pose, z_root_traj, rest_pose_decoder, edge_index, mask=mask
            )

        return output_pose, output_traj, mean, log_var

    def reparametrize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for variational autoencoders.

        Parameters
        ----------
        mean: Mean of the distribution
        log_var: Log variance of the distribution

        Returns:
            Sampled tensor from the distribution
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    @classmethod
    def from_pretrained(
        cls,
        artifact: str,
        checkpoint: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> "TransformerAutoEncoder":
        """
        Load a pretrained model from a checkpoint or artifact.

        Parameters
        ----------
        artifact: Name of the artifact or path to load
        checkpoint: Optional checkpoint dictionary containing model weights
        model_config: Optional model configuration

        Returns:
            Initialized model with pretrained weights
        """
        default_config = {"d_model": 64, "d_input": 9, "nhead": 8, "num_layers": 4}

        # merge default config with model_config
        model_config = {**default_config, **(model_config or {})}

        d_model = model_config["d_model"]
        d_input = model_config["d_input"]
        nhead = model_config["nhead"]
        num_layers = model_config["num_layers"]

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
    def _get_state_dict(
        path_or_artefact: str, use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Get state dict from a path or wandb artifact.

        Parameters
        ----------
        path_or_artefact: Path to model or wandb artifact
        use_cache: If true, uses the cached model

        Returns:
            State dict of the model
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


class Discriminator(nn.Module):
    """
    Discriminator model for GAN training.

    Used to discriminate between real and fake data in a Vanilla GAN setup.

    Attributes:
        transformer_encoder: Transformer encoder for processing input
        pos_encoder: Positional encoding module
        linear: Linear projection layer
        cls_token: Classification token
        linear_cls: Linear layer for classification
        sigmoid: Sigmoid activation for binary classification
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 512,
        dropout: float = 0,
    ):
        """
        Initialize the discriminator.

        Parameters
        ----------
        d_input: Dimension of input features
        d_model: Dimension of the model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        """
        super(Discriminator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.linear = nn.Linear(d_input, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, d_model))
        self.linear_cls = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        rest_pose: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Parameters
        ----------
        x: Input features [batch_size, max_nodes, d_input]
        rest_pose: Rest pose positions [num_nodes, 3]
        edge_index: Graph connectivity [2, num_edges]
        mask: Boolean mask for valid nodes [batch_size, max_nodes]

        Returns:
            Discriminator output [batch_size, 1]
        """
        src = x.clone()
        src = src * self.pos_encoder(rest_pose, edge_index)

        src = graph_to_batch(src, mask)

        distribution_tokens = torch.stack(
            [
                self.cls_token.repeat(src.shape[0], 1),
            ],
            dim=1,
        )

        src = torch.cat([distribution_tokens, src], dim=1)
        mask = torch.cat(
            [torch.ones(src.shape[0], 1, dtype=bool, device=src.device), mask], dim=1
        )

        output = self.transformer_encoder(src, src_key_padding_mask=~mask)

        valid_prob = self.sigmoid(self.linear_cls(output[:, 0]))

        return valid_prob


class StyleEncoder(nn.Module):
    """
    Style encoder for extracting style information from motion data.

    Takes a batch of poses from the same skeleton and encodes them into a single latent vector.
    First encodes poses into latent space, then aggregates using a transformer.
    The order of poses is not important, so positional encoding is not used.
    A style token is added to aggregate pose information.

    Attributes:
        encoder: Transformer encoder for initial encoding
        aggregator: Transformer encoder for aggregating pose encodings
        style_token: Learnable token for style representation
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 512,
        dropout: float = 0,
    ):
        """
        Initialize the style encoder.

        Parameters
        ----------
        d_input: Dimension of input features
        d_model: Dimension of the model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        """
        super(StyleEncoder, self).__init__()
        self.encoder = TransformerEncoder(
            d_input=d_input, d_model=d_model, nhead=nhead, num_layers=num_layers
        )
        self.aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            ),
            num_layers,
        )
        self.style_token = nn.Parameter(torch.randn(1, d_model))

    def forward(
        self,
        x: torch.Tensor,
        rest_pose: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the style encoder.

        Parameters
        ----------
        x: Input features [batch_size, max_nodes, d_input]
        rest_pose: Rest pose positions [num_nodes, 3]
        edge_index: Graph connectivity [2, num_edges]
        mask: Boolean mask for valid nodes [batch_size, max_nodes]

        Returns:
            Style encoding [batch_size, d_model]
        """
        src = self.encoder(x, rest_pose, edge_index, mask=mask)
        distribution_tokens = torch.stack(
            [
                self.style_token.repeat(src.shape[0], 1),
            ],
            dim=1,
        )

        src = torch.cat([distribution_tokens, src], dim=1)
        mask = torch.cat(
            [torch.ones(src.shape[0], 1, dtype=bool, device=src.device), mask], dim=1
        )

        output = self.aggregator(src, src_key_padding_mask=~mask)
        return output[:, 0]


def graph_to_batch(
    x: torch.Tensor, mask: torch.Tensor, pad_with: Union[int, float] = 0
) -> torch.Tensor:
    """
    Converts graph data to batched zero-padded tokens.

    Parameters
    ----------
    x: Graph node features
    mask: Boolean mask for valid nodes
    pad_with: Value to use for padding

    Returns:
        Padded tensor of shape [batch_size, max_nodes, feature_dim]
    """
    counts = mask.sum(1)
    nested_tensor = torch.nested.as_nested_tensor(list(torch.split(x, counts.tolist())))
    padded_tensor = nested_tensor.to_padded_tensor(pad_with)

    return padded_tensor


def batch_to_graph(x: torch.Tensor, mask: torch.Tensor) -> Batch:
    """
    Converts batched zero-padded tokens back to graph data.

    Parameters
    ----------
    x: Batched tensor [batch_size, max_nodes, feature_dim]
    mask: Boolean mask for valid nodes [batch_size, max_nodes]

    Returns:
        PyTorch Geometric Batch object containing the graphs
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
    Creates a mask from a PyTorch Geometric Batch object.

    The mask is True for actual tokens and False for padding.

    Parameters
    ----------
    batch: PyTorch Geometric Batch object

    Returns:
        Boolean mask of shape [batch_size, max_nodes]
    """
    _, counts = batch.batch.unique(return_counts=True)
    indices = torch.arange(max(counts), device=batch.batch.device)
    mask = indices < counts.unsqueeze(1)
    return mask


import wandb


def download_wandb_artefact(artifact_name: str) -> Dict[str, torch.Tensor]:
    """
    Download a model artifact from Weights & Biases.

    Parameters
    ----------
    artifact_name: Name of the artifact to download

    Returns:
        State dict of the model
    """
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
