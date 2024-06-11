from typing import List

import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GCNModel(nn.Module):
    """GCN model"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int = 4,
        dropout: List = None,
        activation: List = None,
        normalize: List = None,
        seed=None,
        is_reg: bool = False,
    ):
        # Init parent
        super().__init__()
        if seed:
            torch.manual_seed(seed)

        self.activation = activation or [nn.ReLU()] * num_layers
        self.normalize = normalize or [False] * num_layers
        self.dropout = dropout or [0.0] * num_layers

        self.is_reg = is_reg

        # GCN layers:
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            GCNConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                normalize=self.normalize[0],
            )
        )
        for i in range(num_layers):
            self.conv_layers.append(
                GCNConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    normalize=self.normalize[i],
                )
            )

        # Output layer
        self.out = Linear(in_features=hidden_channels * 2, out_features=out_channels)

    def forward(self, data):
        # First Conv layer
        x = F.dropout(data.x.float(), p=self.dropout[0], training=self.training)
        hidden = self.conv_layers[0](x, data.edge_index)
        hidden = self.activation[0](hidden)

        # Other Conv layers
        for i in range(1, self.num_layers):
            if self.dropout and self.dropout[i]:
                hidden = F.dropout(x, p=self.dropout[i], training=self.training)
            hidden = self.conv_layers[i](hidden, data.edge_index)
            hidden = self.activation[i](hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, data.batch), gap(hidden, data.batch)], dim=1)

        # Apply a final (linear) predictor
        out = self.out(hidden)
        return out

    def _forward_to_pred(self, inputs):
        self.eval()
        out = self.forward(inputs)
        preds = out["pred"] if isinstance(out, dict) else out
        return preds

    @torch.no_grad()
    def predict(self, inputs):
        if self.is_reg:
            preds = self._forward_to_pred(inputs)
            return preds
        probs = self.predict_proba(inputs)[:, -1]
        preds = torch.ge(probs, 0.5).int()
        return preds

    @torch.no_grad()
    def predict_proba(self, inputs):
        probs = self._forward_to_pred(inputs)
        probs = torch.sigmoid(probs)
        return probs
