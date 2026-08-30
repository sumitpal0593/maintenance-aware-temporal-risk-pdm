from __future__ import annotations

import torch
from torch import nn


class TemporalRiskGRU(nn.Module):
    """
    Fixed-horizon temporal risk model used by the revised manuscript.

    Pipeline
    --------
    operational sequence X_i
        -> GRU
        -> final hidden representation h_i

    encoded static specifications z_i
        -> static MLP
        -> c_i

    [h_i ; c_i]
        -> fusion MLP
        -> v_i
        -> scalar logit
        -> sigmoid risk P(Y_i(tau)=1 | X_i, z_i)

    This intentionally replaces the original K-dimensional dual-hazard
    notebook head so that the implementation matches the paper's fixed-horizon
    formulation.
    """

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        static_hidden: int = 128,
        fusion_hidden: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()

        if dynamic_dim <= 0 or static_dim <= 0:
            raise ValueError("dynamic_dim and static_dim must be positive.")
        if gru_hidden <= 0 or gru_layers <= 0:
            raise ValueError("GRU dimensions/layers must be positive.")

        self.dynamic_dim = int(dynamic_dim)
        self.static_dim = int(static_dim)
        self.gru_hidden = int(gru_hidden)
        self.gru_layers = int(gru_layers)
        self.static_hidden = int(static_hidden)
        self.fusion_hidden = int(fusion_hidden)
        self.dropout_rate = float(dropout)

        self.gru = nn.GRU(
            input_size=self.dynamic_dim,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.gru_layers > 1 else 0.0,
        )

        self.static_net = nn.Sequential(
            nn.Linear(self.static_dim, self.static_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.static_hidden, self.static_hidden),
            nn.ReLU(),
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(
                self.gru_hidden + self.static_hidden,
                self.fusion_hidden,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        self.risk_head = nn.Linear(self.fusion_hidden, 1)

    def encode_temporal(
        self,
        X_pad: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            X_pad,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h = self.gru(packed)
        return h[-1]

    def encode_static(self, static: torch.Tensor) -> torch.Tensor:
        return self.static_net(static)

    def forward(
        self,
        X_pad: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor,
    ):
        h_temporal = self.encode_temporal(X_pad, lengths)
        h_static = self.encode_static(static)

        fused = self.fusion_net(
            torch.cat([h_temporal, h_static], dim=1)
        )
        logit = self.risk_head(fused).squeeze(1)

        # Return representations as well because the reviewer-requested
        # temporal-control experiment will use h_temporal in Stage 2F.
        return logit, h_temporal, fused
