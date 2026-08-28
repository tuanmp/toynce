from collections.abc import Mapping
from functools import partial
from typing import Any, Optional, Union

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from sklearn.metrics import brier_score_loss
from torchmetrics import AUROC, CalibrationError, MetricCollection

from toy_gaussian.model.mlp import MLP


class ToyNCE(pl.LightningModule):
    """Toy NCE model for testing the NCE loss function."""

    def __init__(self, noise_dim:int=45, noise_amplifier: int=1, hidden_dim: int=256, num_layers: int=4, dropout: float=0.0, activation: str="gelu"):
        super().__init__()
        self.net = MLP(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            dropout=dropout,
            activation=activation
        )

        self.noise_dim = noise_dim
        self.noise_amplifier = noise_amplifier
        self.save_hyperparameters(ignore=["net"])

    @property
    def q0(self):
        return torch.distributions.MultivariateNormal(
            torch.zeros(self.noise_dim, device=self.device),
            torch.eye(self.noise_dim, device=self.device)
        )

    def get_input_from_batch(self, batch):
        x = batch["x"]
        noise_sample = self.q0.sample((x.shape[0] * self.noise_amplifier, )).to(x.dtype)
        y = torch.cat([torch.ones(x.shape[0]), torch.zeros(noise_sample.shape[0])], dim=0).to(self.device).to(x.dtype)
        X = torch.cat([x, noise_sample], dim=0)
        return X, y


    def forward(self, x):
        return self.net(x)

    def criterion(self, logits, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    def training_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        logits = self(x)
        loss = self.criterion(logits.squeeze(-1), y)
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, *args, **kwargs):
        self.log("train_loss", outputs["loss"])

    def validation_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        logits = self(x)
        loss = self.criterion(logits.squeeze(-1), y)
        y_hat = torch.sigmoid(logits)
        return {"val_loss": loss, "y_hat": y_hat, "y": y, "logits": logits}

    def estimate_density_ratio(self, yhat):
        score = yhat
        ratio = self.noise_amplifier * (score / (1 - score + 1e-8))  # Avoid division by zero
        return ratio

    def on_validation_batch_end(self, outputs, batch, *args, **kwargs):
        self.log("val_loss", outputs["val_loss"])
        y = outputs["y"]
        y_hat = outputs["y_hat"][y==1]
        estimated_ratio = self.estimate_density_ratio(y_hat.squeeze(-1))
        true_ratio = (batch["log_prob"] - self.q0.log_prob(batch["x"]) ).to(estimated_ratio.dtype).exp()  # Avoid division by zero

        mae = ((estimated_ratio - true_ratio).abs() / true_ratio) .mean()

        self.log("mae", mae)


