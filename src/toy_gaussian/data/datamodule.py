from typing import Optional

import lightning as pl
import torch

from toy_gaussian.data.dataset import ToyNCEDataset


class ToyNCEDataModule(pl.LightningDataModule):
    """Toy NCE data module for testing the NCE loss function."""

    def __init__(self, q0_mean: str, q0_cov: str, n_sample: int, batch_size: int=64, num_workers: int=4):
        super().__init__()
        self.q0_mean = torch.load(q0_mean)
        self.q0_cov = torch.load(q0_cov)
        self.target = torch.distributions.MultivariateNormal(self.q0_mean, self.q0_cov)
        self.n_sample = n_sample
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):

        train_sample = self.target.sample((self.n_sample, ))
        train_log_prob = self.target.log_prob(train_sample)

        print(train_log_prob.mean(), 
              - torch.log(1/torch.sqrt(torch.tensor(2*torch.pi))) * self.q0_mean.shape[0],  
              - torch.log(torch.det(self.q0_cov)) / 2, 
        )

        self.train_dataset = ToyNCEDataset(
            sample=train_sample,
            log_prob=train_log_prob
        )
        val_sample = self.target.sample((self.n_sample, ))
        val_log_prob = self.target.log_prob(val_sample)

        self.val_dataset = ToyNCEDataset(
            sample=val_sample,
            log_prob=val_log_prob
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )