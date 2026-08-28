import torch


class ToyNCEDataset(torch.utils.data.Dataset):
    """Toy NCE dataset for testing the NCE loss function."""

    def __init__(self, sample, log_prob):
        super().__init__()
        self.sample = sample
        self.log_prob = log_prob

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        x = self.sample[idx]
        log_prob = self.log_prob[idx]
        return {"x": x, "log_prob": log_prob}