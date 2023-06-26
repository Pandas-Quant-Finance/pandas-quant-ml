import torch
import torch.nn as nn


class SharpeLoss(nn.Module):

    def __init__(self, output_size: int = 1, scaling_factor: float = 252.0, sortino: bool = False):
        super().__init__()
        self.output_size = output_size  # in case we have multiple targets => output dim[-1] = output_size * n_quantiles
        self.scaling_factor = torch.tensor(scaling_factor)

    def forward(self, y_true, weights):
        """
        Args:
            y_true: asset returns
            weights: predicted weights

        Returns:

        """
        captured_returns = weights * y_true
        mean_returns = torch.mean(captured_returns)
        std = torch.sqrt(torch.mean((captured_returns - mean_returns) ** 2) + 1e-9)
        # if sortino: std = torch.sqrt(torch.mean(torch.minimum(captured_returns - mean_returns, 0) ** 2) + 1e-9)
        return -(
                (mean_returns / std)
                * torch.sqrt(self.scaling_factor)
        )
