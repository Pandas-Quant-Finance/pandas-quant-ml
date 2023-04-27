import torch


class StandardScaler(torch.nn.Module):

    def __init__(self, axis=-1, scaling=0.3):
        super().__init__()
        self.scaling = scaling
        self.axis = axis

    def forward(self, x, *args, **kwargs):
        x -= torch.mean(x, dim=self.axis, keepdim=True)
        std = torch.std(x, dim=self.axis, keepdim=True)
        return torch.nan_to_num(torch.divide(x, std), posinf=0.0, neginf=0.0) * self.scaling
