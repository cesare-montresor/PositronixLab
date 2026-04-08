
import torch


def zclamp_(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Clamp values away from zero to ±eps boundaries."""
    mask = (x.abs() < eps) & (x != 0)
    x[mask] = torch.sign(x[mask]) * eps
    return x

def zclamp(self: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Clamp values away from zero to ±eps boundaries."""
    return zclamp_(self.clone(), eps)



# Monkey patch into Tensor class
torch.Tensor.zclamp_ = zclamp_
torch.Tensor.zclamp = zclamp


def spow_(x: torch.Tensor, power, eps=1e-8) -> torch.Tensor:
    """Safe pow."""
    return x.zclamp_(eps).pow_(power)

def spow(self: torch.Tensor, power, eps=1e-8) -> torch.Tensor:
    """Safe pow."""
    return spow_(self.clone(), power, eps)

# Monkey patch into Tensor class
torch.Tensor.spow = spow_
torch.Tensor.spow = spow
