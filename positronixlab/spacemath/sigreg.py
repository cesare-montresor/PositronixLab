import torch


def sigreg(pos: torch.Tensor, sketch_dim: int = 64, method: str = "weak") -> torch.Tensor:
    """Dispatch isotropic Gaussian regularization ('weak' covariance or 'strong' characteristic function)."""
    if method == "weak":
        return sigreg_weak(pos, sketch_dim)
    elif method == "strong":
        return sigreg_strong(pos, sketch_dim)
    else:
        raise ValueError(f"Unknown method: {method}")

def sigreg_weak(pos: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """Frobenius distance between (optionally sketched) empirical covariance and identity."""
    # https://arxiv.org/abs/2603.05924v1
    N, D = pos.shape
    if D > sketch_dim:
        S = torch.randn(sketch_dim, D, device=pos.device, dtype=pos.dtype) / (D ** 0.5)
        pos = pos @ S.T
        D = sketch_dim
    pos = pos - pos.mean(0, keepdim=True)
    cov = pos.T @ pos / (N - 1 + 1e-6)
    eye = torch.eye(D, device=pos.device, dtype=pos.dtype)
    return torch.sqrt((cov - eye).pow(2).sum() + 1e-12)   # Frobenius, eps contro il NaN

def sigreg_strong(pos: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """Sliced characteristic-function distance to standard normal N(0, I)."""
    # https://arxiv.org/abs/2511.08544
    N, D = pos.shape
    A = torch.randn(D, sketch_dim, device=pos.device, dtype=pos.dtype)
    A = A / (A.norm(dim=0, keepdim=True) + 1e-6)
    t = torch.linspace(-5, 5, 17, device=pos.device, dtype=pos.dtype)
    target = torch.exp(-0.5 * t ** 2)
    xt = (pos @ A).unsqueeze(2) * t                      # (N, S, T)
    re, im = xt.cos().mean(0), xt.sin().mean(0)          # evita i complessi (più sicuro con compile)
    err = ((re - target) ** 2 + im ** 2) * target
    return (torch.trapz(err, t, dim=1) * N).mean()