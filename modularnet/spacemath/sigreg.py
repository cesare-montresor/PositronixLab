import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
#import timm, wandb, hydra, tqdm
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
import torch.distributed as dist



class LeJEPA:

    @classmethod
    def sig_reg(x: torch.Tensor, global_step: int, num_slices: int = 256) -> torch.Tensor:
        """
        Computes Sliced Isotropic Gaussian Regularization via the Epps-Pulley test statistic.
        
        Args:
            x: Input embeddings of shape (batch_size, feature_dim).
            global_step: Current training step (used as seed to synchronize projections).
            num_slices: Number of random 1D projections (slices).
            
        Returns:
            Scalar regularization loss averaged over slices.
        """
        device = x.device
        dev_kwargs = {"device": device}
        
        # 1. Slice sampling: synchronized random 1D unit projection vectors across GPUs
        g = torch.Generator(**dev_kwargs)
        g.manual_seed(global_step)
        proj_shape = (x.size(1), num_slices)
        A = torch.randn(proj_shape, generator=g, **dev_kwargs)
        A = A / A.norm(p=2, dim=0, keepdim=True)  # Unit vectors on sphere S^{d-1}
        
        # 2. Integration grid for the Epps-Pulley test statistic
        # Using 17 evaluation points along the frequency axis t in [-5, 5]
        t = torch.linspace(-5, 5, 17, **dev_kwargs)
        
        # Theoretical characteristic function (CF) for N(0, 1) and Gaussian window weight
        # phi(t) = exp(-0.5 * t^2)
        exp_f = torch.exp(-0.5 * (t ** 2))
        
        # 3. Project data and compute Empirical Characteristic Function (ECF)
        # Projections: (N, num_slices) -> (N, num_slices, num_t_points)
        x_t = (x @ A).unsqueeze(2) * t
        ecf = torch.exp(1j * x_t).mean(dim=0)  # Shape: (num_slices, num_t_points)
        
        # Synchronize empirical CF across distributed workers
        if dist.is_initialized():
            dist.all_reduce(ecf, op=dist.ReduceOp.AVG)
            world_size = dist.get_world_size()
        else:
            world_size = 1
            
        # 4. Weighted L2 distance (Epps-Pulley statistic)
        # Integral over |ECF(t) - CF(t)|^2 * weight(t) dt
        err = (ecf - exp_f).abs().square() * exp_f
        total_samples = x.size(0) * world_size
        
        # Numerical integration using trapezoidal rule
        T = torch.trapezoid(err, t, dim=1) * total_samples
        
        return T.mean()


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

"""

ViT Example

class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)
"""