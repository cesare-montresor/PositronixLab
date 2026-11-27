
import torch


class DisplacementFilter:
    """
    Event-driven spatial gate. 
    Returns True if the tensor deviates from the keyframe, False otherwise.
    """
    def __init__(self, threshold: float = 0.5, method: str = 'max') -> None:
        self.threshold = threshold
        self.method = method
        self._keyframe: dict[str, torch.Tensor] = {}

    def check_and_update(self, key: str, value: torch.Tensor) -> bool:
        kf = self._keyframe.get(key)
        
        # Always emit on initialization or shape change
        if kf is None or kf.shape != value.shape:
            self._keyframe[key] = value.clone()
            return True

        # Calculate physical displacement
        diff_tensor = value.to(kf.device) - kf
        
        if self.method == 'max':
            diff = torch.linalg.norm(diff_tensor, dim=-1).max().item()
        else: # 'rms'
            dists_sq = (diff_tensor ** 2).sum(dim=-1)
            diff = torch.sqrt(dists_sq.mean()).item()

        # Update keyframe and signal emission if threshold crossed
        if diff > self.threshold:
            self._keyframe[key] = value.clone()
            return True
            
        return False