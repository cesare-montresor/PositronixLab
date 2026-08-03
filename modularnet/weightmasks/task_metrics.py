import torch

class TaskMetrics:
    @staticmethod
    def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Standard classification accuracy for tabular, vision, and odd-one-out datasets.
        Expects logits of shape (N, C) and targets of shape (N,).
        """
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0

    @staticmethod
    def compute_sequence_accuracy(predictions: torch.Tensor, targets: torch.Tensor, pad_token_id: int = 0) -> float:
        """
        Exact sequence match for SCAN and DeepMind Math.
        A prediction is only correct if the entire sequence of tokens matches the target.
        Expects predictions and targets of shape (N, seq_len).
        """
        valid_mask = (targets != pad_token_id)
        
        # Element-wise check, ignoring padded indices
        elementwise_match = (predictions == targets) | ~valid_mask
        
        # Collapse sequence dimension: True only if all valid tokens match
        sequence_match = elementwise_match.all(dim=1)
        
        correct = sequence_match.sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0