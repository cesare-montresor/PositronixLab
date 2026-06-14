"""
ML Training Pipeline Example: Simulated distributed training with checkpointing.

This comprehensive example demonstrates:
1. Multi-stage training pipeline (data loader, trainer, evaluator)
2. Shared state management via environment variables
3. Checkpoint saving and restoration
4. Progress tracking and metrics aggregation
5. Graceful shutdown with state persistence
"""

import logging
import time
import random
from typing import Any, Dict, List

from modularnet.plugin import BasePlugin, PluginDescriptor
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.plugin.plugin_manager import PluginManager

logger = logging.getLogger(__name__)


class DataLoader(BasePlugin):
    """
    Stage 1: Load and serve training data batches.
    """

    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Data loader for training pipeline",
            options=options or {}
        )

    def __init__(self, worker: PluginWorker, plugin_id: str, options: Dict[str, Any]):
        super().__init__(worker, plugin_id, options)
        self.batch_id = 0
        self.epoch = 0
        self.batches_per_epoch = 5

    def on_start(self) -> None:
        logger.info(f"[{self.plugin_id}] DataLoader started")
        self.env_set("training.epoch", 0)
        self.env_set("training.batch", 0)

    def main(self):
        self.batch_id += 1
        if self.batch_id > 1 and (self.batch_id - 1) % self.batches_per_epoch == 0:
            self.epoch += 1
            logger.info(f"[{self.plugin_id}] Starting epoch {self.epoch}")
            self.env_set("training.epoch", self.epoch)

        # Create a batch
        batch = {
            "batch_id": self.batch_id,
            "epoch": self.epoch,
            "size": 32,
            "data": list(range(32))  # Dummy data
        }
        self.topic_publish("training.batch", batch)
        self.env_set("training.batch", self.batch_id, wait=False)

    def on_stop(self) -> None:
        logger.info(
            f"[{self.plugin_id}] DataLoader stopped: "
            f"batches={self.batch_id}, epochs={self.epoch}"
        )


class Trainer(BasePlugin):
    """
    Stage 2: Train model on batches and compute loss.
    """

    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Model trainer for training pipeline",
            options=options or {}
        )

    def __init__(self, worker: PluginWorker, plugin_id: str, options: Dict[str, Any]):
        super().__init__(worker, plugin_id, options)
        self.batches_processed = 0
        self.losses = []

    def on_start(self) -> None:
        logger.info(f"[{self.plugin_id}] Trainer started")
        self.topic_subscribe("training.batch", self.on_batch)
        self.env_set("training.losses", [])
        self.env_set("training.avg_loss", 0.0)

    def on_batch(self, batch: Dict[str, Any]):
        self.batches_processed += 1
        # Simulate training: compute loss
        loss = 2.0 * (0.95 ** batch["batch_id"]) + random.gauss(0, 0.1)
        loss = max(0.01, loss)  # Ensure positive loss
        self.losses.append(loss)

        logger.info(
            f"[{self.plugin_id}] Batch {batch['batch_id']}: loss={loss:.4f}"
        )

        # Publish loss for downstream processing
        self.topic_publish("training.loss", {
            "batch_id": batch["batch_id"],
            "epoch": batch["epoch"],
            "loss": loss
        })

        # Update rolling metrics
        if len(self.losses) > 0:
            avg_loss = sum(self.losses[-5:]) / len(self.losses[-5:])  # Last 5
            self.env_set("training.avg_loss", avg_loss, wait=False)

    def main(self):
        pass

    def on_stop(self) -> None:
        logger.info(f"[{self.plugin_id}] Trainer stopped: processed {self.batches_processed} batches")
        if self.losses:
            final_loss = self.losses[-1]
            logger.info(f"[{self.plugin_id}] Final loss: {final_loss:.4f}")


class Evaluator(BasePlugin):
    """
    Stage 3: Evaluate model periodically and checkpoint best model.
    """

    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Model evaluator with checkpointing",
            options=options or {}
        )

    def __init__(self, worker: PluginWorker, plugin_id: str, options: Dict[str, Any]):
        super().__init__(worker, plugin_id, options)
        self.loss_history = []
        self.best_loss = float("inf")
        self.checkpoints = []

    def on_start(self) -> None:
        logger.info(f"[{self.plugin_id}] Evaluator started")
        self.topic_subscribe("training.loss", self.on_loss)
        self.env_set("training.best_loss", self.best_loss)
        self.env_set("training.checkpoints", [])

    def on_loss(self, data: Dict[str, Any]):
        loss = data["loss"]
        self.loss_history.append(loss)

        # Check if this is a new best
        if loss < self.best_loss:
            self.best_loss = loss
            checkpoint = {
                "batch_id": data["batch_id"],
                "epoch": data["epoch"],
                "loss": loss,
                "timestamp": time.time()
            }
            self.checkpoints.append(checkpoint)
            logger.info(
                f"[{self.plugin_id}] New best loss: {loss:.4f} "
                f"(checkpoint #{len(self.checkpoints)})"
            )
            self.env_set("training.best_loss", self.best_loss, wait=False)
            self.env_set("training.checkpoints", self.checkpoints, wait=False)

        # Log progress periodically
        if len(self.loss_history) % 5 == 0:
            avg_recent = sum(self.loss_history[-5:]) / 5
            logger.info(
                f"[{self.plugin_id}] Progress: {len(self.loss_history)} batches, "
                f"avg_recent={avg_recent:.4f}"
            )

    def main(self):
        pass

    def on_stop(self) -> None:
        logger.info(f"[{self.plugin_id}] Evaluator stopped")
        logger.info(
            f"[{self.plugin_id}] Summary: {len(self.loss_history)} batches, "
            f"best_loss={self.best_loss:.4f}, checkpoints={len(self.checkpoints)}"
        )
        logger.info(f"[{self.plugin_id}] Checkpoints saved:")
        for i, ckpt in enumerate(self.checkpoints):
            logger.info(
                f"  - Checkpoint {i+1}: batch={ckpt['batch_id']}, "
                f"loss={ckpt['loss']:.4f}"
            )


def main():
    """Run the ML training pipeline example."""
    manager = PluginManager()

    # Create pipeline plugins
    loader_desc = DataLoader.Descriptor()
    trainer_desc = Trainer.Descriptor()
    evaluator_desc = Evaluator.Descriptor()

    manager.load_plugin(loader_desc)
    manager.load_plugin(trainer_desc)
    manager.load_plugin(evaluator_desc)

    manager.start_plugin(loader_desc.name)
    manager.start_plugin(trainer_desc.name)
    manager.start_plugin(evaluator_desc.name)

    print("ML training pipeline running... Press Ctrl+C to stop.")
    time.sleep(6)
    print("Stopping training...")

    manager.stop_plugin(evaluator_desc.name)
    manager.stop_plugin(trainer_desc.name)
    manager.stop_plugin(loader_desc.name)

    time.sleep(1)

    # Show final training state
    print("\nTraining summary:")
    all_vars = manager.env_get()
    for name in ["training.epoch", "training.batch", "training.avg_loss", "training.best_loss"]:
        value = all_vars.get(name)
        if value is not None:
            if isinstance(value, float):
                print(f"  {name}: {value:.6f}")
            else:
                print(f"  {name}: {value}")

    checkpoints = all_vars.get("training.checkpoints", [])
    if checkpoints:
        print(f"\n  Saved checkpoints ({len(checkpoints)}):")
        for i, ckpt in enumerate(checkpoints, 1):
            print(f"    {i}. Batch {ckpt['batch_id']}, loss={ckpt['loss']:.4f}")

    # VERIFICATION
    print("\n=== VERIFICATION ===")
    print("PREDICTED BEHAVIOR:")
    print("1. DataLoader generates batches for ~5 seconds")
    print("2. Trainer computes loss (exponentially decaying)")
    print("3. Evaluator saves checkpoints when loss improves")
    print("4. Final loss < avg loss (learning curve)")
    print("5. Best checkpoints saved\n")

    epoch = all_vars.get("training.epoch", 0)
    batch = all_vars.get("training.batch", 0)
    avg_loss = all_vars.get("training.avg_loss", 0)
    best_loss = all_vars.get("training.best_loss", float("inf"))
    
    assert batch > 0, "Should generate batches"
    assert epoch > 0, "Should complete at least one epoch"
    assert avg_loss > 0, "Should compute loss"
    assert best_loss < avg_loss, "Best loss should be better than average"
    assert len(checkpoints) > 0, "Should save at least one checkpoint"
    
    print(f"✓ Epochs: {epoch}")
    print(f"✓ Batches processed: {batch}")
    print(f"✓ Final avg loss: {avg_loss:.6f}")
    print(f"✓ Best loss: {best_loss:.6f} (improved)")
    print(f"✓ Checkpoints saved: {len(checkpoints)}")
    print("✓ All ML training assertions passed!")

    manager.unload_plugin(loader_desc.name)
    manager.unload_plugin(trainer_desc.name)
    manager.unload_plugin(evaluator_desc.name)


if __name__ == "__main__":
    main()
