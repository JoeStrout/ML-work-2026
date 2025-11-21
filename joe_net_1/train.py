#!/usr/bin/env python3
"""
Standalone training script for Joe Net 1

Uses the Trainer class for headless training.
For GUI integration, use trainer.py directly with callbacks.
"""

from trainer import Trainer, TrainerConfig, TrainingCallbacks
from dataset import load_volumes


class ConsoleCallbacks(TrainingCallbacks):
    """Console output callbacks for standalone training"""

    def on_training_start(self, total_epochs, model_params):
        print("=" * 60)
        print("Joe Net 1 Training")
        print("=" * 60)
        print(f"Model parameters: {model_params:,} ({model_params/1e6:.2f}M)")
        print(f"Total epochs: {total_epochs}")
        print("=" * 60)

    def on_epoch_start(self, epoch, total_epochs):
        print(f"\nEpoch [{epoch + 1}/{total_epochs}]")

    def on_batch_end(self, epoch, batch_idx, total_batches, loss):
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx + 1}/{total_batches}], Loss: {loss:.4f}")

    def on_epoch_end(self, epoch, train_loss, val_loss=None, metrics=None):
        print(f"  Train Loss: {train_loss:.4f}")
        if val_loss is not None:
            print(f"  Val Loss: {val_loss:.4f}")
        if metrics:
            print(f"  Val Mean IoU: {metrics['mean_iou']:.4f}")
            print(f"  Val Mean Dice: {metrics['mean_dice']:.4f}")
            print(f"  Val Nuclei IoU: {metrics['nuclei_iou']:.4f}")
            print(f"  Val Mito IoU: {metrics['mito_iou']:.4f}")

    def on_validation_end(self, epoch, val_loss, metrics, sample_predictions):
        pass  # Already handled in on_epoch_end

    def on_checkpoint_saved(self, epoch, checkpoint_type, path):
        if checkpoint_type == 'best':
            print(f"  New best model saved! ({path})")
        else:
            print(f"  Checkpoint saved: {path}")

    def on_training_end(self, final_epoch, best_val_loss):
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("=" * 60)


def main():
    # Create config
    config = TrainerConfig()

    print(f"Device: {config.device}")
    print(f"Mixed Precision: {config.use_amp}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print()

    # Load volumes
    print("Loading volumes...")
    volumes = load_volumes(mip=0)
    print()

    # Create callbacks
    callbacks = ConsoleCallbacks()

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(config, volumes, callbacks=callbacks)
    print()

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
