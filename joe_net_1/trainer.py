"""
Trainer class for Joe Net 1

Encapsulates training logic with callback support for GUI integration.
Can be used standalone or with GUI callbacks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import time

from model import UNet, count_parameters
from dataset import EMSegmentationDataset
from utils import CombinedLoss, compute_metrics


class TrainingCallbacks:
    """
    Callback interface for training events

    Override these methods to get training updates for GUI display
    """
    def on_training_start(self, total_epochs, model_params):
        """Called when training starts"""
        pass

    def on_epoch_start(self, epoch, total_epochs):
        """Called at start of each epoch"""
        pass

    def on_batch_end(self, epoch, batch_idx, total_batches, loss, loss_components=None, pred_probs=None):
        """Called after each training batch

        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            total_batches: Total batches in epoch
            loss: Total loss value
            loss_components: Optional dict with BCE/Dice per channel
            pred_probs: Optional prediction probabilities for histograms
        """
        pass

    def on_epoch_end(self, epoch, train_loss, val_loss=None, metrics=None):
        """Called at end of each epoch with loss and metrics"""
        pass

    def on_validation_end(self, epoch, val_loss, metrics, sample_predictions):
        """
        Called when validation ends

        Args:
            sample_predictions: List of (img, target, pred) tuples for visualization
        """
        pass

    def on_checkpoint_saved(self, epoch, checkpoint_type, path):
        """Called when checkpoint is saved ('best', 'latest', 'final')"""
        pass

    def on_training_end(self, final_epoch, best_val_loss):
        """Called when training completes"""
        pass


class TrainerConfig:
    """Training configuration"""
    # Model
    base_channels = 32
    in_channels = 1
    out_channels = 2

    # Data
    batch_size = 16
    num_workers = 4
    patches_per_epoch = 1000

    # Training
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4

    # Loss
    bce_weight = 1.0
    dice_weight = 1.0

    # Scheduler
    use_scheduler = True
    scheduler_type = "cosine"  # "cosine" or "plateau"

    # Mixed precision
    use_amp = True  # Automatic Mixed Precision (FP16)

    # Checkpointing
    checkpoint_dir = "checkpoints"
    save_every = 10  # Save checkpoint every N epochs
    save_best = True  # Save best model based on validation loss

    # Validation
    val_every = 5  # Validate every N epochs

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """
    Trainer for EM segmentation model

    Handles training loop, validation, checkpointing with callback support
    """
    def __init__(self, config, volumes, callbacks=None):
        """
        Args:
            config: TrainerConfig instance
            volumes: (volume_img, volume_nuclei, volume_mito) tuple
            callbacks: Optional TrainingCallbacks instance
        """
        self.config = config
        self.callbacks = callbacks

        # Unpack volumes
        volume_img, volume_nuclei, volume_mito = volumes

        # Create datasets
        # Training set: 3/4 of XY area (all quadrants except top-right)
        self.train_dataset = EMSegmentationDataset(
            volume_img, volume_nuclei, volume_mito,
            is_validation=False,
            patches_per_epoch=config.patches_per_epoch,
            augment=True
        )

        # Validation set: 1/4 of XY area (top-right quadrant: X > mid_x, Y > mid_y)
        self.val_dataset = EMSegmentationDataset(
            volume_img, volume_nuclei, volume_mito,
            is_validation=True,
            patches_per_epoch=200,
            augment=False
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True if config.device == "cuda" else False
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True if config.device == "cuda" else False
        )

        # Create model
        self.model = UNet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_channels=config.base_channels
        )
        self.model = self.model.to(config.device)

        # Loss function
        self.criterion = CombinedLoss(
            bce_weight=config.bce_weight,
            dice_weight=config.dice_weight
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = None
        if config.use_scheduler:
            if config.scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=config.num_epochs
                )
            elif config.scheduler_type == "plateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
                )

        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.config.device)
            targets = targets.to(self.config.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets, return_components=True)
                    loss = loss_dict['loss']

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets, return_components=True)
                loss = loss_dict['loss']
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Callback for batch end (with loss components and predictions for histograms)
            if self.callbacks:
                # Get prediction probabilities for confidence histograms
                with torch.no_grad():
                    pred_probs = torch.sigmoid(outputs)

                self.callbacks.on_batch_end(
                    epoch, batch_idx, len(self.train_loader), loss.item(),
                    loss_components=loss_dict,
                    pred_probs=pred_probs
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, epoch, get_samples=False):
        """Validate the model"""
        self.model.eval()

        total_loss = 0.0
        all_metrics = {}
        num_batches = 0
        samples = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                images = images.to(self.config.device)
                targets = targets.to(self.config.device)

                # Forward pass
                if self.config.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                # Compute metrics
                metrics = compute_metrics(outputs, targets)
                for key, value in metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = 0.0
                    all_metrics[key] += value

                # Collect sample predictions (first batch only)
                if get_samples and batch_idx == 0:
                    for i in range(min(3, images.size(0))):
                        samples.append((
                            images[i].cpu(),
                            targets[i].cpu(),
                            outputs[i].cpu()
                        ))

                num_batches += 1

        # Average metrics
        avg_loss = total_loss / num_batches
        for key in all_metrics:
            all_metrics[key] /= num_batches

        # Callback
        if self.callbacks:
            self.callbacks.on_validation_end(epoch, avg_loss, all_metrics, samples)

        return avg_loss, all_metrics, samples

    def save_checkpoint(self, epoch, loss, checkpoint_type='latest'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config)
        }

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filename = f"{checkpoint_type}.pth"
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)

        if self.callbacks:
            self.callbacks.on_checkpoint_saved(epoch, checkpoint_type, path)

        return path

    def load_checkpoint(self, checkpoint_type='latest'):
        """Load model checkpoint"""
        filename = f"{checkpoint_type}.pth"
        path = os.path.join(self.config.checkpoint_dir, filename)

        if not os.path.exists(path):
            return 0

        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        return self.current_epoch

    def train(self, num_epochs=None):
        """
        Run full training loop

        Args:
            num_epochs: Override config.num_epochs if provided
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        # Callback: training start
        if self.callbacks:
            num_params = count_parameters(self.model)
            self.callbacks.on_training_start(num_epochs, num_params)

        # Try to resume from checkpoint
        start_epoch = self.load_checkpoint('latest')

        # Initialize train_loss in case loop doesn't execute
        train_loss = 0.0

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Callback: epoch start
            if self.callbacks:
                self.callbacks.on_epoch_start(epoch, num_epochs)

            # Train
            train_loss = self.train_epoch(epoch)

            # Update scheduler
            if self.scheduler and self.config.scheduler_type == "cosine":
                self.scheduler.step()

            # Validate
            val_loss = None
            metrics = None
            if (epoch + 1) % self.config.val_every == 0 or epoch == num_epochs - 1:
                val_loss, metrics, _ = self.validate(epoch, get_samples=True)

                # Update plateau scheduler
                if self.scheduler and self.config.scheduler_type == "plateau":
                    self.scheduler.step(val_loss)

                # Save best model
                if self.config.save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch + 1, val_loss, 'best')

            # Callback: epoch end
            if self.callbacks:
                self.callbacks.on_epoch_end(epoch, train_loss, val_loss, metrics)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1, train_loss, 'latest')

        # Save final model
        self.save_checkpoint(num_epochs, train_loss, 'final')

        # Callback: training end
        if self.callbacks:
            self.callbacks.on_training_end(num_epochs, self.best_val_loss)
