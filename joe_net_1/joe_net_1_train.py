#!/usr/bin/env python3
"""
Joe Net 1 - Training GUI

Extended version of joe_net_1.py with integrated training visualization.
Shows live graphs of loss and sample predictions during training.
"""

import wx
import wx.lib.plot as wxplot
from cloudvolume import CloudVolume, Bbox
from PIL import Image
import numpy as np
import random
import threading
import torch
import time

# Import training components
from trainer import Trainer, TrainerConfig, TrainingCallbacks
from dataset import load_volumes
from utils import visualize_prediction

# CloudVolume paths
CLOUDVOLUME_PATH_IMG = "gs://joe_exp/jarvis/jarvis-1-img/"
CLOUDVOLUME_PATH_NUCLEI = "gs://joe_exp/jarvis/jarvis-1-nuclei/"
CLOUDVOLUME_PATH_MITO = "gs://joe_exp/jarvis/jarvis-1-mito/"

# Volume bounds for training data
VOL_MIN_X, VOL_MIN_Y, VOL_MIN_Z = 5852, 4020, 800
VOL_MAX_X, VOL_MAX_Y, VOL_MAX_Z = 7900, 6068, 900

# Slice size
SLICE_SIZE = 256

# Global storage for loaded volumes
volume_img = None
volume_nuclei = None
volume_mito = None
volumes_loaded = False


def load_volumes_global(mip=0):
    """Load volumes into global variables"""
    global volume_img, volume_nuclei, volume_mito, volumes_loaded

    if volumes_loaded:
        return (volume_img, volume_nuclei, volume_mito)

    volume_img, volume_nuclei, volume_mito = load_volumes(mip=mip)
    volumes_loaded = True

    return (volume_img, volume_nuclei, volume_mito)


class ImagePanel(wx.Panel):
    """Panel that displays an image."""

    def __init__(self, parent, size=(256, 256)):
        super().__init__(parent)
        self.size = size
        self.wx_bitmap = None

        self.SetMinSize(size)
        self.SetMaxSize(size)
        self.Bind(wx.EVT_PAINT, self.on_paint)

        # Default to blank image
        blank = Image.new('RGB', size, color='gray')
        self.update_image(blank)

    def update_image(self, pil_image):
        """Update the displayed image."""
        # Convert PIL image to wxPython format
        wx_image = wx.Image(pil_image.size[0], pil_image.size[1])
        wx_image.SetData(pil_image.convert("RGB").tobytes())
        self.wx_bitmap = wx_image.ConvertToBitmap()
        self.Refresh()

    def on_paint(self, event):
        """Paint the image on the panel."""
        dc = wx.PaintDC(self)
        if self.wx_bitmap:
            dc.DrawBitmap(self.wx_bitmap, 0, 0, True)


class TrainingPanel(wx.Panel):
    """Panel showing training progress with graphs and visualizations"""

    def __init__(self, parent):
        super().__init__(parent)

        # Training state
        self.trainer = None
        self.training_thread = None
        self.is_training = False
        self.training_start_time = None

        # Data for plotting
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.iou_epochs = []  # Actual epoch numbers when IoU was measured
        self.nuclei_ious = []
        self.mito_ious = []

        # Batch-level data for live updates
        self.batch_losses = []  # List of (batch_number, loss) tuples
        self.batch_counter = 0  # Global batch counter across all epochs

        # Loss components data (epoch-level, stored per epoch for entire run)
        self.epoch_bce_nuclei = []
        self.epoch_bce_mito = []
        self.epoch_dice_nuclei = []
        self.epoch_dice_mito = []
        self.epoch_total_loss = []

        # Accumulators for averaging loss components within an epoch
        self.current_epoch_bce_nuclei = []
        self.current_epoch_bce_mito = []
        self.current_epoch_dice_nuclei = []
        self.current_epoch_dice_mito = []

        # Histogram data (10 bins each: 0-0.1, 0.1-0.2, ..., 0.9-1.0)
        self.nuclei_hist_bins = [0] * 10
        self.mito_hist_bins = [0] * 10

        # UI update locks to prevent reentrant calls
        self._periodic_update_in_progress = False
        self._epoch_update_in_progress = False

        # Time-based throttling for periodic updates
        self._last_periodic_update_time = 0
        self._periodic_update_interval = 0.5  # seconds

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Create the UI layout - everything visible at once"""
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Top section: Title, buttons, status, progress
        top_sizer = wx.BoxSizer(wx.VERTICAL)

        # Title
        title = wx.StaticText(self, label="Training Dashboard")
        font = title.GetFont()
        font.PointSize += 4
        font = font.Bold()
        title.SetFont(font)
        top_sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)

        # Control buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.start_button = wx.Button(self, label="▶ Start Training")
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start_training)
        button_sizer.Add(self.start_button, 0, wx.ALL, 5)

        self.stop_button = wx.Button(self, label="⏸ Pause Training")
        self.stop_button.Enable(False)
        button_sizer.Add(self.stop_button, 0, wx.ALL, 5)

        top_sizer.Add(button_sizer, 0, wx.ALL | wx.CENTER, 5)

        # Status text
        self.status_text = wx.StaticText(self, label="Ready to train")
        top_sizer.Add(self.status_text, 0, wx.ALL | wx.CENTER, 5)

        # Progress bar
        self.progress = wx.Gauge(self, range=100, size=(1400, 25))
        top_sizer.Add(self.progress, 0, wx.ALL | wx.CENTER, 5)

        main_sizer.Add(top_sizer, 0, wx.ALL | wx.EXPAND, 5)

        # Middle section: Left half (3 rows of graphs) and Right half (Predictions)
        middle_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Left half: Three rows of graphs
        left_half_sizer = wx.BoxSizer(wx.VERTICAL)

        # Row 1: IoU Metrics
        metrics_box = wx.StaticBox(self, label="IoU Metrics")
        metrics_box_sizer = wx.StaticBoxSizer(metrics_box, wx.VERTICAL)

        self.metrics_canvas = wxplot.PlotCanvas(self, size=(700, 250))
        self.metrics_canvas.SetEnableLegend(True)
        self.metrics_canvas.SetEnableGrid(True)
        metrics_box_sizer.Add(self.metrics_canvas, 1, wx.EXPAND | wx.ALL, 5)

        left_half_sizer.Add(metrics_box_sizer, 0, wx.ALL | wx.EXPAND, 5)

        # Row 2: Loss Components (including total loss)
        loss_comp_box = wx.StaticBox(self, label="Loss Components (BCE, Dice, Total)")
        loss_comp_box_sizer = wx.StaticBoxSizer(loss_comp_box, wx.VERTICAL)

        self.loss_comp_canvas = wxplot.PlotCanvas(self, size=(700, 250))
        self.loss_comp_canvas.SetEnableLegend(True)
        self.loss_comp_canvas.SetEnableGrid(True)
        loss_comp_box_sizer.Add(self.loss_comp_canvas, 1, wx.EXPAND | wx.ALL, 5)

        left_half_sizer.Add(loss_comp_box_sizer, 0, wx.ALL | wx.EXPAND, 5)

        # Row 3: Both histograms side-by-side
        hist_row_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Nuclei histogram
        nuclei_hist_box = wx.StaticBox(self, label="Nuclei Confidence")
        nuclei_hist_box_sizer = wx.StaticBoxSizer(nuclei_hist_box, wx.VERTICAL)

        self.nuclei_hist_canvas = wxplot.PlotCanvas(self, size=(340, 200))
        self.nuclei_hist_canvas.SetEnableLegend(False)
        self.nuclei_hist_canvas.SetEnableGrid(True)
        nuclei_hist_box_sizer.Add(self.nuclei_hist_canvas, 1, wx.EXPAND | wx.ALL, 5)

        hist_row_sizer.Add(nuclei_hist_box_sizer, 1, wx.ALL | wx.EXPAND, 5)

        # Mito histogram
        mito_hist_box = wx.StaticBox(self, label="Mito Confidence")
        mito_hist_box_sizer = wx.StaticBoxSizer(mito_hist_box, wx.VERTICAL)

        self.mito_hist_canvas = wxplot.PlotCanvas(self, size=(340, 200))
        self.mito_hist_canvas.SetEnableLegend(False)
        self.mito_hist_canvas.SetEnableGrid(True)
        mito_hist_box_sizer.Add(self.mito_hist_canvas, 1, wx.EXPAND | wx.ALL, 5)

        hist_row_sizer.Add(mito_hist_box_sizer, 1, wx.ALL | wx.EXPAND, 5)

        left_half_sizer.Add(hist_row_sizer, 0, wx.ALL | wx.EXPAND, 5)

        middle_sizer.Add(left_half_sizer, 0, wx.ALL, 5)

        # Right half: Predictions (three rows)
        pred_box = wx.StaticBox(self, label="Predictions (Input | Ground Truth | Prediction)")
        pred_box_sizer = wx.StaticBoxSizer(pred_box, wx.VERTICAL)

        # Three image panels for predictions
        self.pred_panels = []
        for i in range(3):
            panel = ImagePanel(self, size=(768, 256))  # 3x256 wide
            pred_box_sizer.Add(panel, 0, wx.ALL | wx.CENTER, 3)
            self.pred_panels.append(panel)

        middle_sizer.Add(pred_box_sizer, 0, wx.ALL, 5)

        main_sizer.Add(middle_sizer, 0, wx.ALL | wx.EXPAND, 5)

        self.SetSizer(main_sizer)

    def on_start_training(self, event):
        """Start training in background thread"""
        if self.is_training:
            return

        # Immediately disable button and update status
        self.start_button.Enable(False)
        self.status_text.SetLabel("Loading volumes from CloudVolume...")
        self.Update()  # Force UI update

        # Load volumes if not loaded (this is the slow part)
        volumes = load_volumes_global(mip=0)

        # Update status
        self.status_text.SetLabel("Initializing trainer...")
        self.Update()

        # Create config
        config = TrainerConfig()
        config.num_epochs = 100
        config.batch_size = 8  # Smaller for CPU training
        config.val_every = 2  # Validate more frequently for GUI

        # Create callbacks
        callbacks = GUICallbacks(self)

        # Create trainer
        self.trainer = Trainer(config, volumes, callbacks=callbacks)

        # Reset data
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.iou_epochs = []
        self.nuclei_ious = []
        self.mito_ious = []
        self.batch_losses = []
        self.batch_counter = 0

        # Reset loss components data
        self.epoch_bce_nuclei = []
        self.epoch_bce_mito = []
        self.epoch_dice_nuclei = []
        self.epoch_dice_mito = []
        self.epoch_total_loss = []
        self.current_epoch_bce_nuclei = []
        self.current_epoch_bce_mito = []
        self.current_epoch_dice_nuclei = []
        self.current_epoch_dice_mito = []

        # Reset histogram data
        self.nuclei_hist_bins = [0] * 10
        self.mito_hist_bins = [0] * 10

        # Start training in background thread
        self.is_training = True
        self.stop_button.Enable(True)

        self.training_thread = threading.Thread(target=self.run_training, daemon=True)
        self.training_thread.start()

    def run_training(self):
        """Run training (called in background thread)"""
        # Record start time
        self.training_start_time = time.time()
        try:
            self.trainer.train()
        except Exception as e:
            wx.CallAfter(self.on_training_error, str(e))
        finally:
            wx.CallAfter(self.on_training_complete)

    def on_training_error(self, error_msg):
        """Handle training error"""
        wx.MessageBox(f"Training error: {error_msg}", "Error", wx.OK | wx.ICON_ERROR)
        self.on_training_complete()

    def on_training_complete(self):
        """Training finished"""
        # Calculate elapsed time
        elapsed_seconds = time.time() - self.training_start_time if self.training_start_time else 0

        # Format elapsed time nicely
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)

        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        # Print to console
        print(f"\nTraining complete! Total time: {time_str}")

        # Update UI
        self.is_training = False
        self.start_button.Enable(True)
        self.stop_button.Enable(False)
        self.status_text.SetLabel(f"Training complete! Total time: {time_str}")

    def update_periodic(self, status_text, pred_probs, predictions):
        """
        Periodic UI updates (time-throttled, thread-safe with reentrant protection)

        Updates status label, predictions, and confidence histograms.
        Only executes if enough time has passed since last update.

        Args:
            status_text: Status text to display
            pred_probs: Prediction probabilities for histograms
            predictions: List of (img, target, pred) tuples for visualization
        """
        current_time = time.time()

        # Time-based throttling
        if (current_time - self._last_periodic_update_time) < self._periodic_update_interval:
            return

        wx.CallAfter(self._update_periodic, status_text, pred_probs, predictions, current_time)

    def _update_periodic(self, status_text, pred_probs, predictions, current_time):
        """Internal periodic update - skips if already in progress"""
        # Reentrant protection
        if self._periodic_update_in_progress:
            return

        try:
            self._periodic_update_in_progress = True
            self._last_periodic_update_time = current_time

            # Update status label
            self.status_text.SetLabel(status_text)

            # Update predictions
            if predictions is not None:
                for i, (img, target, pred) in enumerate(predictions[:3]):
                    if i < len(self.pred_panels):
                        pil_img = visualize_prediction(img, target, pred)
                        self.pred_panels[i].update_image(pil_img)

            # Update confidence histograms
            if pred_probs is not None:
                self._update_histograms(pred_probs)

        finally:
            self._periodic_update_in_progress = False

    def update_per_epoch(self, epoch, total_epochs):
        """
        Per-epoch UI updates (thread-safe with reentrant protection)

        Updates progress bar and triggers graph redraws.
        Called once per epoch - no throttling needed.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
        """
        wx.CallAfter(self._update_per_epoch, epoch, total_epochs)

    def _update_per_epoch(self, epoch, total_epochs):
        """Internal per-epoch update - skips if already in progress"""
        # Reentrant protection
        if self._epoch_update_in_progress:
            return

        try:
            self._epoch_update_in_progress = True

            # Update progress bar
            progress = int((epoch / total_epochs) * 100)
            self.progress.SetValue(progress)

        finally:
            self._epoch_update_in_progress = False

    def accumulate_batch_data(self, loss, loss_components):
        """
        Accumulate batch data without UI update (called from training thread)

        Args:
            loss: Batch loss value
            loss_components: Dict with BCE/Dice loss components
        """
        # Track batch loss
        self.batch_losses.append((self.batch_counter, loss))
        self.batch_counter += 1

        # Accumulate loss components for epoch averaging
        self.current_epoch_bce_nuclei.append(loss_components['bce_nuclei'])
        self.current_epoch_bce_mito.append(loss_components['bce_mito'])
        self.current_epoch_dice_nuclei.append(loss_components['dice_nuclei'])
        self.current_epoch_dice_mito.append(loss_components['dice_mito'])

    def update_status(self, status):
        """Update status text (thread-safe)"""
        wx.CallAfter(self._update_status, status)

    def _update_status(self, status):
        self.status_text.SetLabel(status)

    def add_loss_point(self, epoch, train_loss, val_loss=None):
        """Add a loss data point (thread-safe)"""
        wx.CallAfter(self._add_loss_point, epoch, train_loss, val_loss)

    def _add_loss_point(self, epoch, train_loss, val_loss):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

    def add_metrics_point(self, epoch, nuclei_iou, mito_iou):
        """Add metrics data point (thread-safe)"""
        wx.CallAfter(self._add_metrics_point, epoch, nuclei_iou, mito_iou)

    def _add_metrics_point(self, epoch, nuclei_iou, mito_iou):
        # Store both the epoch number and the IoU values
        self.iou_epochs.append(epoch)
        self.nuclei_ious.append(nuclei_iou)
        self.mito_ious.append(mito_iou)

        # Update metrics graph (shows entire training run)
        if len(self.nuclei_ious) > 0:
            nuclei_data = list(zip(self.iou_epochs, self.nuclei_ious))
            mito_data = list(zip(self.iou_epochs, self.mito_ious))

            nuclei_line = wxplot.PolyLine(nuclei_data, colour='red', width=2, legend='Nuclei IoU')
            mito_line = wxplot.PolyLine(mito_data, colour='cyan', width=2, legend='Mito IoU')

            graphics = wxplot.PlotGraphics([nuclei_line, mito_line], 'IoU Metrics', 'Epoch', 'IoU')
            self.metrics_canvas.Draw(graphics)

    def update_predictions(self, sample_predictions):
        """Update prediction visualizations at validation time (thread-safe)"""
        wx.CallAfter(self._update_predictions, sample_predictions)

    def _update_predictions(self, sample_predictions):
        """Update predictions from validation samples"""
        for i, (img, target, pred) in enumerate(sample_predictions[:3]):
            if i < len(self.pred_panels):
                pil_img = visualize_prediction(img, target, pred)
                self.pred_panels[i].update_image(pil_img)

    def finalize_epoch_loss_components(self, epoch, train_loss):
        """Finalize loss components at epoch end (thread-safe)"""
        wx.CallAfter(self._finalize_epoch_loss_components, epoch, train_loss)

    def _finalize_epoch_loss_components(self, epoch, train_loss):
        # Average the accumulated values for this epoch
        if len(self.current_epoch_bce_nuclei) > 0:
            self.epoch_bce_nuclei.append(np.mean(self.current_epoch_bce_nuclei))
            self.epoch_bce_mito.append(np.mean(self.current_epoch_bce_mito))
            self.epoch_dice_nuclei.append(np.mean(self.current_epoch_dice_nuclei))
            self.epoch_dice_mito.append(np.mean(self.current_epoch_dice_mito))
            self.epoch_total_loss.append(train_loss)

            # Clear accumulators for next epoch
            self.current_epoch_bce_nuclei = []
            self.current_epoch_bce_mito = []
            self.current_epoch_dice_nuclei = []
            self.current_epoch_dice_mito = []

            # Update graph
            self._update_loss_components_graph()

    def _update_loss_components_graph(self):
        """Update the loss components graph including total loss"""
        if len(self.epoch_bce_nuclei) > 0:
            # Create x-axis data (epochs)
            epochs_x = list(range(len(self.epoch_bce_nuclei)))

            # Total loss - thick gray line
            total_data = list(zip(epochs_x, self.epoch_total_loss))
            total_loss_line = wxplot.PolyLine(total_data, colour='gray', width=3, legend='Total Loss')

            # BCE losses (solid lines)
            bce_nuclei_data = list(zip(epochs_x, self.epoch_bce_nuclei))
            bce_mito_data = list(zip(epochs_x, self.epoch_bce_mito))
            bce_nuclei_line = wxplot.PolyLine(bce_nuclei_data, colour='red', width=2, legend='BCE Nuclei')
            bce_mito_line = wxplot.PolyLine(bce_mito_data, colour='cyan', width=2, legend='BCE Mito')

            # Dice losses (dotted lines)
            dice_nuclei_data = list(zip(epochs_x, self.epoch_dice_nuclei))
            dice_mito_data = list(zip(epochs_x, self.epoch_dice_mito))
            dice_nuclei_line = wxplot.PolyLine(dice_nuclei_data, colour='red', width=1,
                                               legend='Dice Nuclei', style=wx.PENSTYLE_DOT)
            dice_mito_line = wxplot.PolyLine(dice_mito_data, colour='cyan', width=1,
                                             legend='Dice Mito', style=wx.PENSTYLE_DOT)

            lines = [total_loss_line, bce_nuclei_line, bce_mito_line, dice_nuclei_line, dice_mito_line]
            graphics = wxplot.PlotGraphics(lines, 'Loss Components', 'Epoch', 'Loss')
            self.loss_comp_canvas.Draw(graphics)

    def _update_histograms(self, pred_probs):
        """Compute and display histograms from prediction probabilities"""
        # pred_probs: (B, 2, H, W) - batch of predictions
        # Extract nuclei (channel 0) and mito (channel 1) predictions
        nuclei_probs = pred_probs[:, 0].flatten().cpu().numpy()
        mito_probs = pred_probs[:, 1].flatten().cpu().numpy()

        # Compute histograms (10 bins from 0 to 1)
        nuclei_counts, _ = np.histogram(nuclei_probs, bins=10, range=(0, 1))
        mito_counts, _ = np.histogram(mito_probs, bins=10, range=(0, 1))

        # Normalize to percentages
        nuclei_counts = nuclei_counts / nuclei_counts.sum() * 100
        mito_counts = mito_counts / mito_counts.sum() * 100

        # Store
        self.nuclei_hist_bins = nuclei_counts.tolist()
        self.mito_hist_bins = mito_counts.tolist()

        # Draw nuclei histogram using PolyHistogram
        bins = np.linspace(0, 1, 11)  # 11 edges for 10 bins

        hist = wxplot.PolyHistogram(
            np.array(self.nuclei_hist_bins),
            bins,
            fillcolour=wx.RED,
            edgecolour=wx.RED,
            edgewidth=1
        )

        graphics = wxplot.PlotGraphics([hist], 'Nuclei Confidence', 'Probability', 'Percentage')
        max_height = max(self.nuclei_hist_bins) if max(self.nuclei_hist_bins) > 0 else 1
        self.nuclei_hist_canvas.Draw(graphics, xAxis=(0, 1), yAxis=(0, max_height * 1.1))

        # Draw mito histogram using PolyHistogram
        hist = wxplot.PolyHistogram(
            np.array(self.mito_hist_bins),
            bins,
            fillcolour=wx.CYAN,
            edgecolour=wx.CYAN,
            edgewidth=1
        )

        graphics = wxplot.PlotGraphics([hist], 'Mito Confidence', 'Probability', 'Percentage')
        max_height = max(self.mito_hist_bins) if max(self.mito_hist_bins) > 0 else 1
        self.mito_hist_canvas.Draw(graphics, xAxis=(0, 1), yAxis=(0, max_height * 1.1))


class GUICallbacks(TrainingCallbacks):
    """Callbacks that update the GUI during training"""

    def __init__(self, training_panel):
        self.panel = training_panel
        self.last_prediction_batch = -50  # Track when we last updated predictions
        self.fixed_samples = None  # Cache fixed samples for consistent visualization

    def on_training_start(self, total_epochs, model_params):
        self.panel.update_status(f"Training started: {model_params:,} parameters, {total_epochs} epochs")

    def on_epoch_start(self, epoch, total_epochs):
        # Just update status at epoch start
        self.panel.update_status(f"Epoch {epoch + 1}/{total_epochs}")

    def on_batch_end(self, epoch, batch_idx, total_batches, loss, loss_components=None, pred_probs=None):
        # Always accumulate batch data (needed for epoch averaging)
        if loss_components:
            self.panel.accumulate_batch_data(loss, loss_components)

        # Check if periodic update will actually happen (to avoid wasted computation)
        current_time = time.time()
        if (current_time - self.panel._last_periodic_update_time) < self.panel._periodic_update_interval:
            return  # Skip expensive UI updates, throttled

        # Prepare status text
        elapsed_seconds = time.time() - self.panel.training_start_time
        elapsed_time = ':'.join([str(int(elapsed_seconds/60/60 % 60)),
                                 str(int(elapsed_seconds/60 % 60)),
                                 str(int(elapsed_seconds%60))])
        status_text = f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{total_batches}, Loss: {loss:.4f}, Time: {elapsed_time}"

        # Generate live predictions (expensive - only when update will happen)
        predictions = self._get_live_predictions()

        # Periodic update (will accept since we already checked timing)
        self.panel.update_periodic(status_text, pred_probs, predictions)

    def on_epoch_end(self, epoch, train_loss, val_loss=None, metrics=None):
        # Add epoch data points
        self.panel.add_loss_point(epoch, train_loss, val_loss)

        # Finalize loss components for this epoch
        self.panel.finalize_epoch_loss_components(epoch, train_loss)

        if metrics:
            self.panel.add_metrics_point(epoch, metrics['nuclei_iou'], metrics['mito_iou'])

        # Per-epoch UI update
        total_epochs = self.panel.trainer.config.num_epochs
        self.panel.update_per_epoch(epoch + 1, total_epochs)

    def on_validation_end(self, epoch, val_loss, metrics, sample_predictions):
        # Update predictions
        if sample_predictions:
            self.panel.update_predictions(sample_predictions)

    def on_checkpoint_saved(self, epoch, checkpoint_type, path):
        if checkpoint_type == 'best':
            self.panel.update_status(f"New best model saved at epoch {epoch}!")

    def on_training_end(self, final_epoch, best_val_loss):
        self.panel.update_status(f"Training complete! Best val loss: {best_val_loss:.4f}")
        # Final progress bar update
        wx.CallAfter(lambda: self.panel.progress.SetValue(100))

    def _get_live_predictions(self):
        """
        Generate predictions on validation samples

        Returns:
            List of (img, target, pred) tuples, or None if not available
        """
        # Get trainer from panel
        trainer = self.panel.trainer
        if trainer is None:
            return None

        # Get or cache fixed samples (same 3 every time)
        if self.fixed_samples is None:
            self.fixed_samples = []
            for i in range(3):
                img, target = trainer.val_dataset[i]
                self.fixed_samples.append((img, target))

        # Run inference on fixed samples
        trainer.model.eval()
        with torch.no_grad():
            predictions = []
            for img, target in self.fixed_samples:
                # Add batch dimension
                img_batch = img.unsqueeze(0).to(trainer.config.device)
                # Predict
                pred = trainer.model(img_batch)
                # Remove batch dimension and move to CPU
                predictions.append((img, target, pred[0].cpu()))

        return predictions


class MainFrame(wx.Frame):
    """Main application frame with training dashboard"""

    def __init__(self):
        super().__init__(None, title="Joe Net 1 - Training Dashboard", size=(1600, 1004))

        # Create menu bar
        menubar = wx.MenuBar()
        file_menu = wx.Menu()

        quit_item = file_menu.Append(wx.ID_EXIT, "&Quit\tCmd-Q", "Quit the application")
        self.Bind(wx.EVT_MENU, self.on_quit, quit_item)

        menubar.Append(file_menu, "&File")
        self.SetMenuBar(menubar)

        # Create training panel
        self.training_panel = TrainingPanel(self)

        # Center on screen
        self.Centre()

    def on_quit(self, event):
        """Quit the application"""
        self.Close(True)


def main():
    app = wx.App()
    frame = MainFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()


