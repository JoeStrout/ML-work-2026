#!/usr/bin/env python3
"""
Joe Net 1 - My first attempt at training a segmentation and semantic labeling
network, using a wxWindows GUI for visualization.
"""

import wx
from cloudvolume import CloudVolume, Bbox
from PIL import Image
import numpy as np
import random

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


def load_volumes(mip=0):
    """
    Load the full training volumes for all three layers.

    Args:
        mip: MIP level to use (0 for full resolution)

    Returns:
        tuple: (img_volume, nuclei_volume, mito_volume) as numpy arrays
    """
    global volume_img, volume_nuclei, volume_mito

    # Create bounding box for the training volume
    scale_factor = 2 ** mip
    min_x = int(VOL_MIN_X / scale_factor)
    min_y = int(VOL_MIN_Y / scale_factor)
    min_z = int(VOL_MIN_Z / scale_factor)
    max_x = int(VOL_MAX_X / scale_factor)
    max_y = int(VOL_MAX_Y / scale_factor)
    max_z = int(VOL_MAX_Z / scale_factor)

    vol_bbox = Bbox((min_x, min_y, min_z), (max_x, max_y, max_z))

    print(f"Loading training volume bbox: {vol_bbox}")
    print(f"Volume size: {max_x - min_x} x {max_y - min_y} x {max_z - min_z}")

    # Load image volume
    print(f"\nLoading image volume from: {CLOUDVOLUME_PATH_IMG}")
    vol_img = CloudVolume(CLOUDVOLUME_PATH_IMG, progress=False)
    vol_img.fill_missing = True
    img_data = vol_img.download(vol_bbox, mip=mip)
    print(f"Image data shape: {img_data.shape}, dtype: {img_data.dtype}")

    # Extract and normalize to uint8
    volume_img = img_data[:, :, :, 0]  # Remove channel dimension
    if volume_img.dtype != np.uint8:
        if volume_img.max() > volume_img.min():
            volume_img = ((volume_img - volume_img.min()) / (volume_img.max() - volume_img.min()) * 255).astype(np.uint8)
        else:
            volume_img = np.zeros_like(volume_img, dtype=np.uint8)
    print(f"Image volume stored as: {volume_img.shape}, {volume_img.dtype}")

    # Load nuclei volume
    print(f"\nLoading nuclei volume from: {CLOUDVOLUME_PATH_NUCLEI}")
    vol_nuclei_cv = CloudVolume(CLOUDVOLUME_PATH_NUCLEI, progress=False)
    vol_nuclei_cv.fill_missing = True
    nuclei_data = vol_nuclei_cv.download(vol_bbox, mip=mip)
    print(f"Nuclei data shape: {nuclei_data.shape}, dtype: {nuclei_data.dtype}")

    # Convert to binary mask (uint8)
    volume_nuclei = (nuclei_data[:, :, :, 0] > 0).astype(np.uint8)
    print(f"Nuclei volume stored as: {volume_nuclei.shape}, {volume_nuclei.dtype}")

    # Load mito volume
    print(f"\nLoading mito volume from: {CLOUDVOLUME_PATH_MITO}")
    vol_mito_cv = CloudVolume(CLOUDVOLUME_PATH_MITO, progress=False)
    vol_mito_cv.fill_missing = True
    mito_data = vol_mito_cv.download(vol_bbox, mip=mip)
    print(f"Mito data shape: {mito_data.shape}, dtype: {mito_data.dtype}")

    # Convert to binary mask (uint8)
    volume_mito = (mito_data[:, :, :, 0] > 0).astype(np.uint8)
    print(f"Mito volume stored as: {volume_mito.shape}, {volume_mito.dtype}")

    # Calculate total memory usage
    total_bytes = volume_img.nbytes + volume_nuclei.nbytes + volume_mito.nbytes
    total_mb = total_bytes / (1024 * 1024)
    print(f"\nTotal memory used: {total_mb:.2f} MB")

    return volume_img, volume_nuclei, volume_mito


def extract_slice(min_x, min_y, z, size=256):
    """
    Extract a slice from the loaded volumes.

    Args:
        min_x: Top-left X coordinate (in original world coordinates)
        min_y: Top-left Y coordinate (in original world coordinates)
        z: Z coordinate (in original world coordinates)
        size: Size of the square slice (default 256x256)

    Returns:
        tuple: (img_slice, nuclei_slice, mito_slice) as 2D numpy arrays
    """
    global volume_img, volume_nuclei, volume_mito

    if volume_img is None or volume_nuclei is None or volume_mito is None:
        raise RuntimeError("Volumes not loaded. Call load_volumes() first.")

    # Convert world coordinates to volume array indices
    vol_x = min_x - VOL_MIN_X
    vol_y = min_y - VOL_MIN_Y
    vol_z = z - VOL_MIN_Z

    # Extract the slices
    img_slice = volume_img[vol_x:vol_x + size, vol_y:vol_y + size, vol_z]
    nuclei_slice = volume_nuclei[vol_x:vol_x + size, vol_y:vol_y + size, vol_z]
    mito_slice = volume_mito[vol_x:vol_x + size, vol_y:vol_y + size, vol_z]

    return img_slice, nuclei_slice, mito_slice


def composite_images(img_data, nuclei_data, mito_data):
    """
    Composite the image, nuclei, and mito layers.

    Args:
        img_data: Grayscale image data (2D array)
        nuclei_data: Nuclei layer data (2D array)
        mito_data: Mito layer data (2D array)

    Returns:
        PIL.Image: RGB composited image
    """
    # Start with grayscale image as base (convert to RGB)
    # Transpose so Y is vertical, X is horizontal
    # Use float32 to avoid overflow during calculations
    img_rgb = np.stack([img_data.T.astype(np.float32),
                        img_data.T.astype(np.float32),
                        img_data.T.astype(np.float32)], axis=2)

    # Create masks for nuclei and mito (nonzero areas)
    nuclei_mask = nuclei_data.T > 0
    mito_mask = mito_data.T > 0

    # Tint nuclei areas red (increase red channel, reduce others)
    img_rgb[nuclei_mask, 0] = np.minimum(255, img_rgb[nuclei_mask, 0] + 100)  # Boost red
    img_rgb[nuclei_mask, 1] = img_rgb[nuclei_mask, 1] * 0.5  # Reduce green
    img_rgb[nuclei_mask, 2] = img_rgb[nuclei_mask, 2] * 0.5  # Reduce blue

    # Tint mito areas cyan (increase green and blue, reduce red)
    img_rgb[mito_mask, 0] = img_rgb[mito_mask, 0] * 0.5  # Reduce red
    img_rgb[mito_mask, 1] = np.minimum(255, img_rgb[mito_mask, 1] + 100)  # Boost green
    img_rgb[mito_mask, 2] = np.minimum(255, img_rgb[mito_mask, 2] + 100)  # Boost blue

    # Clamp all values to [0, 255] and convert to uint8
    img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    # Convert to PIL Image
    pil_image = Image.fromarray(img_rgb)

    return pil_image


class ImagePanel(wx.Panel):
    """Panel that displays an image at fixed size."""

    def __init__(self, parent, pil_image):
        super().__init__(parent)
        self.pil_image = pil_image
        self.wx_bitmap = None

        # Set fixed size to match image size
        self.SetMinSize((pil_image.size[0], pil_image.size[1]))
        self.SetMaxSize((pil_image.size[0], pil_image.size[1]))

        # Convert PIL image to wx.Bitmap
        self.update_image(pil_image)

        # Bind events
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def update_image(self, pil_image):
        """Update the displayed image."""
        self.pil_image = pil_image
        # Convert PIL image to wxPython format
        wx_image = wx.Image(pil_image.size[0], pil_image.size[1])
        wx_image.SetData(pil_image.convert("RGB").tobytes())
        self.wx_bitmap = wx_image.ConvertToBitmap()
        self.Refresh()

    def on_paint(self, event):
        """Paint the image on the panel."""
        dc = wx.PaintDC(self)
        if self.wx_bitmap:
            # Draw bitmap at original size (no scaling)
            dc.DrawBitmap(self.wx_bitmap, 0, 0, True)


class MainFrame(wx.Frame):
    """Main application frame."""

    def __init__(self):
        super().__init__(None, title="Joe Net 1 - Training Data Viewer", size=(800, 600))

        # Store current slice position
        self.current_slice_x = None
        self.current_slice_y = None
        self.current_slice_z = None

        # Create status bar
        self.CreateStatusBar()
        self.SetStatusText("Ready")

        # Create menu bar
        menubar = wx.MenuBar()
        file_menu = wx.Menu()

        # Add Quit menu item with cmd-Q shortcut
        quit_item = file_menu.Append(wx.ID_EXIT, "&Quit\tCmd-Q", "Quit the application")
        self.Bind(wx.EVT_MENU, self.on_quit, quit_item)

        menubar.Append(file_menu, "&File")
        self.SetMenuBar(menubar)

        # Load the full training volumes
        load_volumes(mip=0)

        # Create main panel
        panel = wx.Panel(self)

        # Create vertical sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Title label
        self.title_label = wx.StaticText(panel, label="Loading...")
        font = self.title_label.GetFont()
        font.PointSize += 2
        font = font.Bold()
        self.title_label.SetFont(font)
        main_sizer.Add(self.title_label, 0, wx.ALL | wx.CENTER, 10)

        # Create a dummy image panel (will be updated with actual data)
        dummy_image = Image.new('RGB', (SLICE_SIZE, SLICE_SIZE), color='gray')
        self.image_panel = ImagePanel(panel, dummy_image)
        main_sizer.Add(self.image_panel, 0, wx.ALL | wx.CENTER, 5)

        # Create button panel
        button_panel = wx.Panel(panel)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Add dice button
        dice_button = wx.Button(button_panel, label="🎲 Random Slice")
        dice_button.Bind(wx.EVT_BUTTON, self.on_random_slice)
        button_sizer.Add(dice_button, 0, wx.ALL, 5)

        button_panel.SetSizer(button_sizer)
        main_sizer.Add(button_panel, 0, wx.ALL | wx.CENTER, 5)

        # Set sizer
        panel.SetSizer(main_sizer)

        # Center on screen
        self.Centre()

        # Load initial random slice
        self.on_random_slice(None)

    def on_random_slice(self, event):
        """Pick a random slice and update the display."""
        # Pick random coordinates within valid bounds
        # Ensure we have room for a full 256x256 slice
        max_x = VOL_MAX_X - SLICE_SIZE
        max_y = VOL_MAX_Y - SLICE_SIZE
        max_z = VOL_MAX_Z - 1

        slice_x = random.randint(VOL_MIN_X, max_x)
        slice_y = random.randint(VOL_MIN_Y, max_y)
        slice_z = random.randint(VOL_MIN_Z, max_z)

        # Store current position
        self.current_slice_x = slice_x
        self.current_slice_y = slice_y
        self.current_slice_z = slice_z

        print(f"\nExtracting slice at ({slice_x}, {slice_y}, {slice_z})")

        # Extract slices
        img_data, nuclei_data, mito_data = extract_slice(slice_x, slice_y, slice_z, size=SLICE_SIZE)

        # Composite the images
        pil_image = composite_images(img_data, nuclei_data, mito_data)

        # Update the image panel
        self.image_panel.update_image(pil_image)

        # Update the title
        self.title_label.SetLabel(f"XY Slice at ({slice_x}, {slice_y}, {slice_z})")

        # Update status bar
        self.SetStatusText(f"Displaying slice at ({slice_x}, {slice_y}, {slice_z})")

    def on_quit(self, event):
        """Handle quit menu command."""
        self.Close()


def main():
    app = wx.App()
    frame = MainFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
