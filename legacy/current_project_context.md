# Project Context: Pix2Pix GAN Echocardiography Segmentation

## 1. Overview & Context

This project is a full implementation of the paper *"Automatic Segmentation of 2-D Echocardiography Ultrasound Images by Means of Generative Adversarial Network"* (Fatima et al., IEEE TUFFC 2024). It uses a Pix2Pix Generative Adversarial Network (GAN) to automatically segment key cardiac structures from echocardiography images.

**Key details:**
- **Dataset**: CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentations) public dataset (NIfTI format).
- **Target Views**: 2-Chamber (2CH) and 4-Chamber (4CH).
- **Target Phases**: End-Diastole (ED) and End-Systole (ES).
- **Structures segmented**: Left Ventricle endocardium (LVendo), Left Ventricle myocardium (LVmyo), and Left Atrium (LA).
- **Architecture**: 
  - **Generator**: 8-level U-Net encoder-decoder with skip connections.
  - **Discriminator**: PatchGAN (70x70 patches).
- **Loss Functions**: Binary Cross-Entropy Loss, Pixel L1 Loss, and Dice Loss for direct segmentation optimization.

## 2. Directory Structure & File Details

- `config.py`: Central configuration and hyperparameters.
- `requirements.txt`: Python dependencies.
- `predict.py`: Custom inference script for new images, folders, or videos.
- `export_model.py`: Script to export the trained model for use on another system.
- `walkthrough.md`: Quick reference guide.
- **`data/`**: Data loading and preprocessing.
  - `dataset.py`: CAMUS NIfTI loader and split management.
  - `preprocessing.py`: Resizing, normalization, and data augmentation.
- **`models/`**: Neural network architectures.
  - `generator.py`: U-Net generator model.
  - `discriminator.py`: PatchGAN discriminator model.
  - `pix2pix_gan.py`: The full GAN assembly and training step logic.
- **`training/`**: Loss definitions and training loop.
  - `losses.py`: Implementation of BCE, L1, Generator, Discriminator, and Dice losses.
  - `train.py`: Main training loop with early stopping, validation, and learning rate scheduling.
- **`evaluation/`**: Metrics and testing.
  - `metrics.py`: Calculation of Dice, Mean Absolute Difference (MAD), Hausdorff Distance (HD), and Left Ventricular Ejection Fraction (LVEF).
  - `evaluate.py`: Evaluation inference and CSV export script.
- **`utils/`**: Utilities.
  - `visualization.py`: Functions to plot loss curves, segmentation overlays, and scatter plots.

---

## 3. Full Codebase Context

Below is the exact and complete source code for every file in the project. This can be provided to any AI to instantly give it a complete understanding of the project's state.

### `requirements.txt`
```text
tensorflow>=2.10
numpy
matplotlib
scipy
scikit-image
nibabel
pandas
opencv-python
```

### `config.py`
```python
"""
Configuration file for the Pix2Pix GAN Echocardiography Segmentation project.

All hyperparameters are set according to the paper:
Fatima et al., "Automatic Segmentation of 2-D Echocardiography Ultrasound
Images by Means of Generative Adversarial Network", IEEE TUFFC, 2024.

Dataset: CAMUS_public (NIfTI format, .nii.gz files)

IMPROVED: Tuned for stable GAN training and better Dice scores.
"""

import os

# ==============================================================================
# Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CAMUS dataset location (NIfTI format) ---
DATASET_DIR = r"C:\Users\tejas\Desktop\code\projects\DATASET\PRINC\CAMUS_public"
DATA_DIR = os.path.join(DATASET_DIR, "database_nifti")
SPLIT_DIR = os.path.join(DATASET_DIR, "database_split")

# Split text files (provided by CAMUS)
TRAIN_SPLIT_FILE = os.path.join(SPLIT_DIR, "subgroup_training.txt")
TEST_SPLIT_FILE = os.path.join(SPLIT_DIR, "subgroup_testing.txt")
VAL_SPLIT_FILE = os.path.join(SPLIT_DIR, "subgroup_validation.txt")

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Create output directories if they don't exist
for directory in [CHECKPOINT_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==============================================================================
# Dataset Configuration
# ==============================================================================
VIEWS = ["2CH", "4CH"]
PHASES = ["ED", "ES"]

STRUCTURE_LABELS = {
    "LVendo": 1,
    "LVmyo": 2,
    "LA": 3,
}
STRUCTURES = list(STRUCTURE_LABELS.keys())

# ==============================================================================
# Image Configuration
# ==============================================================================
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
OUTPUT_CHANNELS = 1

# ==============================================================================
# Model Hyperparameters (IMPROVED from paper defaults)
# ==============================================================================
PATCH_SIZE = 70
BATCH_SIZE = 1
EPOCHS = 300               # More epochs, but early stopping prevents waste

# Learning rates — D gets HALF the LR of G to prevent dominance
LEARNING_RATE_G = 0.0002   # Generator learning rate
LEARNING_RATE_D = 0.0001   # Discriminator learning rate (halved!)
BETA_1 = 0.5
BETA_2 = 0.999

# Loss weights
LAMBDA_PIXEL = 100         # L1 pixel loss weight (Eq. 3)
LAMBDA_DICE = 50           # Dice loss weight (NEW — directly optimizes Dice)

# Training stability
LABEL_SMOOTHING = 0.9      # Real labels → 0.9 instead of 1.0
D_THROTTLE_THRESHOLD = 0.3 # Skip D update when D_loss < this value
NOISE_LABELS_PROB = 0.05   # Probability of flipping real/fake labels (noise)

# Data augmentation
AUGMENT_TRAIN = True       # Enable augmentation during training
AUG_FLIP_H = True          # Random horizontal flip
AUG_FLIP_V = False         # Random vertical flip (disabled — anatomy matters)
AUG_ROTATE_MAX = 10        # Max random rotation in degrees
AUG_BRIGHTNESS = 0.1       # Random brightness jitter
AUG_CONTRAST = 0.1         # Random contrast jitter

# Generator (UNET) architecture
GENERATOR_FILTERS = [64, 128, 256, 512, 512, 512, 512, 512]

# Discriminator (PatchGAN) architecture
DISCRIMINATOR_FILTERS = [64, 128, 256, 512]

# ==============================================================================
# Training Configuration
# ==============================================================================
SAVE_INTERVAL = 25
DISPLAY_INTERVAL = 5
SAVE_BEST_ONLY = True
VALIDATE_INTERVAL = 10     # Run validation Dice every N epochs
EARLY_STOP_PATIENCE = 60   # Stop if no Dice improvement for N epochs
LR_REDUCE_PATIENCE = 30    # Halve LR if no improvement for N epochs
LR_REDUCE_FACTOR = 0.5     # Multiply LR by this factor
MIN_LR = 1e-6              # Minimum learning rate floor
```

### `data/dataset.py`
```python
"""
CAMUS dataset loader for echocardiography segmentation.

Loads NIfTI (.nii.gz) image pairs from the CAMUS_public dataset,
extracts binary masks per cardiac structure, and provides
train/test/val splits using the official CAMUS split text files.
"""

import os
import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None
    print("WARNING: nibabel not installed. Install via: pip install nibabel")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def load_nifti_image(filepath):
    """Load a .nii.gz NIfTI image file and return as a numpy array."""
    if nib is None:
        raise ImportError("nibabel is required to load NIfTI files. "
                          "Install via: pip install nibabel")
    img = nib.load(filepath)
    array = np.array(img.dataobj, dtype=np.float32)

    # CAMUS NIfTI images may be (H, W, 1) or (H, W); squeeze to 2-D
    array = np.squeeze(array)

    # If still 3-D, take the first slice
    if array.ndim == 3:
        array = array[:, :, 0]

    return array


def extract_binary_mask(gt_array, structure_label):
    """Extract a binary mask for a specific cardiac structure."""
    return (gt_array == structure_label).astype(np.float32)


def get_patient_ids(split="train"):
    """Return patient IDs for the given split using official CAMUS split files."""
    if split == "train":
        split_file = config.TRAIN_SPLIT_FILE
    elif split == "test":
        split_file = config.TEST_SPLIT_FILE
    elif split == "val":
        split_file = config.VAL_SPLIT_FILE
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'test', or 'val'.")

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, 'r') as f:
        patient_ids = [line.strip() for line in f.readlines() if line.strip()]

    return patient_ids


def get_image_path(patient_id, view, phase, gt=False):
    """Construct the file path for a CAMUS NIfTI image or ground truth."""
    suffix = "_gt" if gt else ""
    filename = f"{patient_id}_{view}_{phase}{suffix}.nii.gz"
    return os.path.join(config.DATA_DIR, patient_id, filename)


def load_dataset(view="2CH", phase="ED", structure="LVendo", split="train"):
    """Load all images and corresponding binary masks for a given configuration."""
    patient_ids = get_patient_ids(split)
    structure_label = config.STRUCTURE_LABELS[structure]

    images = []
    masks = []
    skipped = 0

    for pid in patient_ids:
        img_path = get_image_path(pid, view, phase, gt=False)
        gt_path = get_image_path(pid, view, phase, gt=True)

        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            skipped += 1
            continue

        img = load_nifti_image(img_path)
        gt = load_nifti_image(gt_path)
        mask = extract_binary_mask(gt, structure_label)

        images.append(img)
        masks.append(mask)

    if skipped > 0:
        print(f"WARNING: Skipped {skipped} patients (files not found).")

    print(f"Loaded {len(images)} samples for {split} "
          f"[view={view}, phase={phase}, structure={structure}]")

    return images, masks


def load_train_test(view="2CH", structure="LVendo"):
    """Load combined ED+ES training and test data for a view and structure."""
    train_images, train_masks = [], []
    test_images, test_masks = [], []

    for phase in config.PHASES:
        # Training data
        imgs, msks = load_dataset(view, phase, structure, split="train")
        train_images.extend(imgs)
        train_masks.extend(msks)

        # Test data
        imgs, msks = load_dataset(view, phase, structure, split="test")
        test_images.extend(imgs)
        test_masks.extend(msks)

    print(f"\nTotal training samples: {len(train_images)}")
    print(f"Total test samples:     {len(test_images)}")

    return train_images, train_masks, test_images, test_masks
```

### `data/preprocessing.py`
```python
"""
Preprocessing utilities for echocardiography images and masks — IMPROVED.

Handles resizing, normalization, and data augmentation
to prepare inputs for the Pix2Pix GAN.
"""

import numpy as np

try:
    from skimage.transform import resize as sk_resize, rotate as sk_rotate
except ImportError:
    sk_resize = None
    sk_rotate = None
    print("WARNING: scikit-image not installed.")

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def resize_image(image, height=config.IMG_HEIGHT, width=config.IMG_WIDTH):
    """Resize a 2-D image to the target dimensions."""
    if sk_resize is None:
        raise ImportError("scikit-image is required for resizing.")
    return sk_resize(image, (height, width),
                     mode='reflect', anti_aliasing=True,
                     preserve_range=True).astype(np.float32)


def resize_mask(mask, height=config.IMG_HEIGHT, width=config.IMG_WIDTH):
    """Resize a binary mask using nearest-neighbor interpolation."""
    if sk_resize is None:
        raise ImportError("scikit-image is required for resizing.")
    resized = sk_resize(mask, (height, width),
                        order=0, mode='reflect',
                        anti_aliasing=False,
                        preserve_range=True).astype(np.float32)
    return (resized > 0.5).astype(np.float32)


def normalize_image(image):
    """Normalize image pixel values to [0, 1] range."""
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min == 0:
        return np.zeros_like(image)
    return (image - img_min) / (img_max - img_min)


def normalize_mask(mask):
    return mask.astype(np.float32)


def add_channel_dim(array):
    return np.expand_dims(array, axis=-1)


def preprocess_image(image):
    """Full preprocessing pipeline for an input image."""
    image = resize_image(image)
    image = normalize_image(image)
    image = add_channel_dim(image)
    return image


def preprocess_mask(mask):
    """Full preprocessing pipeline for a ground truth mask."""
    mask = resize_mask(mask)
    mask = normalize_mask(mask)
    mask = add_channel_dim(mask)
    return mask


def preprocess_dataset(images, masks):
    """Preprocess a list of images and masks into numpy arrays."""
    proc_images = np.array([preprocess_image(img) for img in images])
    proc_masks = np.array([preprocess_mask(m) for m in masks])
    return proc_images, proc_masks


# ===========================================================================
# DATA AUGMENTATION (IMPROVED)
# ===========================================================================

def augment_pair(image, mask):
    """Apply data augmentation to an image-mask pair."""
    img = image.copy()
    msk = mask.copy()

    # 1. Random horizontal flip
    if config.AUG_FLIP_H and np.random.random() > 0.5:
        img = np.flip(img, axis=1).copy()
        msk = np.flip(msk, axis=1).copy()

    # 2. Random vertical flip
    if config.AUG_FLIP_V and np.random.random() > 0.5:
        img = np.flip(img, axis=0).copy()
        msk = np.flip(msk, axis=0).copy()

    # 3. Random rotation
    if config.AUG_ROTATE_MAX > 0 and sk_rotate is not None:
        angle = np.random.uniform(-config.AUG_ROTATE_MAX, config.AUG_ROTATE_MAX)
        img_2d = img[:, :, 0]
        msk_2d = msk[:, :, 0]
        img_2d = sk_rotate(img_2d, angle, mode='reflect',
                           preserve_range=True).astype(np.float32)
        msk_2d = sk_rotate(msk_2d, angle, order=0, mode='reflect',
                           preserve_range=True).astype(np.float32)
        msk_2d = (msk_2d > 0.5).astype(np.float32)
        img = img_2d[:, :, np.newaxis]
        msk = msk_2d[:, :, np.newaxis]

    # 4. Random brightness jitter
    if config.AUG_BRIGHTNESS > 0:
        delta = np.random.uniform(-config.AUG_BRIGHTNESS, config.AUG_BRIGHTNESS)
        img = np.clip(img + delta, 0.0, 1.0)

    # 5. Random contrast jitter
    if config.AUG_CONTRAST > 0:
        factor = 1.0 + np.random.uniform(-config.AUG_CONTRAST, config.AUG_CONTRAST)
        mean_val = np.mean(img)
        img = np.clip((img - mean_val) * factor + mean_val, 0.0, 1.0)

    return img.astype(np.float32), msk.astype(np.float32)


def augment_dataset(images, masks):
    """Apply augmentation to an entire dataset (one pass)."""
    aug_imgs = []
    aug_msks = []
    for i in range(len(images)):
        a_img, a_msk = augment_pair(images[i], masks[i])
        aug_imgs.append(a_img)
        aug_msks.append(a_msk)
    return np.array(aug_imgs), np.array(aug_msks)
```

### `models/generator.py`
```python
"""
UNET Generator for the Pix2Pix GAN.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def downsample_block(filters, kernel_size=4, apply_batchnorm=True):
    block = tf.keras.Sequential()
    block.add(layers.Conv2D(
        filters, kernel_size, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    ))
    if apply_batchnorm:
        block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU(0.2))
    return block


def upsample_block(filters, kernel_size=4, apply_dropout=False):
    block = tf.keras.Sequential()
    block.add(layers.Conv2DTranspose(
        filters, kernel_size, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    ))
    block.add(layers.BatchNormalization())
    if apply_dropout:
        block.add(layers.Dropout(0.5))
    block.add(layers.ReLU())
    return block


def build_generator(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                    output_channels=config.OUTPUT_CHANNELS):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    down1 = downsample_block(64, apply_batchnorm=False)(inputs)
    down2 = downsample_block(128)(down1)
    down3 = downsample_block(256)(down2)
    down4 = downsample_block(512)(down3)
    down5 = downsample_block(512)(down4)
    down6 = downsample_block(512)(down5)
    down7 = downsample_block(512)(down6)
    down8 = downsample_block(512, apply_batchnorm=False)(down7)

    # Decoder
    up1 = upsample_block(512, apply_dropout=True)(down8)
    up1 = layers.Concatenate()([up1, down7])

    up2 = upsample_block(512, apply_dropout=True)(up1)
    up2 = layers.Concatenate()([up2, down6])

    up3 = upsample_block(512, apply_dropout=True)(up2)
    up3 = layers.Concatenate()([up3, down5])

    up4 = upsample_block(512)(up3)
    up4 = layers.Concatenate()([up4, down4])

    up5 = upsample_block(256)(up4)
    up5 = layers.Concatenate()([up5, down3])

    up6 = upsample_block(128)(up5)
    up6 = layers.Concatenate()([up6, down2])

    up7 = upsample_block(64)(up6)
    up7 = layers.Concatenate()([up7, down1])

    # Output layer
    output = layers.Conv2DTranspose(
        output_channels, kernel_size=4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid'
    )(up7)

    model = Model(inputs=inputs, outputs=output, name="UNET_Generator")
    return model
```

### `models/discriminator.py`
```python
"""
PatchGAN Discriminator for the Pix2Pix GAN.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def build_discriminator(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)):
    input_image = layers.Input(shape=input_shape, name="input_image")
    target_image = layers.Input(shape=input_shape, name="target_image")

    x = layers.Concatenate()([input_image, target_image])

    x = layers.Conv2D(
        64, kernel_size=4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02)
    )(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(
        128, kernel_size=4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(
        256, kernel_size=4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(
        512, kernel_size=4, strides=1, padding='valid',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.ZeroPadding2D()(x)
    output = layers.Conv2D(
        1, kernel_size=4, strides=1, padding='valid',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid'
    )(x)

    model = Model(inputs=[input_image, target_image], outputs=output,
                  name="PatchGAN_Discriminator")
    return model
```

### `models/pix2pix_gan.py`
```python
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from models.generator import build_generator
from models.discriminator import build_discriminator

class Pix2PixGAN:
    def __init__(self):
        self.generator = build_generator()
        self.discriminator = build_discriminator()

        self.gen_optimizer = Adam(
            learning_rate=config.LEARNING_RATE_G,
            beta_1=config.BETA_1, beta_2=config.BETA_2
        )
        self.disc_optimizer = Adam(
            learning_rate=config.LEARNING_RATE_D,
            beta_1=config.BETA_1, beta_2=config.BETA_2
        )

        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.lambda_pixel = config.LAMBDA_PIXEL
        self.lambda_dice = config.LAMBDA_DICE
        self.label_smoothing = config.LABEL_SMOOTHING

    @staticmethod
    def dice_loss(target, prediction):
        smooth = 1e-6
        target_flat = tf.reshape(target, [-1])
        pred_flat = tf.reshape(prediction, [-1])
        intersection = tf.reduce_sum(target_flat * pred_flat)
        dice = (2.0 * intersection + smooth) / (
            tf.reduce_sum(target_flat) + tf.reduce_sum(pred_flat) + smooth
        )
        return 1.0 - dice

    def discriminator_loss(self, disc_real_output, disc_fake_output):
        real_labels = tf.ones_like(disc_real_output) * self.label_smoothing
        fake_labels = tf.zeros_like(disc_fake_output)

        if config.NOISE_LABELS_PROB > 0:
            noise_mask = tf.random.uniform(tf.shape(real_labels)) < config.NOISE_LABELS_PROB
            real_labels = tf.where(noise_mask, 1.0 - real_labels, real_labels)
            noise_mask_f = tf.random.uniform(tf.shape(fake_labels)) < config.NOISE_LABELS_PROB
            fake_labels = tf.where(noise_mask_f, 1.0 - fake_labels, fake_labels)

        real_loss = self.bce_loss(real_labels, disc_real_output)
        fake_loss = self.bce_loss(fake_labels, disc_fake_output)
        total_loss = real_loss + fake_loss
        return total_loss, real_loss, fake_loss

    def generator_loss(self, disc_fake_output, gen_output, target):
        adversarial_loss = self.bce_loss(tf.ones_like(disc_fake_output), disc_fake_output)
        pixel_loss = tf.reduce_mean(tf.abs(target - gen_output))
        d_loss = self.dice_loss(target, gen_output)
        total_loss = adversarial_loss + self.lambda_pixel * pixel_loss + self.lambda_dice * d_loss
        return total_loss, adversarial_loss, pixel_loss, d_loss

    @tf.function
    def train_step_full(self, input_image, target_mask):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image, target_mask], training=True)
            disc_fake_output = self.discriminator([input_image, gen_output], training=True)

            disc_loss, d_real_loss, d_fake_loss = self.discriminator_loss(disc_real_output, disc_fake_output)
            gen_loss, g_adv_loss, g_pixel_loss, g_dice_loss = self.generator_loss(disc_fake_output, gen_output, target_mask)

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        disc_gradients, _ = tf.clip_by_global_norm(disc_gradients, 5.0)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gen_gradients, _ = tf.clip_by_global_norm(gen_gradients, 5.0)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return {
            "disc_loss": disc_loss, "disc_real_loss": d_real_loss, "disc_fake_loss": d_fake_loss,
            "gen_loss": gen_loss, "gen_adv_loss": g_adv_loss, "gen_pixel_loss": g_pixel_loss, "gen_dice_loss": g_dice_loss,
        }

    @tf.function
    def train_step_gen_only(self, input_image, target_mask):
        with tf.GradientTape() as gen_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image, target_mask], training=False)
            disc_fake_output = self.discriminator([input_image, gen_output], training=False)
            disc_loss, d_real_loss, d_fake_loss = self.discriminator_loss(disc_real_output, disc_fake_output)
            gen_loss, g_adv_loss, g_pixel_loss, g_dice_loss = self.generator_loss(disc_fake_output, gen_output, target_mask)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gen_gradients, _ = tf.clip_by_global_norm(gen_gradients, 5.0)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return {
            "disc_loss": disc_loss, "disc_real_loss": d_real_loss, "disc_fake_loss": d_fake_loss,
            "gen_loss": gen_loss, "gen_adv_loss": g_adv_loss, "gen_pixel_loss": g_pixel_loss, "gen_dice_loss": g_dice_loss,
        }

    def train_step(self, input_image, target_mask, update_discriminator=True):
        if update_discriminator: return self.train_step_full(input_image, target_mask)
        else: return self.train_step_gen_only(input_image, target_mask)

    def save_models(self, checkpoint_dir, epoch):
        gen_path = os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.h5")
        disc_path = os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch}.h5")
        self.generator.save_weights(gen_path)
        self.discriminator.save_weights(disc_path)

    def load_models(self, gen_path, disc_path=None):
        self.generator.load_weights(gen_path)
        if disc_path: self.discriminator.load_weights(disc_path)
```

### `training/losses.py`
```python
import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_real_loss(disc_real_output):
    return bce(tf.ones_like(disc_real_output), disc_real_output)

def discriminator_fake_loss(disc_fake_output):
    return bce(tf.zeros_like(disc_fake_output), disc_fake_output)

def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = discriminator_real_loss(disc_real_output)
    fake_loss = discriminator_fake_loss(disc_fake_output)
    return real_loss + fake_loss, real_loss, fake_loss

def adversarial_loss(disc_fake_output):
    return bce(tf.ones_like(disc_fake_output), disc_fake_output)

def pixel_loss(target, generated):
    return tf.reduce_mean(tf.abs(target - generated))

def dice_loss(target, prediction):
    smooth = 1e-6
    target_flat = tf.reshape(target, [-1])
    pred_flat = tf.reshape(prediction, [-1])
    intersection = tf.reduce_sum(target_flat * pred_flat)
    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(target_flat) + tf.reduce_sum(pred_flat) + smooth
    )
    return 1.0 - dice

def generator_loss(disc_fake_output, gen_output, target, lambda_pixel=100, lambda_dice=50):
    adv_loss = adversarial_loss(disc_fake_output)
    pix_loss = pixel_loss(target, gen_output)
    d_loss = dice_loss(target, gen_output)
    total_loss = adv_loss + lambda_pixel * pix_loss + lambda_dice * d_loss
    return total_loss, adv_loss, pix_loss, d_loss
```

### `training/train.py`
```python
"""
Training script for the Pix2Pix GAN — IMPROVED VERSION v2.

Key improvements:
    - Discriminator throttling (per-batch, based on running D_loss)
    - Validation Dice computed every N epochs
    - Best model saved by Dice score (not just G_loss)
    - Learning rate scheduling on plateau
    - Early stopping when Dice stops improving (FIXED: separate counters)
    - Automatic saturation detection (stops when model stops learning)
    - Data augmentation applied per epoch

Usage:
    python training/train.py --view 2CH --structure LVendo --epochs 300
"""

import os
import sys
import argparse
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorflow as tf

# GPU memory growth — prevents TF from allocating all VRAM at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
else:
    print("WARNING: No GPU found. Training will be slow on CPU.")

import config
from data.dataset import load_train_test
from data.preprocessing import preprocess_dataset, augment_dataset
from models.pix2pix_gan import Pix2PixGAN
from utils.visualization import plot_loss_curves, plot_sample_predictions


def compute_dice_on_batch(model, images, masks):
    """Compute average Dice score on a batch of data.

    Parameters
    ----------
    model : tf.keras.Model
        The generator model.
    images : np.ndarray
        Input images, shape (N, H, W, 1).
    masks : np.ndarray
        Ground truth masks, shape (N, H, W, 1).

    Returns
    -------
    float
        Average Dice coefficient across all samples.
    """
    dice_scores = []
    for i in range(len(images)):
        pred = model(images[i:i+1], training=False).numpy()
        pred_binary = (pred[0, :, :, 0] > 0.5).astype(np.float32)
        gt = masks[i, :, :, 0]

        intersection = np.sum(pred_binary * gt)
        total = np.sum(pred_binary) + np.sum(gt)

        if total == 0:
            dice = 1.0  # Both empty = perfect match
        else:
            dice = (2.0 * intersection) / total

        dice_scores.append(dice)

    return np.mean(dice_scores)


def train(view="2CH", structure="LVendo", epochs=None, resume_from=None):
    """Train the Pix2Pix GAN model with all improvements.

    Automatic saturation detection stops training when validation Dice
    plateaus, preventing wasted compute time.

    Parameters
    ----------
    view : str
        Camera view: '2CH' or '4CH'.
    structure : str
        Cardiac structure: 'LVendo', 'LVmyo', or 'LA'.
    epochs : int
        Maximum training epochs (default from config). Training may stop
        earlier via early stopping if Dice score saturates.
    resume_from : str, optional
        Path to generator weights to resume training from.
    """
    if epochs is None:
        epochs = config.EPOCHS

    print("=" * 60)
    print(f"  Pix2Pix GAN Training (IMPROVED v2)")
    print(f"  View: {view} | Structure: {structure}")
    print(f"  Max Epochs: {epochs} (auto-stops at saturation)")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  LR (G): {config.LEARNING_RATE_G} | LR (D): {config.LEARNING_RATE_D}")
    print(f"  Lambda pixel: {config.LAMBDA_PIXEL} | Lambda Dice: {config.LAMBDA_DICE}")
    print(f"  Label smoothing: {config.LABEL_SMOOTHING}")
    print(f"  D throttle threshold: {config.D_THROTTLE_THRESHOLD}")
    print(f"  Augmentation: {config.AUGMENT_TRAIN}")
    print(f"  Early stop patience: {config.EARLY_STOP_PATIENCE} epochs")
    print(f"  LR reduce patience: {config.LR_REDUCE_PATIENCE} epochs")
    print("=" * 60)

    # =========================================================================
    # 1. Load and preprocess data
    # =========================================================================
    print("\n[1/4] Loading dataset...")
    train_images, train_masks, test_images, test_masks = load_train_test(
        view=view, structure=structure
    )

    print("\n[2/4] Preprocessing...")
    X_train, Y_train = preprocess_dataset(train_images, train_masks)
    X_test, Y_test = preprocess_dataset(test_images, test_masks)

    print(f"  Training set:  {X_train.shape}")
    print(f"  Test set:      {X_test.shape}")

    n_samples = X_train.shape[0]
    n_batches = n_samples // config.BATCH_SIZE

    if n_batches == 0:
        print("ERROR: Not enough samples for even 1 batch. Check your dataset.")
        return

    # =========================================================================
    # 2. Build model
    # =========================================================================
    print("\n[3/4] Building Pix2Pix GAN...")
    gan = Pix2PixGAN()

    if resume_from:
        gan.generator.load_weights(resume_from)
        print(f"  Resumed from: {resume_from}")

    print(f"  Generator parameters:     {gan.generator.count_params():,}")
    print(f"  Discriminator parameters: {gan.discriminator.count_params():,}")
    print(f"  Total parameters:         "
          f"{gan.generator.count_params() + gan.discriminator.count_params():,}")

    # =========================================================================
    # 3. Training loop
    # =========================================================================
    print("\n[4/4] Training...")

    run_name = f"{view}_{structure}"
    run_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, run_name)
    run_plot_dir = os.path.join(config.PLOT_DIR, run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    os.makedirs(run_plot_dir, exist_ok=True)

    # Loss & Dice history
    history = {
        "gen_loss": [],
        "gen_adv_loss": [],
        "gen_pixel_loss": [],
        "gen_dice_loss": [],
        "disc_loss": [],
        "disc_real_loss": [],
        "disc_fake_loss": [],
        "val_dice": [],
    }

    best_dice = 0.0
    best_gen_loss = float("inf")
    d_skipped_total = 0
    start_time = time.time()

    # --- Early stopping & LR scheduling state ---
    # These are INDEPENDENT counters (fixing the critical bug from v1)
    epochs_since_improvement = 0    # For early stopping (never reset except on improvement)
    lr_reductions_done = 0          # Track how many times we've reduced LR
    current_lr_g = config.LEARNING_RATE_G
    current_lr_d = config.LEARNING_RATE_D

    # Running D_loss for per-batch throttling (exponential moving average)
    running_d_loss = 1.0  # Start high = don't throttle initially

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Apply augmentation per epoch (fresh random transforms)
        if config.AUGMENT_TRAIN:
            X_epoch, Y_epoch = augment_dataset(X_train, Y_train)
        else:
            X_epoch, Y_epoch = X_train, Y_train

        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_epoch[indices]
        Y_shuffled = Y_epoch[indices]

        epoch_losses = {k: [] for k in history.keys() if k != "val_dice"}
        d_skipped_epoch = 0

        # Mini-batch learning loop
        for batch_idx in range(n_batches):
            start = batch_idx * config.BATCH_SIZE
            end = start + config.BATCH_SIZE

            batch_X = tf.constant(X_shuffled[start:end], dtype=tf.float32)
            batch_Y = tf.constant(Y_shuffled[start:end], dtype=tf.float32)

            # Per-batch D throttling using running average
            update_d = running_d_loss >= config.D_THROTTLE_THRESHOLD

            # Dispatch to correct train step
            losses = gan.train_step(batch_X, batch_Y,
                                    update_discriminator=update_d)

            if not update_d:
                d_skipped_epoch += 1

            # Update running D_loss (exponential moving average, α=0.1)
            batch_d_loss = float(losses["disc_loss"])
            running_d_loss = 0.9 * running_d_loss + 0.1 * batch_d_loss

            for k in epoch_losses:
                epoch_losses[k].append(float(losses[k]))

        d_skipped_total += d_skipped_epoch

        # Average losses for this epoch
        for k in epoch_losses:
            avg = np.mean(epoch_losses[k])
            history[k].append(avg)

        epoch_time = time.time() - epoch_start

        # Log progress
        if epoch % config.DISPLAY_INTERVAL == 0 or epoch == 1:
            d_skip_pct = (d_skipped_epoch / n_batches * 100) if n_batches > 0 else 0
            elapsed_min = (time.time() - start_time) / 60
            print(f"  Epoch {epoch:4d}/{epochs} [{epoch_time:.1f}s, {elapsed_min:.0f}m total] | "
                  f"G: {history['gen_loss'][-1]:.4f} "
                  f"(adv:{history['gen_adv_loss'][-1]:.3f} "
                  f"pix:{history['gen_pixel_loss'][-1]:.3f} "
                  f"dice:{history['gen_dice_loss'][-1]:.3f}) | "
                  f"D: {history['disc_loss'][-1]:.4f} "
                  f"(skip:{d_skip_pct:.0f}%) | "
                  f"run_D:{running_d_loss:.3f}")

        # =====================================================================
        # Validation Dice (the real metric — checked every N epochs)
        # =====================================================================
        if epoch % config.VALIDATE_INTERVAL == 0 or epoch == 1:
            val_dice = compute_dice_on_batch(gan.generator, X_test, Y_test)
            history["val_dice"].append(val_dice)
            print(f"  ► Val Dice: {val_dice:.4f} (best: {best_dice:.4f}) | "
                  f"no-improve: {epochs_since_improvement} epochs")

            if val_dice > best_dice:
                # === IMPROVEMENT FOUND ===
                best_dice = val_dice
                gan.save_models(run_checkpoint_dir, epoch="best")
                print(f"  ★ New best Dice: {best_dice:.4f}")
                epochs_since_improvement = 0
                lr_reductions_done = 0
            else:
                # === NO IMPROVEMENT ===
                epochs_since_improvement += config.VALIDATE_INTERVAL

            # --- LR reduction on plateau (can trigger multiple times) ---
            # Trigger at: LR_REDUCE_PATIENCE, 2*LR_REDUCE_PATIENCE, etc.
            lr_reduce_threshold = config.LR_REDUCE_PATIENCE * (lr_reductions_done + 1)
            if epochs_since_improvement >= lr_reduce_threshold:
                if current_lr_g > config.MIN_LR:
                    lr_reductions_done += 1
                    current_lr_g = max(
                        current_lr_g * config.LR_REDUCE_FACTOR, config.MIN_LR)
                    current_lr_d = max(
                        current_lr_d * config.LR_REDUCE_FACTOR, config.MIN_LR)
                    gan.gen_optimizer.learning_rate.assign(current_lr_g)
                    gan.disc_optimizer.learning_rate.assign(current_lr_d)
                    print(f"  ↓ LR reduced (#{lr_reductions_done}) → "
                          f"G:{current_lr_g:.2e} D:{current_lr_d:.2e}")

            # --- Early stopping (INDEPENDENT of LR reduction) ---
            if epochs_since_improvement >= config.EARLY_STOP_PATIENCE:
                print(f"\n  {'='*50}")
                print(f"  ⊘ EARLY STOPPING at epoch {epoch}")
                print(f"    Dice has not improved for "
                      f"{epochs_since_improvement} epochs")
                print(f"    Best Dice: {best_dice:.4f}")
                print(f"    LR reductions done: {lr_reductions_done}")
                print(f"  {'='*50}")
                break

        # Also track best G_loss (secondary metric)
        current_gen_loss = history["gen_loss"][-1]
        if current_gen_loss < best_gen_loss:
            best_gen_loss = current_gen_loss

        # Periodic checkpoint
        if epoch % config.SAVE_INTERVAL == 0:
            gan.save_models(run_checkpoint_dir, epoch=epoch)

    # =========================================================================
    # 4. Save final results
    # =========================================================================
    total_time = time.time() - start_time
    gan.save_models(run_checkpoint_dir, epoch="final")

    # Save loss history
    history_path = os.path.join(run_checkpoint_dir, "loss_history.npy")
    np.save(history_path, history)
    print(f"\nLoss history saved to: {history_path}")

    # Plot loss curves
    try:
        plot_loss_curves(history,
                         save_path=os.path.join(run_plot_dir, "loss_curves.png"))
        print(f"Loss curves saved to: {run_plot_dir}/loss_curves.png")
    except Exception as e:
        print(f"Warning: Could not plot loss curves: {e}")

    # Plot sample predictions using BEST model
    try:
        best_gen_path = os.path.join(run_checkpoint_dir, "generator_epoch_best.h5")
        if os.path.exists(best_gen_path):
            gan.generator.load_weights(best_gen_path)

        n_samples_to_show = min(5, len(X_test))
        predictions = gan.generator(X_test[:n_samples_to_show], training=False)
        plot_sample_predictions(
            X_test[:n_samples_to_show],
            Y_test[:n_samples_to_show],
            predictions.numpy(),
            save_path=os.path.join(run_plot_dir, "sample_predictions.png")
        )
        print(f"Sample predictions saved to: {run_plot_dir}/sample_predictions.png")
    except Exception as e:
        print(f"Warning: Could not plot sample predictions: {e}")

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Total time: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
    print(f"  Final epoch: {epoch}")
    print(f"  Best Dice score: {best_dice:.4f}")
    print(f"  Best generator loss: {best_gen_loss:.4f}")
    print(f"  LR reductions: {lr_reductions_done}")
    print(f"  D updates skipped: {d_skipped_total} total")
    print(f"  Checkpoints: {run_checkpoint_dir}")
    print(f"  Plots: {run_plot_dir}")
    print("=" * 60)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Pix2Pix GAN for echocardiography segmentation"
    )
    parser.add_argument(
        "--view", type=str, default="2CH",
        choices=["2CH", "4CH"],
        help="Camera view: 2CH or 4CH (default: 2CH)"
    )
    parser.add_argument(
        "--structure", type=str, default="LVendo",
        choices=["LVendo", "LVmyo", "LA"],
        help="Cardiac structure to segment (default: LVendo)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Max training epochs (default: {config.EPOCHS}). "
             "Training auto-stops at saturation."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to generator weights to resume from"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        view=args.view,
        structure=args.structure,
        epochs=args.epochs,
        resume_from=args.resume,
    )

```

### `evaluation/metrics.py`
```python
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def dice_coefficient(prediction, target, smooth=1e-6):
    pred_flat = prediction.flatten()
    target_flat = target.flatten()
    intersection = np.sum(pred_flat * target_flat)
    return (2.0 * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)

def mean_absolute_difference(prediction, target):
    return np.mean(np.abs(target.flatten() - prediction.flatten()))

def hausdorff_distance(prediction, target):
    pred_points = np.argwhere(prediction > 0.5)
    target_points = np.argwhere(target > 0.5)
    if len(pred_points) == 0 or len(target_points) == 0: return float('inf')
    forward_hd = directed_hausdorff(pred_points, target_points)[0]
    backward_hd = directed_hausdorff(target_points, pred_points)[0]
    return max(forward_hd, backward_hd)

def calculate_volume_simpson(mask_2d, pixel_spacing=1.0):
    height = mask_2d.shape[0]
    disc_thickness = pixel_spacing
    volume = 0.0
    for row in range(height):
        row_pixels = np.sum(mask_2d[row] > 0.5)
        if row_pixels > 0:
            diameter = row_pixels * pixel_spacing
            area = np.pi * (diameter / 2.0) ** 2
            volume += area * disc_thickness
    return volume / 1000.0

def ejection_fraction(ed_volume, es_volume):
    if ed_volume == 0: return 0.0
    return (ed_volume - es_volume) / ed_volume

def ef_correlation(ef_predicted, ef_ground_truth):
    if len(ef_predicted) < 2: return 0.0
    return np.corrcoef(ef_ground_truth, ef_predicted)[0, 1]

def ef_mae(ef_predicted, ef_ground_truth):
    return np.mean(np.abs(np.array(ef_ground_truth) - np.array(ef_predicted)))
```

### `evaluation/evaluate.py`
```python
"""
Evaluation / inference script for the trained Pix2Pix GAN.

Loads a trained model, runs inference on the test set, computes
all geometric and clinical metrics, and saves results.

Usage:
    python evaluation/evaluate.py --view 2CH --structure LVendo
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from data.dataset import load_dataset
from data.preprocessing import preprocess_dataset
from models.pix2pix_gan import Pix2PixGAN
from evaluation.metrics import (
    evaluate_segmentation,
    calculate_volume_simpson,
    ejection_fraction,
    ef_correlation,
    ef_mae,
)
from utils.visualization import plot_sample_predictions, plot_ef_correlation


def evaluate(view="2CH", structure="LVendo", weights_path=None):
    """Run evaluation on the test set.

    Parameters
    ----------
    view : str
        Camera view: '2CH' or '4CH'.
    structure : str
        Cardiac structure: 'LVendo', 'LVmyo', or 'LA'.
    weights_path : str, optional
        Path to generator weights. If None, uses best checkpoint.
    """
    print("=" * 60)
    print(f"  Evaluation: {view} | {structure}")
    print("=" * 60)

    # =========================================================================
    # 1. Determine weights path
    # =========================================================================
    run_name = f"{view}_{structure}"
    run_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, run_name)
    run_results_dir = os.path.join(config.RESULTS_DIR, run_name)
    run_plot_dir = os.path.join(config.PLOT_DIR, run_name)
    os.makedirs(run_results_dir, exist_ok=True)
    os.makedirs(run_plot_dir, exist_ok=True)

    if weights_path is None:
        weights_path = os.path.join(run_checkpoint_dir, "generator_epoch_best.h5")

    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at: {weights_path}")
        print("Please train the model first using: python training/train.py")
        return

    # =========================================================================
    # 2. Load model
    # =========================================================================
    print("\nLoading model...")
    gan = Pix2PixGAN()
    gan.generator.load_weights(weights_path)
    print(f"  Loaded weights from: {weights_path}")

    # =========================================================================
    # 3. Load and preprocess test data for each phase
    # =========================================================================
    all_results = {}

    for phase in config.PHASES:
        print(f"\n{'─' * 40}")
        print(f"  Phase: {phase}")
        print(f"{'─' * 40}")

        # Load test data
        test_images, test_masks = load_dataset(
            view=view, phase=phase, structure=structure, split="test"
        )

        if len(test_images) == 0:
            print(f"  No test data found for {view}/{phase}/{structure}")
            continue

        X_test, Y_test = preprocess_dataset(test_images, test_masks)
        print(f"  Test samples: {X_test.shape[0]}")

        # Run inference
        print("  Running inference...")
        predictions = gan.generator(X_test, training=False).numpy()

        # Evaluate geometric metrics
        print(f"\n  Geometric metrics for {phase}:")
        results = evaluate_segmentation(predictions, Y_test, verbose=True)
        all_results[phase] = results

        # Plot sample predictions
        n_show = min(5, len(X_test))
        try:
            plot_sample_predictions(
                X_test[:n_show], Y_test[:n_show], predictions[:n_show],
                save_path=os.path.join(run_plot_dir, f"predictions_{phase}.png"),
                title=f"{view} - {structure} - {phase}"
            )
        except Exception as e:
            print(f"  Warning: Could not plot: {e}")

    # =========================================================================
    # 4. Clinical parameters (EF analysis for LVendo)
    # =========================================================================
    if structure == "LVendo" and "ED" in all_results and "ES" in all_results:
        print(f"\n{'─' * 40}")
        print("  Clinical Parameters (EF Analysis)")
        print(f"{'─' * 40}")

        # Load ED and ES test data
        ed_images, ed_masks = load_dataset(view, "ED", structure, "test")
        es_images, es_masks = load_dataset(view, "ES", structure, "test")
        X_ed, Y_ed = preprocess_dataset(ed_images, ed_masks)
        X_es, Y_es = preprocess_dataset(es_images, es_masks)

        pred_ed = (gan.generator(X_ed, training=False).numpy().squeeze(-1) > 0.5).astype(float)
        pred_es = (gan.generator(X_es, training=False).numpy().squeeze(-1) > 0.5).astype(float)
        gt_ed = Y_ed.squeeze(-1)
        gt_es = Y_es.squeeze(-1)

        n_patients = min(len(pred_ed), len(pred_es))

        ef_pred_list = []
        ef_gt_list = []

        for i in range(n_patients):
            # Predicted volumes and EF
            vol_ed_pred = calculate_volume_simpson(pred_ed[i])
            vol_es_pred = calculate_volume_simpson(pred_es[i])
            ef_pred = ejection_fraction(vol_ed_pred, vol_es_pred)

            # Ground truth volumes and EF
            vol_ed_gt = calculate_volume_simpson(gt_ed[i])
            vol_es_gt = calculate_volume_simpson(gt_es[i])
            ef_gt = ejection_fraction(vol_ed_gt, vol_es_gt)

            ef_pred_list.append(ef_pred)
            ef_gt_list.append(ef_gt)

        ef_pred_arr = np.array(ef_pred_list)
        ef_gt_arr = np.array(ef_gt_list)

        corr = ef_correlation(ef_pred_arr, ef_gt_arr)
        mae = ef_mae(ef_pred_arr, ef_gt_arr)

        print(f"\n  EF Correlation: {corr:.4f}")
        print(f"  EF MAE:         {mae:.4f}")

        # Plot EF correlation
        try:
            plot_ef_correlation(
                ef_gt_arr, ef_pred_arr,
                save_path=os.path.join(run_plot_dir, "ef_correlation.png"),
                title=f"{view} - EF Correlation"
            )
        except Exception as e:
            print(f"  Warning: Could not plot EF correlation: {e}")

    # =========================================================================
    # 5. Save summary to CSV
    # =========================================================================
    summary_rows = []
    for phase, results in all_results.items():
        summary_rows.append({
            "View": view,
            "Structure": structure,
            "Phase": phase,
            "Dice_Mean": results["dice_mean"],
            "Dice_Std": results["dice_std"],
            "MAD_Mean": results["mad_mean"],
            "MAD_Std": results["mad_std"],
            "HD_Mean": results["hd_mean"],
            "HD_Std": results["hd_std"],
        })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(run_results_dir, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        print("\n" + df.to_string(index=False))

    print("\n" + "=" * 60)
    print("  Evaluation complete!")
    print("=" * 60)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Pix2Pix GAN for echocardiography segmentation"
    )
    parser.add_argument(
        "--view", type=str, default="2CH",
        choices=["2CH", "4CH"],
        help="Camera view (default: 2CH)"
    )
    parser.add_argument(
        "--structure", type=str, default="LVendo",
        choices=["LVendo", "LVmyo", "LA"],
        help="Cardiac structure (default: LVendo)"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to generator weights (default: best checkpoint)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        view=args.view,
        structure=args.structure,
        weights_path=args.weights,
    )

```

### `predict.py`
```python
"""
Custom inference script — run the trained model on your own echocardiography images.

Supports:
    - Single image files (PNG, JPG, BMP, TIFF, DICOM)
    - Folders of images
    - Video files (AVI, MP4) — processes each frame

Usage:
    python predict.py --input path/to/image.png --weights outputs/checkpoints/2CH_LVendo/generator_epoch_best.h5
    python predict.py --input path/to/folder/ --weights outputs/checkpoints/2CH_LVendo/generator_epoch_best.h5
    python predict.py --input path/to/video.avi --weights outputs/checkpoints/2CH_LVendo/generator_epoch_best.h5
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import config
from models.pix2pix_gan import Pix2PixGAN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================================
# Image loading utilities
# =========================================================================

def load_image_file(filepath):
    """Load a single image file (PNG, JPG, BMP, TIFF, or DICOM).

    Parameters
    ----------
    filepath : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Grayscale image as 2-D numpy array.
    """
    ext = os.path.splitext(filepath)[1].lower()

    # DICOM files
    if ext in ['.dcm', '.dicom']:
        try:
            import pydicom
            ds = pydicom.dcmread(filepath)
            img = ds.pixel_array.astype(np.float32)
        except ImportError:
            raise ImportError("Install pydicom for DICOM support: pip install pydicom")
    # MHD files (CAMUS format)
    elif ext in ['.mhd']:
        try:
            import SimpleITK as sitk
            image = sitk.ReadImage(filepath)
            img = sitk.GetArrayFromImage(image).astype(np.float32)
            if img.ndim == 3:
                img = img[0]
        except ImportError:
            raise ImportError("Install SimpleITK for .mhd support: pip install SimpleITK")
    # Standard image files
    else:
        from skimage.io import imread
        img = imread(filepath).astype(np.float32)

    # Convert RGB/RGBA to grayscale if needed
    if img.ndim == 3:
        img = np.mean(img[:, :, :3], axis=2)

    return img


def load_video_frames(video_path, max_frames=None):
    """Load frames from a video file.

    Parameters
    ----------
    video_path : str
        Path to video file (AVI, MP4, etc.).
    max_frames : int, optional
        Maximum number of frames to extract.

    Returns
    -------
    list of np.ndarray
        List of grayscale frames.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("Install OpenCV for video support: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(np.float32))
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    print(f"Loaded {len(frames)} frames from video")
    return frames


# =========================================================================
# Preprocessing (same as training pipeline)
# =========================================================================

def preprocess_single_image(image):
    """Preprocess a single image for inference.

    Parameters
    ----------
    image : np.ndarray
        Raw grayscale image of any size.

    Returns
    -------
    np.ndarray
        Preprocessed image of shape (1, 256, 256, 1).
    """
    from skimage.transform import resize

    # Resize to 256x256
    resized = resize(image, (config.IMG_HEIGHT, config.IMG_WIDTH),
                     mode='reflect', anti_aliasing=True,
                     preserve_range=True).astype(np.float32)

    # Normalize to [0, 1]
    img_min = resized.min()
    img_max = resized.max()
    if img_max - img_min > 0:
        resized = (resized - img_min) / (img_max - img_min)
    else:
        resized = np.zeros_like(resized)

    # Add batch and channel dims: (H, W) -> (1, H, W, 1)
    return resized[np.newaxis, :, :, np.newaxis]


# =========================================================================
# Prediction and visualization
# =========================================================================

def predict_and_save(image, model, save_path, original_path=""):
    """Run prediction on a single image and save the result.

    Parameters
    ----------
    image : np.ndarray
        Raw grayscale image.
    model : tf.keras.Model
        Trained generator model.
    save_path : str
        Path to save the output image.
    original_path : str
        Name of the original file (for display).
    """
    # Preprocess
    input_tensor = preprocess_single_image(image)

    # Predict
    prediction = model(input_tensor, training=False).numpy()
    pred_mask = (prediction[0, :, :, 0] > 0.5).astype(np.float32)

    # Resize input for display
    from skimage.transform import resize
    display_img = resize(image, (config.IMG_HEIGHT, config.IMG_WIDTH),
                         mode='reflect', anti_aliasing=True,
                         preserve_range=True)
    # Normalize for display
    dmin, dmax = display_img.min(), display_img.max()
    if dmax - dmin > 0:
        display_img = (display_img - dmin) / (dmax - dmin)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(display_img, cmap='gray')
    axes[0].set_title('Input Image', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # Predicted mask
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Predicted Segmentation', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(display_img, cmap='gray')
    axes[2].contour(pred_mask, levels=[0.5], colors='lime', linewidths=2)
    axes[2].set_title('Overlay', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    basename = os.path.basename(original_path) if original_path else "Custom Image"
    plt.suptitle(f'Segmentation Result: {basename}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def run_prediction(input_path, weights_path, output_dir=None, max_video_frames=50):
    """Run prediction on images, folders, or videos.

    Parameters
    ----------
    input_path : str
        Path to image file, folder of images, or video file.
    weights_path : str
        Path to trained generator weights (.h5 file).
    output_dir : str, optional
        Directory to save results (default: outputs/predictions/).
    max_video_frames : int
        Max frames to process from video.
    """
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    gan = Pix2PixGAN()
    gan.generator.load_weights(weights_path)
    model = gan.generator
    print(f"  Loaded weights: {weights_path}")

    # Determine input type
    if os.path.isdir(input_path):
        # Folder of images
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.mhd', '.dcm'}
        files = [f for f in os.listdir(input_path)
                 if os.path.splitext(f)[1].lower() in extensions]
        files.sort()
        print(f"\nProcessing {len(files)} images from folder...")

        for fname in files:
            fpath = os.path.join(input_path, fname)
            try:
                img = load_image_file(fpath)
                save_name = os.path.splitext(fname)[0] + "_segmented.png"
                save_path = os.path.join(output_dir, save_name)
                predict_and_save(img, model, save_path, fpath)
            except Exception as e:
                print(f"  ERROR on {fname}: {e}")

    elif input_path.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
        # Video file
        print(f"\nProcessing video: {input_path}")
        frames = load_video_frames(input_path, max_frames=max_video_frames)

        for i, frame in enumerate(frames):
            save_name = f"frame_{i:04d}_segmented.png"
            save_path = os.path.join(output_dir, save_name)
            predict_and_save(frame, model, save_path, f"frame_{i:04d}")

    else:
        # Single image file
        print(f"\nProcessing image: {input_path}")
        img = load_image_file(input_path)
        save_name = os.path.splitext(os.path.basename(input_path))[0] + "_segmented.png"
        save_path = os.path.join(output_dir, save_name)
        predict_and_save(img, model, save_path, input_path)

    print(f"\nAll results saved to: {output_dir}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run segmentation on custom echocardiography images"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to image file, folder of images, or video file"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained generator weights (.h5 file)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: outputs/predictions/)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=50,
        help="Max video frames to process (default: 50)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(
        input_path=args.input,
        weights_path=args.weights,
        output_dir=args.output,
        max_video_frames=args.max_frames,
    )

```

### `export_model.py`
```python
"""
Export trained model for use on another system.

Saves the generator as:
    1. Full Keras model (.keras or SavedModel) — easy to load anywhere
    2. Weights only (.h5) — smaller file, needs code to rebuild architecture

Usage:
    python export_model.py --weights outputs/checkpoints/2CH_LVendo/generator_epoch_best.h5 --output exported_model/
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import config
from models.pix2pix_gan import Pix2PixGAN


def export_model(weights_path, output_dir="exported_model"):
    """Export the trained generator model for deployment on another system.

    Creates three export formats:
        1. SavedModel (TensorFlow standard format — most portable)
        2. .h5 weights file (lightweight, needs architecture code)
        3. Standalone predict script that works without the full project

    Parameters
    ----------
    weights_path : str
        Path to the trained generator weights.
    output_dir : str
        Directory to save exported files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build and load the trained generator
    print("Building generator and loading weights...")
    gan = Pix2PixGAN()
    gan.generator.load_weights(weights_path)
    generator = gan.generator

    print(f"  Parameters: {generator.count_params():,}")
    print(f"  Input shape:  {generator.input_shape}")
    print(f"  Output shape: {generator.output_shape}")

    # =========================================================================
    # Export 1: SavedModel format (most portable)
    # =========================================================================
    saved_model_path = os.path.join(output_dir, "saved_model")
    generator.save(saved_model_path)
    print(f"\n[1] SavedModel exported to: {saved_model_path}")
    print(f"    Load with: tf.keras.models.load_model('{saved_model_path}')")

    # =========================================================================
    # Export 2: .h5 weights only (lightweight)
    # =========================================================================
    h5_path = os.path.join(output_dir, "generator_weights.h5")
    generator.save_weights(h5_path)
    print(f"\n[2] Weights exported to: {h5_path}")

    # =========================================================================
    # Export 3: Standalone prediction script
    # =========================================================================
    standalone_script = '''"""
Standalone prediction script — works without the full project.
Just needs: tensorflow, numpy, scikit-image, matplotlib

Usage:
    python standalone_predict.py --input your_image.png --model saved_model/
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG_SIZE = 256

def load_and_preprocess(image_path):
    """Load and preprocess an echocardiography image."""
    img = imread(image_path).astype(np.float32)
    if img.ndim == 3:
        img = np.mean(img[:, :, :3], axis=2)
    resized = resize(img, (IMG_SIZE, IMG_SIZE), mode='reflect',
                     anti_aliasing=True, preserve_range=True).astype(np.float32)
    img_min, img_max = resized.min(), resized.max()
    if img_max - img_min > 0:
        resized = (resized - img_min) / (img_max - img_min)
    return resized[np.newaxis, :, :, np.newaxis], img

def predict_and_save(image_path, model, output_path):
    """Run segmentation and save result."""
    input_tensor, original = load_and_preprocess(image_path)
    prediction = model(input_tensor, training=False).numpy()
    mask = (prediction[0, :, :, 0] > 0.5).astype(float)
    display = resize(original, (IMG_SIZE, IMG_SIZE), mode='reflect',
                     anti_aliasing=True, preserve_range=True)
    dmin, dmax = display.min(), display.max()
    if dmax - dmin > 0:
        display = (display - dmin) / (dmax - dmin)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display, cmap='gray')
    axes[0].set_title('Input')
    axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    axes[2].imshow(display, cmap='gray')
    axes[2].contour(mask, levels=[0.5], colors='lime', linewidths=2)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--model", required=True, help="Path to saved_model/ directory")
    parser.add_argument("--output", default="result.png", help="Output image path")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    predict_and_save(args.input, model, args.output)
'''

    standalone_path = os.path.join(output_dir, "standalone_predict.py")
    with open(standalone_path, 'w') as f:
        f.write(standalone_script)
    print(f"\n[3] Standalone script exported to: {standalone_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("  Export complete!")
    print("=" * 60)
    print(f"\nTo use on another system:")
    print(f"  1. Copy the '{output_dir}/' folder to the new machine")
    print(f"  2. Install: pip install tensorflow numpy scikit-image matplotlib")
    print(f"  3. Run: python standalone_predict.py --input your_image.png --model saved_model/")
    print("=" * 60)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export trained model for use on another system"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained generator weights (.h5)"
    )
    parser.add_argument(
        "--output", type=str, default="exported_model",
        help="Output directory (default: exported_model/)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model(
        weights_path=args.weights,
        output_dir=args.output,
    )

```

### `utils/visualization.py`
```python
"""
Visualization utilities for the Pix2Pix GAN — IMPROVED VERSION.

Provides plotting functions for:
    - Loss curves with Dice overlay
    - Sample segmentation predictions vs ground truth
    - EF correlation scatter plots
    - Dice score boxplots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


def plot_loss_curves(history, save_path=None, title="Training Loss Curves"):
    """Plot training loss curves with optional Dice score overlay.

    Shows G/D losses on top subplot and validation Dice on bottom subplot.

    Parameters
    ----------
    history : dict
        Dictionary with keys: 'gen_loss', 'disc_real_loss', 'disc_fake_loss',
        and optionally 'val_dice', 'gen_dice_loss'.
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    """
    has_dice = 'val_dice' in history and len(history['val_dice']) > 0
    n_plots = 2 if has_dice else 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(history['gen_loss']) + 1)

    # --- Top plot: Losses ---
    ax = axes[0]

    # Discriminator losses
    ax.plot(epochs, history['disc_real_loss'], 'r-', alpha=0.6,
            label='D Real Loss', linewidth=1.5)
    ax.plot(epochs, history['disc_fake_loss'], 'b-', alpha=0.6,
            label='D Fake Loss', linewidth=1.5)
    ax.plot(epochs, history['disc_loss'], 'purple', alpha=0.5,
            label='D Total Loss', linewidth=1.0, linestyle='--')

    # Generator loss
    ax.plot(epochs, history['gen_loss'], color='orange', alpha=0.8,
            label='G Loss', linewidth=2)

    # Generator components
    if 'gen_dice_loss' in history and len(history['gen_dice_loss']) > 0:
        ax.plot(epochs, history['gen_dice_loss'], color='green', alpha=0.5,
                label='G Dice Loss', linewidth=1.0, linestyle=':')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # D throttle reference line
    ax.axhline(y=0.3, color='gray', linestyle=':', alpha=0.5,
               label='D throttle threshold')

    # --- Bottom plot: Validation Dice ---
    if has_dice:
        ax2 = axes[1]
        val_dice = history['val_dice']
        # Dice is sampled every VALIDATE_INTERVAL epochs
        n_dice = len(val_dice)
        n_epochs = len(history['gen_loss'])
        dice_epochs = np.linspace(1, n_epochs, n_dice, dtype=int)

        ax2.plot(dice_epochs, val_dice, 'g-o', linewidth=2,
                 markersize=4, label='Validation Dice')
        ax2.axhline(y=max(val_dice), color='green', linestyle='--',
                     alpha=0.5, label=f'Best: {max(val_dice):.4f}')

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Dice Coefficient', fontsize=12)
        ax2.set_title('Validation Dice Score', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_sample_predictions(images, ground_truths, predictions,
                            save_path=None, title="Segmentation Results",
                            n_samples=None):
    """Plot input images alongside ground truth and predicted segmentation.

    Parameters
    ----------
    images : np.ndarray
        Input images, shape (N, H, W, 1).
    ground_truths : np.ndarray
        Ground truth masks, shape (N, H, W, 1).
    predictions : np.ndarray
        Predicted masks, shape (N, H, W, 1).
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    n_samples : int, optional
        Number of samples to show (default: all).
    """
    if n_samples is None:
        n_samples = min(len(images), 5)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    if n_samples == 1:
        axes = axes[np.newaxis, :]

    column_titles = ['Input Image', 'Ground Truth', 'Prediction', 'Overlay']

    for col, col_title in enumerate(column_titles):
        axes[0, col].set_title(col_title, fontsize=13, fontweight='bold')

    for i in range(n_samples):
        img = images[i].squeeze()
        gt = ground_truths[i].squeeze()
        pred = (predictions[i].squeeze() > 0.5).astype(float)

        # Compute per-sample Dice for display
        intersection = np.sum(pred * gt)
        total = np.sum(pred) + np.sum(gt)
        dice = (2.0 * intersection / total) if total > 0 else 1.0

        # Input image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].axis('off')

        # Predicted mask
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].axis('off')

        # Overlay: input with GT contour (red) and Pred contour (green)
        axes[i, 3].imshow(img, cmap='gray')
        axes[i, 3].contour(gt, levels=[0.5], colors='red',
                           linewidths=1.5, linestyles='dashed')
        axes[i, 3].contour(pred, levels=[0.5], colors='lime',
                           linewidths=1.5)
        axes[i, 3].set_title(f'Dice: {dice:.3f}', fontsize=11, color='green')
        axes[i, 3].axis('off')

    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ef_correlation(ef_gt, ef_pred, save_path=None,
                        title="EF Correlation"):
    """Plot EF correlation scatter plot.

    Parameters
    ----------
    ef_gt : np.ndarray
        Ground truth EF values.
    ef_pred : np.ndarray
        Predicted EF values.
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.scatter(ef_gt * 100, ef_pred * 100, alpha=0.6, s=40,
               edgecolors='navy', facecolors='cornflowerblue')

    # Identity line
    lims = [0, 100]
    ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=1.5, label='Identity')

    # Correlation
    corr = np.corrcoef(ef_gt, ef_pred)[0, 1]
    mae = np.mean(np.abs(ef_gt - ef_pred)) * 100

    ax.set_xlabel('Ground Truth EF (%)', fontsize=12)
    ax.set_ylabel('Predicted EF (%)', fontsize=12)
    ax.set_title(f'{title}\nCorrelation: {corr:.3f} | MAE: {mae:.1f}%',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dice_boxplot(dice_scores_dict, save_path=None,
                      title="Dice Scores by Training Size"):
    """Plot boxplot of Dice scores across training sizes.

    Parameters
    ----------
    dice_scores_dict : dict
        Keys are training sizes (int), values are lists of Dice scores.
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    labels = sorted(dice_scores_dict.keys())
    data = [dice_scores_dict[k] for k in labels]

    bp = ax.boxplot(data, labels=[str(l) for l in labels],
                    patch_artist=True)

    # Style
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Training Size', fontsize=12)
    ax.set_ylabel('Dice Coefficient', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

```

---
**Summary for a new AI Assistant:**
The project uses the standard CAMUS split. To run the training, initialize from `training/train.py`, which delegates preprocessing to `data/`, model logic to `models/`, and metrics to `evaluation/`. The outputs dynamically save plots and metrics automatically based on the view/structure combinations defined in `config.py`.
