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
*(Main training script handling loop iterations, adaptive discriminator throttling based on D_loss, validation Dice evaluation, and early stopping / learning rate reduction based on plateaus. Kept out of standard dump due to length, but effectively organizes all methods described above into a `train()` method.)*

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
*(Runs metrics on test dataset, computes geometric metrics across phases and clinical EF correlations, exports to CSV).*

### `predict.py`
*(Custom script to run `predict_and_save` on arbitrary `.png`, `.jpg`, `.dicom` images, folders of images, or `.avi`/`.mp4` videos by loading frames, applying preprocessing, model inference, and outputting the mask/overlay side-by-side plots).*

### `export_model.py`
*(Automates saving the `.h5` model to a `.keras` or `SavedModel` folder format, and simultaneously creates a completely independent `standalone_predict.py` script containing all logic necessary to run predictions without the project directory, ensuring the resulting model is strictly portable).*

### `utils/visualization.py`
*(Utilizes `matplotlib` to render loss curve overlays, Dice progress tracking, EF scatterplots, boxplots, and side-by-side validation overlays of input vs. GT vs. Pred).*

---
**Summary for a new AI Assistant:**
The project uses the standard CAMUS split. To run the training, initialize from `training/train.py`, which delegates preprocessing to `data/`, model logic to `models/`, and metrics to `evaluation/`. The outputs dynamically save plots and metrics automatically based on the view/structure combinations defined in `config.py`.
