"""
train.py — CNN semantic segmentation training for litter detection.

This file IS modified by the agent. Everything is fair game:
  - Model architecture (encoder depth, decoder, attention, backbone, etc.)
  - Loss function (BCE, Dice, Focal, combo)
  - Optimizer and LR schedule
  - Data augmentation strategy
  - Batch size, image crop size
  - Any other technique the agent wants to try

Constraint: training stops after a fixed epoch budget so every experiment is
comparable. The primary metric logged to MLflow is val_iou (higher is better).

Usage:
    uv run python auto-research/train.py [--run-name NAME] [--epochs N] [--model NAME|all]
"""

import argparse
import hashlib
import json
import platform
import random
import sqlite3
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Hyperparameters (edit freely) ─────────────────────────────────────────────

DEFAULT_EPOCHS   = 50
DEFAULT_SEED     = 42
BATCH_SIZE       = 8
CROP_SIZE        = 256        # random-crop spatial resolution during training
LR               = 8e-4
WEIGHT_DECAY     = 1e-3
ENCODER_CHANNELS = [64, 128, 256, 512]   # U-Net encoder stage widths
DECODER_CHANNELS = [256, 128, 64, 32]    # U-Net decoder stage widths
DROPOUT          = 0.25
POS_WEIGHT       = 5.0        # BCEWithLogitsLoss pos_weight (handles class imbalance)
USE_GROUND_ROI   = False
GROUND_ROI_TOP   = 0.4
DEFAULT_THRESHOLD = 0.7
THRESHOLD_CANDIDATES = [0.5, 0.6, 0.7, 0.8, 0.85]

USE_POSTPROCESSING = True
MIN_COMPONENT_SIZE = 200
USE_ERROR_ANALYSIS = True
ERROR_ANALYSIS_DIR = "error_analysis"
MAX_ERROR_SAMPLES_PER_EPOCH = 5
FALSE_POSITIVE_THRESHOLD = 0.01
                               # override with value from data/meta.json

# ── Data ──────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR   = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR  = DATA_DIR / "masks"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "mlflow"
MODELS_DIR = REPO_ROOT / "models" / "checkpoints"
MLFLOW_DB = ARTIFACTS_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = ARTIFACTS_DIR / "mlruns"
MLFLOW_EXPERIMENT = "litter-segmentation"


def load_meta() -> dict:
    p = DATA_DIR / "meta.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def dataset_fingerprint() -> str:
    hasher = hashlib.sha256()
    for path in (DATA_DIR / "train.txt", DATA_DIR / "val.txt", DATA_DIR / "meta.json"):
        if path.exists():
            hasher.update(path.name.encode("utf-8"))
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def crop_ground_roi(image: np.ndarray, mask: np.ndarray, roi_top: float):
    h = image.shape[0]
    y0 = int(h * roi_top)
    cropped_image = image[y0:, :]
    cropped_mask = mask[y0:, :]
    if cropped_mask.sum() == 0:
        return image, mask
    return cropped_image, cropped_mask


class LitterDataset(Dataset):
    def __init__(self, split: str, crop_size: int = CROP_SIZE, augment: bool = True):
        stems_file = DATA_DIR / f"{split}.txt"
        self.stems = [s.strip() for s in stems_file.read_text().splitlines() if s.strip()]
        self.augment = augment

        if augment:
            self.spatial_transforms = A.Compose([
                A.SmallestMaxSize(max_size=crop_size),
                A.PadIfNeeded(min_height=crop_size, min_width=crop_size,
                              border_mode=0, value=0, mask_value=0),
                A.RandomCrop(height=crop_size, width=crop_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
            ], additional_targets={"mask": "mask"})
            self.image_transforms = A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.05, p=0.5),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.spatial_transforms = A.Compose([
                A.SmallestMaxSize(max_size=crop_size),
                A.PadIfNeeded(min_height=crop_size, min_width=crop_size,
                              border_mode=0, value=0, mask_value=0),
                A.CenterCrop(height=crop_size, width=crop_size),
            ], additional_targets={"mask": "mask"})
            self.image_transforms = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        image = np.array(Image.open(IMAGES_DIR / f"{stem}.jpg").convert("RGB"))
        mask  = (np.array(Image.open(MASKS_DIR / f"{stem}.png")) > 127).astype(np.float32)

        if USE_GROUND_ROI:
            image, mask = crop_ground_roi(image, mask, GROUND_ROI_TOP)

        out = self.spatial_transforms(image=image, mask=mask)
        image = out["image"]
        mask = out["mask"]

        out = self.image_transforms(image=image)
        return out["image"], torch.from_numpy(mask).unsqueeze(0)


# ── Model ─────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Double conv + BN + ReLU block."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for multi-scale context.
    Applies dilated convolutions at multiple rates and fuses outputs.
    """
    def __init__(self, in_ch: int, out_ch: int, rates=(6, 12, 18)):
        super().__init__()
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Dilated 3x3 convolutions
        self.dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ) for r in rates
        ])
        # Global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Fusion projection: (1 + len(rates) + 1) * out_ch → out_ch
        n_branches = 1 + len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(n_branches * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        branches = [self.conv1(x)]
        for dil in self.dilated:
            branches.append(dil(x))
        gap_out = self.gap(x)
        gap_out = F.interpolate(gap_out, size=(h, w), mode='bilinear', align_corners=False)
        branches.append(gap_out)
        return self.project(torch.cat(branches, dim=1))


class UNet(nn.Module):
    """
    Vanilla U-Net for binary segmentation.
    Encoder depth and channel widths are controlled by ENCODER_CHANNELS /
    DECODER_CHANNELS — the agent is free to change these.
    """
    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: list[int] = ENCODER_CHANNELS,
        decoder_channels: list[int] = DECODER_CHANNELS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        assert len(encoder_channels) == len(decoder_channels)

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for out_ch in encoder_channels:
            self.encoders.append(ConvBlock(ch, out_ch, dropout))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(ch, ch * 2, dropout)
        ch = ch * 2

        # ── Decoder ───────────────────────────────────────────────────────
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for enc_ch, dec_ch in zip(reversed(encoder_channels), decoder_channels):
            self.upconvs.append(nn.ConvTranspose2d(ch, enc_ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(enc_ch * 2, dec_ch, dropout))
            ch = dec_ch

        # ── Head ──────────────────────────────────────────────────────────
        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # handle odd spatial sizes
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)


class ResNet34UNet(nn.Module):
    """
    U-Net with a pretrained ResNet34 encoder.

    Skip connections come from ResNet34 feature stages:
      stem  (64 ch,  H/2)   — after maxpool (stride-2 conv + BN + ReLU)
      layer1 (64 ch,  H/4)
      layer2 (128 ch, H/8)
      layer3 (256 ch, H/16)
      layer4 (512 ch, H/32)  — used as bottleneck

    The decoder mirrors a 4-stage U-Net decoder.
    BN layers in the backbone are frozen to preserve ImageNet statistics.
    """
    # Skip channel sizes from stem through layer3
    ENC_CHANNELS = [64, 64, 128, 256]   # stem, layer1, layer2, layer3
    BOTTLENECK_CH = 512                  # layer4

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()

        # ── Pretrained ResNet34 backbone ──────────────────────────────────
        backbone = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)

        # Stem: conv1 + bn1 + relu (output: 64 ch, stride 2)
        self.stem_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.stem_pool = backbone.maxpool   # stride 2 → H/4 total after stem+pool
        self.layer1 = backbone.layer1       # 64 ch,  H/4  (maxpool already applied)
        self.layer2 = backbone.layer2       # 128 ch, H/8
        self.layer3 = backbone.layer3       # 256 ch, H/16
        self.layer4 = backbone.layer4       # 512 ch, H/32  (bottleneck)

        # Freeze BN parameters in the backbone to preserve ImageNet stats
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

        # ── Decoder (4 stages) ────────────────────────────────────────────
        # Stage 1: upsample from 512 → 256, concat with layer3 skip (256) → 256
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256 + 256, 256, dropout)

        # Stage 2: upsample from 256 → 128, concat with layer2 skip (128) → 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 128, 128, dropout)

        # Stage 3: upsample from 128 → 64, concat with layer1 skip (64) → 64
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + 64, 64, dropout)

        # Stage 4: upsample from 64 → 32, concat with stem skip (64) → 32
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(32 + 64, 32, dropout)

        # Final upsample ×2 to recover full input resolution
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = ConvBlock(16, 16, dropout)

        # ── Head ──────────────────────────────────────────────────────────
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def _align(self, x, ref):
        """Bilinear resize x to match ref spatial dimensions if needed."""
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear",
                               align_corners=False)
        return x

    def forward(self, x):
        # Encoder
        s0 = self.stem_conv(x)         # 64 ch, H/2
        s1 = self.layer1(self.stem_pool(s0))  # 64 ch, H/4
        s2 = self.layer2(s1)           # 128 ch, H/8
        s3 = self.layer3(s2)           # 256 ch, H/16
        s4 = self.layer4(s3)           # 512 ch, H/32  (bottleneck)

        # Decoder
        d = self.up1(s4)
        d = self._align(d, s3)
        d = self.dec1(torch.cat([d, s3], dim=1))  # 256 ch, H/16

        d = self.up2(d)
        d = self._align(d, s2)
        d = self.dec2(torch.cat([d, s2], dim=1))  # 128 ch, H/8

        d = self.up3(d)
        d = self._align(d, s1)
        d = self.dec3(torch.cat([d, s1], dim=1))  # 64 ch, H/4

        d = self.up4(d)
        d = self._align(d, s0)
        d = self.dec4(torch.cat([d, s0], dim=1))  # 32 ch, H/2

        d = self.final_up(d)           # 16 ch, H/1
        d = self.final_conv(d)

        return self.head(d)            # 1 ch, H/1


class ResNet50UNet(nn.Module):
    """
    U-Net with a pretrained ResNet50 encoder.

    ResNet50 uses bottleneck blocks so channel counts differ from ResNet34:
      stem   (64 ch,  H/2)
      layer1 (256 ch, H/4)
      layer2 (512 ch, H/8)
      layer3 (1024 ch, H/16)
      layer4 (2048 ch, H/32)  — used as bottleneck

    BN layers in the backbone are frozen to preserve ImageNet statistics.
    """

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()

        # ── Pretrained ResNet50 backbone ──────────────────────────────────
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)

        # Stem: conv1 + bn1 + relu (output: 64 ch, stride 2)
        self.stem_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.stem_pool = backbone.maxpool   # stride 2 → H/4 total after stem+pool
        self.layer1 = backbone.layer1       # 256 ch, H/4
        self.layer2 = backbone.layer2       # 512 ch, H/8
        self.layer3 = backbone.layer3       # 1024 ch, H/16
        self.layer4 = backbone.layer4       # 2048 ch, H/32  (bottleneck)

        # Freeze BN parameters in the backbone to preserve ImageNet stats
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

        # ── Decoder (4 stages) ────────────────────────────────────────────
        # Stage 1: upsample from 2048 → 512, concat with layer3 skip (1024) → 512
        self.up1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(512 + 1024, 512, dropout)

        # Stage 2: upsample from 512 → 256, concat with layer2 skip (512) → 256
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256 + 512, 256, dropout)

        # Stage 3: upsample from 256 → 128, concat with layer1 skip (256) → 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 256, 128, dropout)

        # Stage 4: upsample from 128 → 64, concat with stem skip (64) → 64
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(64 + 64, 64, dropout)

        # Final upsample ×2 to recover full input resolution
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = ConvBlock(32, 32, dropout)

        # ── Head ──────────────────────────────────────────────────────────
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def _align(self, x, ref):
        """Bilinear resize x to match ref spatial dimensions if needed."""
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear",
                               align_corners=False)
        return x

    def forward(self, x):
        # Encoder
        s0 = self.stem_conv(x)                   # 64 ch, H/2
        s1 = self.layer1(self.stem_pool(s0))      # 256 ch, H/4
        s2 = self.layer2(s1)                      # 512 ch, H/8
        s3 = self.layer3(s2)                      # 1024 ch, H/16
        s4 = self.layer4(s3)                      # 2048 ch, H/32  (bottleneck)

        # Decoder
        d = self.up1(s4)
        d = self._align(d, s3)
        d = self.dec1(torch.cat([d, s3], dim=1))  # 512 ch, H/16

        d = self.up2(d)
        d = self._align(d, s2)
        d = self.dec2(torch.cat([d, s2], dim=1))  # 256 ch, H/8

        d = self.up3(d)
        d = self._align(d, s1)
        d = self.dec3(torch.cat([d, s1], dim=1))  # 128 ch, H/4

        d = self.up4(d)
        d = self._align(d, s0)
        d = self.dec4(torch.cat([d, s0], dim=1))  # 64 ch, H/2

        d = self.final_up(d)                       # 32 ch, H/1
        d = self.final_conv(d)

        return self.head(d)                        # 1 ch, H/1


class EfficientNetB3UNet(nn.Module):
    """
    U-Net with a pretrained EfficientNet-B3 encoder.

    Skip connections from EfficientNet-B3 feature stages:
      features[1]: 24 ch,  H/2  (stem after initial conv)
      features[2]: 32 ch,  H/4
      features[3]: 48 ch,  H/8
      features[5]: 136 ch, H/16
      features[7]: 384 ch, H/32  — used as bottleneck

    BN layers in the backbone are frozen to preserve ImageNet statistics.
    """

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()

        # ── Pretrained EfficientNet-B3 backbone ───────────────────────────
        backbone = tv_models.efficientnet_b3(
            weights=tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        features = backbone.features

        # Extract feature stages as separate modules
        self.stage0 = features[0]    # 40 ch,  H/2 (initial conv+bn+act)
        self.stage1 = features[1]    # 24 ch,  H/2 (MBConv1 blocks)
        self.stage2 = features[2]    # 32 ch,  H/4 (MBConv6 stride-2)
        self.stage3 = features[3]    # 48 ch,  H/8 (MBConv6 stride-2)
        self.stage4 = features[4]    # 96 ch,  H/16 (MBConv6 stride-2)
        self.stage5 = features[5]    # 136 ch, H/16 (MBConv6)
        self.stage6 = features[6]    # 232 ch, H/32 (MBConv6 stride-2)
        self.stage7 = features[7]    # 384 ch, H/32 (MBConv6)

        # Freeze BN parameters in the backbone to preserve ImageNet stats
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

        # ── Decoder (4 stages) ────────────────────────────────────────────
        # Stage 1: upsample from 384 → 136, concat with stage5 skip (136) → 256
        self.up1 = nn.ConvTranspose2d(384, 136, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(136 + 136, 256, dropout)

        # Stage 2: upsample from 256 → 128, concat with stage3 skip (48) → 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 48, 128, dropout)

        # Stage 3: upsample from 128 → 64, concat with stage2 skip (32) → 64
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + 32, 64, dropout)

        # Stage 4: upsample from 64 → 32, concat with stage1 skip (24) → 32
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(32 + 24, 32, dropout)

        # Final upsample ×2 to recover full input resolution
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = ConvBlock(16, 16, dropout)

        # ── Head ──────────────────────────────────────────────────────────
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def _align(self, x, ref):
        """Bilinear resize x to match ref spatial dimensions if needed."""
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear",
                               align_corners=False)
        return x

    def forward(self, x):
        # Encoder
        s0 = self.stage0(x)          # 40 ch, H/2
        s1 = self.stage1(s0)         # 24 ch, H/2
        s2 = self.stage2(s1)         # 32 ch, H/4
        s3 = self.stage3(s2)         # 48 ch, H/8
        s4 = self.stage4(s3)         # 96 ch, H/16
        s5 = self.stage5(s4)         # 136 ch, H/16
        s6 = self.stage6(s5)         # 232 ch, H/32
        s7 = self.stage7(s6)         # 384 ch, H/32  (bottleneck)

        # Decoder
        d = self.up1(s7)
        d = self._align(d, s5)
        d = self.dec1(torch.cat([d, s5], dim=1))  # 256 ch, H/16

        d = self.up2(d)
        d = self._align(d, s3)
        d = self.dec2(torch.cat([d, s3], dim=1))  # 128 ch, H/8

        d = self.up3(d)
        d = self._align(d, s2)
        d = self.dec3(torch.cat([d, s2], dim=1))  # 64 ch, H/4

        d = self.up4(d)
        d = self._align(d, s1)
        d = self.dec4(torch.cat([d, s1], dim=1))  # 32 ch, H/2

        d = self.final_up(d)                       # 16 ch, H/1
        d = self.final_conv(d)

        return self.head(d)                        # 1 ch, H/1


class EfficientNetB4UNet(nn.Module):
    """
    U-Net with a pretrained EfficientNet-B4 encoder.

    Skip connections from EfficientNet-B4 feature stages:
      features[0]: 48 ch,  H/2   (initial conv+bn+act)
      features[1]: 24 ch,  H/2   (MBConv1 blocks)
      features[2]: 32 ch,  H/4   (MBConv6 stride-2)
      features[3]: 56 ch,  H/8   (MBConv6 stride-2)
      features[5]: 160 ch, H/16  (MBConv6)
      features[7]: 448 ch, H/32  — used as bottleneck

    BN layers in the backbone are frozen to preserve ImageNet statistics.
    """

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()

        backbone = tv_models.efficientnet_b4(
            weights=tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        features = backbone.features

        self.stage0 = features[0]    # 48 ch,  H/2
        self.stage1 = features[1]    # 24 ch,  H/2
        self.stage2 = features[2]    # 32 ch,  H/4
        self.stage3 = features[3]    # 56 ch,  H/8
        self.stage4 = features[4]    # 112 ch, H/16
        self.stage5 = features[5]    # 160 ch, H/16
        self.stage6 = features[6]    # 272 ch, H/32
        self.stage7 = features[7]    # 448 ch, H/32  (bottleneck)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

        # Stage 1: 448 → 160, concat stage5 (160) → 256
        self.up1  = nn.ConvTranspose2d(448, 160, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(160 + 160, 256, dropout)

        # Stage 2: 256 → 128, concat stage3 (56) → 128
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 56, 128, dropout)

        # Stage 3: 128 → 64, concat stage2 (32) → 64
        self.up3  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + 32, 64, dropout)

        # Stage 4: 64 → 32, concat stage1 (24) → 32
        self.up4  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(32 + 24, 32, dropout)

        # Final ×2 to full resolution
        self.final_up   = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = ConvBlock(16, 16, dropout)
        self.head       = nn.Conv2d(16, 1, kernel_size=1)

    def _align(self, x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        s0 = self.stage0(x)          # 48 ch,  H/2
        s1 = self.stage1(s0)         # 24 ch,  H/2
        s2 = self.stage2(s1)         # 32 ch,  H/4
        s3 = self.stage3(s2)         # 56 ch,  H/8
        s4 = self.stage4(s3)         # 112 ch, H/16
        s5 = self.stage5(s4)         # 160 ch, H/16
        s6 = self.stage6(s5)         # 272 ch, H/32
        s7 = self.stage7(s6)         # 448 ch, H/32

        d = self.up1(s7);  d = self._align(d, s5)
        d = self.dec1(torch.cat([d, s5], dim=1))   # 256 ch, H/16

        d = self.up2(d);   d = self._align(d, s3)
        d = self.dec2(torch.cat([d, s3], dim=1))   # 128 ch, H/8

        d = self.up3(d);   d = self._align(d, s2)
        d = self.dec3(torch.cat([d, s2], dim=1))   # 64 ch,  H/4

        d = self.up4(d);   d = self._align(d, s1)
        d = self.dec4(torch.cat([d, s1], dim=1))   # 32 ch,  H/2

        d = self.final_up(d)
        d = self.final_conv(d)
        return self.head(d)                         # 1 ch,   H/1


class EfficientNetB1UNet(nn.Module):
    """U-Net with a pretrained EfficientNet-B1 encoder."""

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()

        backbone = tv_models.efficientnet_b1(
            weights=tv_models.EfficientNet_B1_Weights.IMAGENET1K_V1
        )
        features = backbone.features

        self.stage0 = features[0]    # 32 ch, H/2
        self.stage1 = features[1]    # 16 ch, H/2
        self.stage2 = features[2]    # 24 ch, H/4
        self.stage3 = features[3]    # 40 ch, H/8
        self.stage4 = features[4]    # 80 ch, H/16
        self.stage5 = features[5]    # 112 ch, H/16
        self.stage6 = features[6]    # 192 ch, H/32
        self.stage7 = features[7]    # 320 ch, H/32

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

        self.up1 = nn.ConvTranspose2d(320, 112, 2, 2)
        self.dec1 = ConvBlock(112 + 112, 256, dropout)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = ConvBlock(128 + 40, 128, dropout)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec3 = ConvBlock(64 + 24, 64, dropout)

        self.up4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec4 = ConvBlock(32 + 16, 32, dropout)

        self.final_up = nn.ConvTranspose2d(32, 16, 2, 2)
        self.final_conv = ConvBlock(16, 16, dropout)
        self.head = nn.Conv2d(16, 1, 1)

    def _align(self, x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        s6 = self.stage6(s5)
        s7 = self.stage7(s6)

        d = self.up1(s7)
        d = self._align(d, s5)
        d = self.dec1(torch.cat([d, s5], dim=1))

        d = self.up2(d)
        d = self._align(d, s3)
        d = self.dec2(torch.cat([d, s3], dim=1))

        d = self.up3(d)
        d = self._align(d, s2)
        d = self.dec3(torch.cat([d, s2], dim=1))

        d = self.up4(d)
        d = self._align(d, s1)
        d = self.dec4(torch.cat([d, s1], dim=1))

        d = self.final_up(d)
        d = self.final_conv(d)
        return self.head(d)


# ── Loss ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """BCE + Dice loss (equal weight) with label smoothing."""
    def __init__(self, pos_weight: float = POS_WEIGHT, label_smoothing: float = 0.01):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.label_smoothing = label_smoothing

    def dice_loss(self, logits, targets, smooth: float = 1.0):
        probs = torch.sigmoid(logits)
        num   = 2 * (probs * targets).sum() + smooth
        den   = probs.sum() + targets.sum() + smooth
        return 1 - num / den

    def forward(self, logits, targets):
        # Apply label smoothing: shift targets away from 0 and 1
        if self.label_smoothing > 0:
            targets_smooth = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        return self.bce(logits, targets_smooth) + self.dice_loss(logits, targets)


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_iou(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    use_postprocessing: bool = False,
    min_component_size: int = 0,
) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    if use_postprocessing:
        preds = remove_small_components(preds, min_component_size)
    inter = (preds * masks).sum().item()
    union = (preds + masks - preds * masks).sum().item()
    return inter / max(union, 1.0)


def _iou_from_probs(
    probs: torch.Tensor,
    masks: torch.Tensor,
    threshold: float,
    use_postprocessing: bool = False,
    min_component_size: int = 0,
) -> float:
    preds = (probs > threshold).float()
    if use_postprocessing:
        preds = remove_small_components(preds, min_component_size)
    inter = (preds * masks).sum().item()
    union = (preds + masks - preds * masks).sum().item()
    return inter / max(union, 1.0)


def remove_small_components(preds: torch.Tensor, min_size: int) -> torch.Tensor:
    preds_np = preds.cpu().numpy().astype(np.uint8)
    cleaned = np.zeros_like(preds_np)

    for b in range(preds_np.shape[0]):
        mask = preds_np[b, 0]
        visited = np.zeros_like(mask, dtype=bool)
        h, w = mask.shape

        for y in range(h):
            for x in range(w):
                if mask[y, x] == 0 or visited[y, x]:
                    continue

                stack = [(y, x)]
                component = []
                visited[y, x] = True

                while stack:
                    cy, cx = stack.pop()
                    component.append((cy, cx))

                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] == 1 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))

                if len(component) >= min_size:
                    for cy, cx in component:
                        cleaned[b, 0, cy, cx] = 1

    return torch.from_numpy(cleaned).to(preds.device).float()


def save_error_sample(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    save_dir: Path,
    epoch: int,
    sample_idx: int,
    pred_area: float,
    gt_area: float,
    iou: float,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image * std + mean).clip(0, 1)
    image = (image * 255).astype(np.uint8)

    mask = (mask_tensor.detach().cpu().numpy()[0] * 255).astype(np.uint8)
    pred = (pred_tensor.detach().cpu().numpy()[0] * 255).astype(np.uint8)
    stem = f"epoch{epoch:03d}_sample{sample_idx:03d}"

    Image.fromarray(image).save(save_dir / f"{stem}_image.png")
    Image.fromarray(mask).save(save_dir / f"{stem}_mask.png")
    Image.fromarray(pred).save(save_dir / f"{stem}_pred.png")

    info = {
        "epoch": epoch,
        "sample_idx": sample_idx,
        "pred_area": float(pred_area),
        "gt_area": float(gt_area),
        "iou": float(iou),
    }
    (save_dir / f"{stem}_info.json").write_text(json.dumps(info, indent=2))


# ── Training loop ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _artifact_suffix(location: str) -> Path | None:
    """Extract the path suffix behind 'mlruns/' from legacy MLflow locations."""
    if not location:
        return None

    normalized = location.replace("\\", "/")
    if normalized.startswith("file://"):
        normalized = unquote(urlparse(normalized).path).replace("\\", "/")
        # Windows file URIs look like "/C:/..."; strip the leading slash back off.
        if len(normalized) >= 3 and normalized[0] == "/" and normalized[2] == ":":
            normalized = normalized[1:]

    marker = "mlruns/"
    idx = normalized.lower().find(marker)
    if idx == -1:
        stripped = normalized.lstrip("./").strip("/")
        return Path() if stripped.lower() == "mlruns" else None

    suffix = normalized[idx + len(marker):].strip("/")
    return Path(suffix) if suffix else Path()


def configure_local_mlflow() -> None:
    """
    Keep MLflow on the repo-local SQLite DB and repair legacy artifact paths.

    The repository already contains old runs from another machine with absolute
    artifact paths. Rewriting them to the repo-local artifact folder prevents the
    MLflow UI from failing to fetch artifacts on this machine.
    """
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.resolve().as_posix()}")
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MLFLOW_DB.exists():
        return

    conn = sqlite3.connect(MLFLOW_DB)
    try:
        cur = conn.cursor()

        for experiment_id, location in cur.execute(
            "SELECT experiment_id, artifact_location FROM experiments"
        ).fetchall():
            suffix = _artifact_suffix(location)
            if suffix is None:
                continue

            desired_location = (MLFLOW_ARTIFACTS_DIR / suffix).resolve().as_uri()
            if location != desired_location:
                cur.execute(
                    "UPDATE experiments SET artifact_location=? WHERE experiment_id=?",
                    (desired_location, experiment_id),
                )

        for run_id, artifact_uri in cur.execute(
            "SELECT run_uuid, artifact_uri FROM runs"
        ).fetchall():
            suffix = _artifact_suffix(artifact_uri)
            if suffix is None:
                continue

            desired_uri = (MLFLOW_ARTIFACTS_DIR / suffix).resolve().as_uri()
            if artifact_uri != desired_uri:
                cur.execute(
                    "UPDATE runs SET artifact_uri=? WHERE run_uuid=?",
                    (desired_uri, run_id),
                )

        conn.commit()
    except sqlite3.OperationalError:
        # Frische DBs haben die Tabellen evtl. noch nicht, bevor MLflow sie anlegt.
        pass
    finally:
        conn.close()


MODEL_REGISTRY = {
    # Zentrale Modellliste, damit CLI-Auswahl und Vergleichslaeufe dieselbe
    # Definition verwenden.
    "unet": {
        "label": "UNet",
        "factory": lambda: UNet(dropout=DROPOUT),
    },
    "resnet34": {
        "label": "ResNet34-pretrained",
        "factory": lambda: ResNet34UNet(dropout=DROPOUT),
    },
    "resnet50": {
        "label": "ResNet50-pretrained",
        "factory": lambda: ResNet50UNet(dropout=DROPOUT),
    },
    "efficientnetb3": {
        "label": "EfficientNetB3-pretrained",
        "factory": lambda: EfficientNetB3UNet(dropout=DROPOUT),
    },
    "efficientnetb4": {
        "label": "EfficientNetB4-pretrained",
        "factory": lambda: EfficientNetB4UNet(dropout=DROPOUT),
    },
    "efficientnetb1": {
        "label": "EfficientNetB1-pretrained",
        "factory": lambda: EfficientNetB1UNet(dropout=DROPOUT),
    },
}


def get_model_names(selection: str) -> list[str]:
    """Resolve a CLI selection to one or more concrete model names."""
    if selection == "all":
        return list(MODEL_REGISTRY.keys())
    if selection not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{selection}'. Available: {', '.join(['all', *MODEL_REGISTRY.keys()])}"
        )
    return [selection]


def train(run_name: str, epochs: int, model_name: str, seed: int):
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"model_name must be one of: {', '.join(MODEL_REGISTRY.keys())}")

    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    meta = load_meta()
    pos_weight = meta.get("pos_weight_suggestion", POS_WEIGHT)
    data_fingerprint = dataset_fingerprint()
    loader_generator = torch.Generator().manual_seed(seed)

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = LitterDataset("train", crop_size=CROP_SIZE, augment=True)
    val_ds   = LitterDataset("val",   crop_size=CROP_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True,
                              persistent_workers=True, worker_init_fn=seed_worker,
                              generator=loader_generator)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True,
                              persistent_workers=True, worker_init_fn=seed_worker,
                              generator=loader_generator)

    # ── Model ─────────────────────────────────────────────────────────────
    # Das Modell wird jetzt ueber die Registry gewaehlt, damit alle
    # Architekturen mit identischem Trainingsablauf verglichen werden koennen.
    model_info = MODEL_REGISTRY[model_name]
    model = model_info["factory"]().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}")
    print(f"Model parameters: {total_params:,}")

    # ── Optimizer + Schedule ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.15,
        anneal_strategy="cos",
    )
    criterion = CombinedLoss(pos_weight=pos_weight).to(device)
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    # ── MLflow ────────────────────────────────────────────────────────────
    configure_local_mlflow()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        # Eigene Artifact-Location setzen, damit neue Runs nicht von impliziten
        # oder rechnerabhaengigen Default-Pfaden abhaengen.
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=MLFLOW_ARTIFACTS_DIR.resolve().as_uri(),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        active_run = mlflow.active_run()
        assert active_run is not None
        run_id = active_run.info.run_id
        steps_per_epoch = len(train_loader)
        optimizer_steps_target = epochs * steps_per_epoch

        mlflow.log_params({
            "model_name":        model_name,
            "batch_size":        BATCH_SIZE,
            "crop_size":         CROP_SIZE,
            "lr":                LR,
            "weight_decay":      WEIGHT_DECAY,
            "encoder_channels":  model_info["label"],
            "decoder_channels":  str(DECODER_CHANNELS),
            "dropout":           DROPOUT,
            "pos_weight":        pos_weight,
            "optimizer":         "AdamW",
            "scheduler":         "OneCycleLR",
            "loss":              "BCE+Dice",
            "total_params":      total_params,
            "device":            str(device),
            "epochs_target":     epochs,
            "steps_per_epoch":   steps_per_epoch,
            "optimizer_steps_target": optimizer_steps_target,
            "train_samples":     len(train_ds),
            "val_samples":       len(val_ds),
            "seed":              seed,
            "default_threshold": DEFAULT_THRESHOLD,
            "threshold_candidates": str(THRESHOLD_CANDIDATES),
            "use_ground_roi": USE_GROUND_ROI,
            "ground_roi_top": GROUND_ROI_TOP,
            "use_postprocessing": USE_POSTPROCESSING,
            "min_component_size": MIN_COMPONENT_SIZE,
            "use_error_analysis": USE_ERROR_ANALYSIS,
            "max_error_samples_per_epoch": MAX_ERROR_SAMPLES_PER_EPOCH,
            "false_positive_threshold": FALSE_POSITIVE_THRESHOLD,
        })
        mlflow.set_tags({
            "comparison_basis": "fixed_epochs",
            "dataset_fingerprint": data_fingerprint,
            "host": platform.node() or "unknown-host",
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "run_schema": "epoch_based_v3",
        })

        best_val_iou = 0.0
        best_epoch = 0
        best_threshold_overall = DEFAULT_THRESHOLD
        best_model_path = MODELS_DIR / f"best_model_{model_name}_{run_id}.pth"
        best_threshold_path = MODELS_DIR / f"best_threshold_{model_name}_{run_id}.json"
        error_analysis_dir = REPO_ROOT / ERROR_ANALYSIS_DIR
        if USE_ERROR_ANALYSIS:
            error_analysis_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        step = 0

        # Die Schleife ist jetzt explizit epochenbasiert statt indirekt über
        # einen Abbruch mitten in einer while-Schleife.
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            train_iou  = 0.0

            for images, masks in train_loader:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=amp_enabled):
                    logits = model(images)
                    loss   = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()
                train_iou  += compute_iou(
                    logits,
                    masks,
                    threshold=DEFAULT_THRESHOLD,
                    use_postprocessing=USE_POSTPROCESSING,
                    min_component_size=MIN_COMPONENT_SIZE,
                )
                step += 1

            # ── Validation ────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            val_iou  = 0.0
            error_candidates = []
            threshold_scores = {threshold: 0.0 for threshold in THRESHOLD_CANDIDATES}
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks  = masks.to(device,  non_blocking=True)
                    logits = model(images)
                    probs = torch.sigmoid(logits)

                    if USE_ERROR_ANALYSIS:
                        preds = (probs > DEFAULT_THRESHOLD).float()
                        if USE_POSTPROCESSING:
                            preds = remove_small_components(preds, MIN_COMPONENT_SIZE)

                        for b in range(images.size(0)):
                            pred_area = preds[b].sum().item()
                            gt_area = masks[b].sum().item()
                            if gt_area < 1.0 and pred_area > FALSE_POSITIVE_THRESHOLD * preds[b].numel():
                                iou = _iou_from_probs(
                                    probs[b:b + 1],
                                    masks[b:b + 1],
                                    DEFAULT_THRESHOLD,
                                    USE_POSTPROCESSING,
                                    MIN_COMPONENT_SIZE,
                                )
                                error_candidates.append((
                                    iou,
                                    images[b],
                                    masks[b],
                                    preds[b],
                                    pred_area,
                                    gt_area,
                                ))

                    val_loss += criterion(logits, masks).item()
                    val_iou  += _iou_from_probs(
                        probs,
                        masks,
                        DEFAULT_THRESHOLD,
                        USE_POSTPROCESSING,
                        MIN_COMPONENT_SIZE,
                    )
                    for threshold in THRESHOLD_CANDIDATES:
                        threshold_scores[threshold] += _iou_from_probs(
                            probs,
                            masks,
                            threshold,
                            USE_POSTPROCESSING,
                            MIN_COMPONENT_SIZE,
                        )

            n_train = len(train_loader)
            n_val   = len(val_loader)
            train_loss_mean = train_loss / max(n_train, 1)
            train_iou_mean = train_iou / max(n_train, 1)
            val_loss_mean = val_loss / max(n_val, 1)
            val_iou_mean = val_iou / max(n_val, 1)
            elapsed = time.time() - t0
            threshold_scores = {t: v / max(n_val, 1) for t, v in threshold_scores.items()}
            best_threshold = max(threshold_scores, key=threshold_scores.get)
            best_threshold_iou = threshold_scores[best_threshold]

            metrics = {
                "train_loss": train_loss_mean,
                "train_iou":  train_iou_mean,
                "val_loss":   val_loss_mean,
                "val_iou":    val_iou_mean,
                "epoch":      epoch,
                "elapsed_s":  elapsed,
                "lr":         scheduler.get_last_lr()[0],
                "best_threshold": best_threshold,
                "best_threshold_iou": best_threshold_iou,
            }
            # MLflow-Step ebenfalls auf Epoche umstellen, damit UI und Logging
            # dieselbe Zeitskala verwenden.
            mlflow.log_metrics(metrics, step=epoch)

            if best_threshold_iou > best_val_iou:
                best_val_iou = best_threshold_iou
                best_epoch = epoch
                best_threshold_overall = best_threshold
                torch.save(model.state_dict(), best_model_path)
                mlflow.set_tag("best_checkpoint", best_model_path.name)
                best_threshold_path.write_text(json.dumps({"threshold": best_threshold_overall}, indent=2))

            print(
                f"epoch {epoch:3d}/{epochs:3d}  "
                f"train_loss={train_loss_mean:.4f}  "
                f"train_iou={train_iou_mean:.4f}  "
                f"val_loss={val_loss_mean:.4f}  "
                f"val_iou={val_iou_mean:.4f}  "
                f"best_threshold={best_threshold:.2f}  "
                f"best_threshold_iou={best_threshold_iou:.4f}  "
                f"best_val_iou={best_val_iou:.4f}@{best_epoch}"
            )

            if USE_ERROR_ANALYSIS and error_candidates:
                error_candidates.sort(key=lambda x: x[0])
                epoch_error_dir = error_analysis_dir / f"epoch_{epoch:03d}"
                for sample_idx, (iou, image_t, mask_t, pred_t, pred_area, gt_area) in enumerate(
                    error_candidates[:MAX_ERROR_SAMPLES_PER_EPOCH]
                ):
                    save_error_sample(
                        image_tensor=image_t,
                        mask_tensor=mask_t,
                        pred_tensor=pred_t,
                        save_dir=epoch_error_dir,
                        epoch=epoch,
                        sample_idx=sample_idx,
                        pred_area=pred_area,
                        gt_area=gt_area,
                        iou=iou,
                    )

        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), artifact_path="checkpoints")
        if best_threshold_path.exists():
            mlflow.log_artifact(str(best_threshold_path), artifact_path="checkpoints")

        mlflow.log_metrics({
            "best_val_iou": float(best_val_iou),
            "best_epoch": float(best_epoch),
            "best_threshold_final": float(best_threshold_overall),
            "epochs_completed": float(epochs),
            "optimizer_steps_completed": float(step),
        }, step=epochs)
        mlflow.log_dict({
            "run_id": run_id,
            "run_name": run_name,
            "model_name": model_name,
            "comparison_basis": "fixed_epochs",
            "dataset_fingerprint": data_fingerprint,
            "seed": seed,
            "epochs_target": epochs,
            "epochs_completed": epochs,
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps_target": optimizer_steps_target,
            "optimizer_steps_completed": step,
            "best_epoch": best_epoch,
            "best_val_iou": best_val_iou,
            "best_threshold": best_threshold_overall,
            "best_checkpoint": best_model_path.name,
        }, "comparison_manifest.json")
        print(f"\nBest val_iou: {best_val_iou:.4f}")
        print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",   default="baseline",
                        help="MLflow run name")
    parser.add_argument("--epochs", "--epochen", dest="epochs", type=int,
                        default=DEFAULT_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed used for fair, reproducible comparisons")
    parser.add_argument(
        "--model",
        default="efficientnetb1",
        help="Model to train: all, unet, resnet34, resnet50, efficientnetb3, efficientnetb4, efficientnetb1",
    )
    args = parser.parse_args()

    selected_models = get_model_names(args.model)
    for model_name in selected_models:
        # Bei Sammellaeufen einen eindeutigen Run-Namen pro Modell erzeugen,
        # damit MLflow die Vergleichslaeufe sauber auseinanderhalten kann.
        run_name = args.run_name if len(selected_models) == 1 else f"{args.run_name}-{model_name}"
        train(run_name=run_name, epochs=args.epochs, model_name=model_name, seed=args.seed)
