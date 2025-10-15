# ===================================================================
# Project specific settings
# ===================================================================
DATA_DIR = "./dataset"
NUM_CLASSES = 118 # As per your dataset description
DEVICE = "cuda"

# ===================================================================
# Model Architecture Settings
# ===================================================================
BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"
EMBEDDING_DIM = 512

# Sub-center ArcFace settings
ARCFACE_SUB_CENTERS_K = 3
ARCFACE_SCALE = 64.0
ARCFACE_MARGIN = 0.50

# ===================================================================
# Loss Function Settings
# ===================================================================
# Proxy-Anchor settings
PROXY_ANCHOR_ALPHA = 32.0
PROXY_ANCHOR_DELTA = 0.1

# Hybrid loss weight
HYBRID_LOSS_ALPHA = 1.0

# NEW: Label Smoothing setting for the classification loss
LABEL_SMOOTHING = 0.1

# ===================================================================
# Training Settings
# ===================================================================
# General
BATCH_SIZE = 32
NUM_EPOCHS = 40
NUM_WORKERS = 4

# Optimizer settings
BASE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 5e-4
SAM_RHO = 0.05 # For the SAM optimizer

# Training curriculum stages
HEAD_WARMUP_EPOCHS = 3
EARLY_STAGE_EPOCHS = 7 # Head warmup + this = 10 epochs
# The rest will be the final polishing stage

# Image sizes for progressive learning
LOW_RES = 224
MID_RES = 320
HIGH_RES = 384

# ===================================================================
# Augmentation Settings
# ===================================================================
MIXUP_CUTMIX_ALPHA_LOW = 0.2
MIXUP_CUTMIX_ALPHA_HIGH = 1.0

