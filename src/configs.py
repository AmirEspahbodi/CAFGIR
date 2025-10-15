class TrainingConfig:
    """
    A centralized configuration class for the entire training pipeline.
    """
    # ===================================================================
    # Project specific settings
    # ===================================================================
    DATA_DIR = "/content/CAFGIR/augmented_car_dataset/"
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

    # Label Smoothing setting for the classification loss
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

    # Label Smoothing setting for the classification loss
    LABEL_SMOOTHING = 0.1

    # ===================================================================
    # Optimizer Settings
    # ===================================================================
    BASE_LR = 1e-4
    HEAD_LR = 1e-3
    WEIGHT_DECAY = 5e-4
    SAM_RHO = 0.05 # For the SAM optimizer

    # ===================================================================
    # Training Curriculum Stages
    # ===================================================================
    NUM_WORKERS = 4

    # --- Stage 1: Head Warm-up ---
    STAGE1_EPOCHS = 3
    STAGE1_IMG_SIZE = 224
    STAGE1_BATCH_SIZE = 48  # Can be larger due to smaller image size
    STAGE1_AUG_STRENGTH = 0.0 # No Mixup/CutMix during warm-up

    # --- Stage 2: Early Full Fine-Tuning ---
    STAGE2_EPOCHS = 7
    STAGE2_IMG_SIZE = 320
    STAGE2_BATCH_SIZE = 32
    STAGE2_AUG_STRENGTH = 0.2 # Mild Mixup/CutMix alpha

    # --- Stage 3: Final High-Resolution Polishing ---
    STAGE3_EPOCHS = 30
    STAGE3_IMG_SIZE = 384
    STAGE3_BATCH_SIZE = 32 # May need to be reduced based on VRAM
    STAGE3_AUG_STRENGTH = 1.0 # Strong Mixup/CutMix alpha
    PIN_MEMORY = True
    
    SEED=42
    OUTPUT_DIR='./output'