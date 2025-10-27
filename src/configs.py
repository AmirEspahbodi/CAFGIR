class TrainingConfigBase:
    DATA_DIR = "D:\\amir_es\\car_accessories_dataset_augmented"
    NUM_CLASSES = 413 # As per your dataset description
    DEVICE = "cuda"

    FBACKBONE = "facebook/convnextv2-tiny-22k-384"
    # facebook/convnextv2-base-22k-384
    
    TBACKBONE = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
    # convnextv2_base.fcmae_ft_in22k_in1k_384
    
    EMBEDDING_DIM = 512

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
    STAGE1_EPOCHS = 4
    STAGE1_LR = 1e-3
    STAGE1_IMG_SIZE = 224
    STAGE1_BATCH_SIZE = 64  # Can be larger due to smaller image size
    STAGE1_ACCUMULATION_STEPS = 1
    STAGE1_AUG_STRENGTH = 0.0 # No Mixup/CutMix during warm-up

    # --- Stage 2: Early Full Fine-Tuning ---
    STAGE2_EPOCHS = 10
    STAGE2_BASE_LR = 1e-5 # Differential LR for backbone
    STAGE2_HEAD_LR = 1e-4 # Differential LR for head
    STAGE2_IMG_SIZE = 320
    STAGE2_BATCH_SIZE = 32 # Must be STAGE2_SAMPLER_P * STAGE2_SAMPLER_K
    STAGE2_ACCUMULATION_STEPS = 1
    STAGE2_AUG_STRENGTH = 0.2 # Mild Mixup/CutMix alpha
    # --- New settings for Stage 2 ---
    STAGE2_SAMPLER_P = 8 # Number of classes per batch
    STAGE2_SAMPLER_K = 4 # Number of images per class (P * K = 16)
    STAGE2_WARMUP_EPOCHS = 1 # Linear warm-up for the first epoch
   
    # --- Stage 3: Final High-Resolution Polishing ---
    STAGE3_EPOCHS = 15
    STAGE3_BASE_LR = 1e-6 # Lower LR for final polishing
    STAGE3_HEAD_LR = 1e-5
    STAGE3_IMG_SIZE = 384
    STAGE3_BATCH_SIZE = 16 # Must be STAGE3_SAMPLER_P * STAGE3_SAMPLER_K
    STAGE3_ACCUMULATION_STEPS = 4
    STAGE3_AUG_STRENGTH = 1.0 # Strong Mixup/CutMix alpha
    # --- New settings for Stage 3 ---
    STAGE3_SAMPLER_P = 4 # Number of classes per batch
    STAGE3_SAMPLER_K = 4 # Number of images per class (P * K = 8)
    STAGE3_WARMUP_EPOCHS = 1 # Linear warm-up for the first epoch
    
    PIN_MEMORY = True
    
    SEED=42
    OUTPUT_DIR='./output'
    CHECKPOINT_DIR = './checkpoints'

    RESUME = True  # Set to True to resume training from a checkpoint
    # Example: RESUME_CHECKPOINT_PATH = "checkpoints/best_model_stage2_epoch3_acc0.8500.pth"
    RESUME_CHECKPOINT_PATH = "D:\\amir_es\\CAFGIR\\checkpoints\\tiny_proxy_best_model_stage1_epoch4_R1@0.0372.pth" 
    RESUME_STAGE = 1            # The stage number (1, 2, or 3) to resume from
    RESUME_EPOCH = 4            # The epoch number within that stage to resume from. (0-indexed)

    GEM_P_INIT = 3.0 # Initial 'p' value for GeM pooling. 1.0 = AvgPool, >1.0 = emphasizes salient features

    # Multi-Scale Feature Settings
    # Intermediate dimension for the 2-layer embedding head
    EMBEDDING_HEAD_INTERMEDIATE_DIM = 1024 


class TrainingConfigTiny(TrainingConfigBase):
    BASE_MODEL = "tiny"
    pass


class TrainingConfigLarge(TrainingConfigBase): # Renamed from 'Base' to 'Large'
    BASE_MODEL = "base"
    
    # Recalculate P/K for different batch sizes if needed
    STAGE1_BATCH_SIZE = 32 # P=8, K=4 (or disable P-K for stage 1)
    
    STAGE2_BATCH_SIZE = 16 # P=8, K=2
    STAGE2_SAMPLER_P = 8
    STAGE2_SAMPLER_K = 2
    STAGE2_ACCUMULATION_STEPS = 4

    STAGE3_BATCH_SIZE = 8 # P=4, K=2
    STAGE3_SAMPLER_P = 4
    STAGE3_SAMPLER_K = 2
    STAGE3_ACCUMULATION_STEPS = 8

    STAGE1_EPOCHS = 7
    STAGE2_EPOCHS = 15
    STAGE3_EPOCHS = 20

    TBACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"


class TrainingConfigDinoV2Base(TrainingConfigBase):
    # This is the new backbone you want to use
    TBACKBONE = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
    BASE_MODEL = "dono_v2"
    
    STAGE1_BATCH_SIZE = 32 # P=8, K=4 (or disable P-K for stage 1)
    
    STAGE2_BATCH_SIZE = 16 # P=8, K=2
    STAGE2_SAMPLER_P = 8
    STAGE2_SAMPLER_K = 2
    # Increased accumulation steps for the larger DINOv2 model
    STAGE2_ACCUMULATION_STEPS = 4

    STAGE3_BATCH_SIZE = 8 # P=4, K=2
    STAGE3_SAMPLER_P = 4
    STAGE3_SAMPLER_K = 2
    # Increased accumulation steps for the larger DINOv2 model
    STAGE3_ACCUMULATION_STEPS = 8

    # Increased epochs, as DINOv2 may benefit from longer fine-tuning
    STAGE1_EPOCHS = 7
    STAGE2_EPOCHS = 15
    STAGE3_EPOCHS = 20

    STAGE1_IMG_SIZE = 224
    STAGE2_IMG_SIZE = 518
    STAGE3_IMG_SIZE = 518



TrainingConfig = TrainingConfigTiny
