import torch
import random
import numpy as np
import os

from src.configs import TrainingConfig
from src.models.embedding_model import EmbeddingModel
from src.losses.proxy_anchor import ProxyLoss
from src.engine import Trainer

def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Load configuration
    config = TrainingConfig()
    
    # Set seed for reproducibility
    set_seed(config.SEED)

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Initialize Model, Loss, and Trainer
    model = EmbeddingModel(config)
    criterion = ProxyLoss(config)
    trainer = Trainer(model, criterion, config, "proxy")

    # Run the training curriculum
    trainer.run_training_curriculum()

if __name__ == "__main__":
    main()
