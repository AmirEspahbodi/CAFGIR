import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import torch.optim as optim
from src.optimizer.sam import SAM
from src.data.dataset import create_dataloaders

class Trainer:
    """The main training engine that orchestrates the entire curriculum."""
    def __init__(self, model, criterion, config):
        self.model = model.to(config.DEVICE)
        self.criterion = criterion.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE

    def train_one_epoch(self, dataloader, optimizer, use_sam=False):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for anchor_img, labels, distractor_img in progress_bar:
            anchor_img, labels, distractor_img = \
                anchor_img.to(self.device), labels.to(self.device), distractor_img.to(self.device)

            if use_sam:
                # First forward/backward pass for SAM
                outputs = self.model(anchor_img, labels, distractor_img) # <-- Pass labels here
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # Second forward/backward pass for SAM
                outputs = self.model(anchor_img, labels, distractor_img) # <-- and here
                self.criterion(outputs, labels).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                outputs = self.model(anchor_img, labels, distractor_img) # <-- and here
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            outputs = self.model(imgs, labels) # <-- Pass labels here too
            # Use the classification part of the loss for validation
            loss = self.criterion.classification_loss(outputs['logits'], labels)
            total_loss += loss.item()
        return total_loss / len(dataloader)
    


    def run_training_curriculum(self):
        """Executes the full multi-stage training curriculum."""
        print("--- Starting Training Curriculum ---")

        # --- Stage 1: Head Warm-up ---
        print("\n--- STAGE 1: Head Warm-up ---")
        self.model.freeze_backbone()
        stage1_config = {
            'img_size': self.config.STAGE1_IMG_SIZE,
            'batch_size': self.config.STAGE1_BATCH_SIZE,
            'aug_strength': self.config.STAGE1_AUG_STRENGTH
        }
        train_loader, val_loader = create_dataloaders(self.config, stage1_config)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.config.STAGE1_LR
        )
        for epoch in range(self.config.STAGE1_EPOCHS):
            train_loss = self.train_one_epoch(train_loader, optimizer, use_sam=False)
            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{self.config.STAGE1_EPOCHS} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # --- Stage 2: Early Full Fine-Tuning ---
        print("\n--- STAGE 2: Early Full Fine-Tuning with SAM ---")
        self.model.unfreeze_backbone()
        stage2_config = {
            'img_size': self.config.STAGE2_IMG_SIZE,
            'batch_size': self.config.STAGE2_BATCH_SIZE,
            'aug_strength': self.config.STAGE2_AUG_STRENGTH
        }
        train_loader, val_loader = create_dataloaders(self.config, stage2_config)
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': self.config.STAGE2_LR_BACKBONE},
            {'params': self.model.cbam_stages.parameters(), 'lr': self.config.STAGE2_LR_HEAD},
            {'params': self.model.pwca.parameters(), 'lr': self.config.STAGE2_LR_HEAD},
            {'params': self.model.embedding_head.parameters(), 'lr': self.config.STAGE2_LR_HEAD},
            {'params': self.model.arcface_head.parameters(), 'lr': self.config.STAGE2_LR_HEAD},
        ]
        base_optimizer = optim.AdamW
        optimizer = SAM(param_groups, base_optimizer, rho=self.config.SAM_RHO)
        for epoch in range(self.config.STAGE2_EPOCHS):
            train_loss = self.train_one_epoch(train_loader, optimizer, use_sam=True)
            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{self.config.STAGE2_EPOCHS} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # --- Stage 3: Final High-Resolution Polishing ---
        print("\n--- STAGE 3: Final Polishing with SAM and Scheduler ---")
        stage3_config = {
            'img_size': self.config.STAGE3_IMG_SIZE,
            'batch_size': self.config.STAGE3_BATCH_SIZE,
            'aug_strength': self.config.STAGE3_AUG_STRENGTH
        }
        train_loader, val_loader = create_dataloaders(self.config, stage3_config)
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': self.config.STAGE3_LR_BACKBONE},
            {'params': self.model.cbam_stages.parameters(), 'lr': self.config.STAGE3_LR_HEAD},
            {'params': self.model.pwca.parameters(), 'lr': self.config.STAGE3_LR_HEAD},
            {'params': self.model.embedding_head.parameters(), 'lr': self.config.STAGE3_LR_HEAD},
            {'params': self.model.arcface_head.parameters(), 'lr': self.config.STAGE3_LR_HEAD},
        ]
        base_optimizer = optim.AdamW
        optimizer = SAM(param_groups, base_optimizer, rho=self.config.SAM_RHO)
        scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=self.config.SCHEDULER_T_MAX, eta_min=self.config.SCHEDULER_ETA_MIN)

        for epoch in range(self.config.STAGE3_EPOCHS):
            train_loss = self.train_one_epoch(train_loader, optimizer, use_sam=True)
            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{self.config.STAGE3_EPOCHS} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.7f}")
            scheduler.step()

        # Save the final model
        final_model_path = os.path.join(self.config.OUTPUT_DIR, f"{self.config.MODEL_NAME}.pth")
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), final_model_path)
        print(f"\nTraining complete. Model saved to {final_model_path}")

