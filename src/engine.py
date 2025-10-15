import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import torch.optim as optim
from src.optimizer.sam import SAM
from src.data.dataset import create_dataloaders

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = EmbeddingModel(config).to(config.DEVICE)
        self.loss_fn = HybridLoss(config)
        self.optimizer = None # Initialized in stages
        self.scheduler = None # Initialized in stages

    def _get_optimizer(self, stage_config):
        # Differential learning rates
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': stage_config.get('base_lr', stage_config['lr'])},
            {'params': self.model.cbam_stages.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
            {'params': self.model.pwca.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
            {'params': self.model.embedding_head.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
            {'params': self.model.arcface_head.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
        ]
        
        base_optimizer = optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)
        
        # Use SAM for stages 2 and 3
        if 'base_lr' in stage_config:
            return SAM(self.model.parameters(), base_optimizer, rho=self.config.SAM_RHO)
        else:
            return base_optimizer

    def _train_one_epoch(self, dataloader):
        self.model.train()
        train_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for images, labels, distractor_images in progress_bar:
            images, labels, distractor_images = images.to(self.config.DEVICE), labels.to(self.config.DEVICE), distractor_images.to(self.config.DEVICE)

            if isinstance(self.optimizer, SAM):
                # First forward-backward pass for SAM
                outputs = self.model(images, labels, distractor_images)
                loss = self.loss_fn(outputs['embedding'], outputs['logits'], labels)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                # Second forward-backward pass for SAM
                outputs_2 = self.model(images, labels, distractor_images)
                loss_2 = self.loss_fn(outputs_2['embedding'], outputs_2['logits'], labels)
                loss_2.backward()
                self.optimizer.second_step(zero_grad=True)
                
                loss_val = loss_2.item()
            else:
                self.optimizer.zero_grad()
                outputs = self.model(images, labels, distractor_images)
                loss = self.loss_fn(outputs['embedding'], outputs['logits'], labels)
                loss.backward()
                self.optimizer.step()
                loss_val = loss.item()

            train_loss += loss_val
            progress_bar.set_postfix(loss=loss_val)
        
        return train_loss / len(dataloader)

    def _validate_one_epoch(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        all_embeddings = []
        all_labels = []
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, labels, _ in progress_bar:
                images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)
                outputs = self.model(images, labels)
                loss = self.loss_fn(outputs['embedding'], outputs['logits'], labels)
                val_loss += loss.item()

                # =================== FIX IS HERE ===================
                # Move tensors to CPU before appending to the list.
                # This frees up GPU VRAM and prevents the memory leak.
                all_embeddings.append(outputs['embedding'].cpu())
                all_labels.append(labels.cpu())
                # ===================================================

        # Here you would typically calculate metrics like mAP, Recall@K, etc.
        # For now, we just return the loss.
        # all_embeddings = torch.cat(all_embeddings)
        # all_labels = torch.cat(all_labels)

        return val_loss / len(dataloader)


    def run_training_curriculum(self):
        print("--- Starting Training Curriculum ---")

        # --- Stage 1: Head Warm-up ---
        print("\n--- STAGE 1: Head Warm-up ---")
        stage1_config = {
            'epochs': self.config.STAGE1_EPOCHS,
            'lr': self.config.STAGE1_LR,
            'img_size': self.config.STAGE1_IMG_SIZE,
            'batch_size': self.config.STAGE1_BATCH_SIZE,
            'aug_strength': self.config.STAGE1_AUG_STRENGTH
        }
        train_loader, val_loader = create_dataloaders(self.config, stage1_config)
        self.model.freeze_backbone()
        print("Freezing backbone.")
        self.optimizer = self._get_optimizer(stage1_config)
        
        for epoch in range(stage1_config['epochs']):
            print(f"Epoch {epoch+1}/{stage1_config['epochs']}")
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._validate_one_epoch(val_loader)
            print(f"Stage 1 -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # --- Stage 2: Early Full Fine-Tuning ---
        print("\n--- STAGE 2: Early Full Fine-Tuning ---")
        stage2_config = {
            'epochs': self.config.STAGE2_EPOCHS,
            'base_lr': self.config.STAGE2_BASE_LR,
            'head_lr': self.config.STAGE2_HEAD_LR,
            'img_size': self.config.STAGE2_IMG_SIZE,
            'batch_size': self.config.STAGE2_BATCH_SIZE,
            'aug_strength': self.config.STAGE2_AUG_STRENGTH
        }
        train_loader, val_loader = create_dataloaders(self.config, stage2_config)
        self.model.unfreeze_backbone()
        print("Unfreezing backbone.")
        self.optimizer = self._get_optimizer(stage2_config)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=stage2_config['epochs'])

        for epoch in range(stage2_config['epochs']):
            print(f"Epoch {epoch+1}/{stage2_config['epochs']}")
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._validate_one_epoch(val_loader)
            self.scheduler.step()
            print(f"Stage 2 -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # --- Stage 3: Final High-Resolution Polishing ---
        print("\n--- STAGE 3: Final High-Resolution Polishing ---")
        stage3_config = {
            'epochs': self.config.STAGE3_EPOCHS,
            'base_lr': self.config.STAGE3_BASE_LR,
            'head_lr': self.config.STAGE3_HEAD_LR,
            'img_size': self.config.STAGE3_IMG_SIZE,
            'batch_size': self.config.STAGE3_BATCH_SIZE,
            'aug_strength': self.config.STAGE3_AUG_STRENGTH
        }
        train_loader, val_loader = create_dataloaders(self.config, stage3_config)
        self.optimizer = self._get_optimizer(stage3_config)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=stage3_config['epochs'])

        for epoch in range(stage3_config['epochs']):
            print(f"Epoch {epoch+1}/{stage3_config['epochs']}")
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._validate_one_epoch(val_loader)
            self.scheduler.step()
            print(f"Stage 3 -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        print("\n--- Training Finished ---")
