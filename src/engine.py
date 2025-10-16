import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import torch.optim as optim
from src.optimizer.sam import SAM
from src.data.dataset import create_dataloaders
from src.models.embedding_model import EmbeddingModel
from src.losses.hybrid_loss import HybridLoss
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchmetrics import MetricCollection

class Trainer:
    """
    Handles the model training curriculum, including multi-stage training,
    optimization, validation, and metric reporting.
    """
    def __init__(self, model, criterion, config):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The embedding model to be trained.
            criterion (nn.Module): The loss function.
            config (TrainingConfig): A configuration object with training parameters.
        """
        self.config = config
        self.model = model.to(config.DEVICE)
        self.loss_fn = criterion
        self.optimizer = None
        self.scheduler = None

        # --- METRICS INITIALIZATION ---
        # Initialize metrics for both training and validation phases
        self.train_metrics = self._create_metrics_collection().to(config.DEVICE)
        self.val_metrics = self._create_metrics_collection().to(config.DEVICE)

    def _create_metrics_collection(self):   
        """Helper function to create a standardized collection of metrics."""
        return MetricCollection({
            'accuracy': MulticlassAccuracy(num_classes=self.config.NUM_CLASSES, average='macro'),
            'precision': MulticlassPrecision(num_classes=self.config.NUM_CLASSES, average='macro'),
            'recall': MulticlassRecall(num_classes=self.config.NUM_CLASSES, average='macro'),
            'f1_score': MulticlassF1Score(num_classes=self.config.NUM_CLASSES, average='macro')
        })

    def _get_optimizer(self, stage_config):
        """
        Initializes the optimizer with differential learning rates for different
        parts of the model. Uses SAM optimizer in later stages.
        """
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': stage_config.get('base_lr', stage_config['lr'])},
            {'params': self.model.cbam_stages.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
            {'params': self.model.pwca.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
            {'params': self.model.embedding_head.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
            {'params': self.model.arcface_head.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
        ]

        base_optimizer = optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)

        # Use Sharpness-Aware Minimization (SAM) for more robust training in later stages
        if 'base_lr' in stage_config:
            print("Using SAM optimizer.")
            return SAM(self.model.parameters(), base_optimizer, rho=self.config.SAM_RHO)
        else:
            print("Using AdamW optimizer.")
            return base_optimizer

    def _adjust_learning_rate(self, stage_config):
        """
        Dynamically adjusts the learning rate for each parameter group at the
        start of a new training stage, preserving optimizer state.
        """
        general_lr = stage_config.get('lr')

        base_lr = stage_config.get('base_lr', general_lr)
        head_lr = stage_config.get('head_lr', general_lr)

        if base_lr is None or head_lr is None:
            raise ValueError("Learning rate configuration is missing. "
                             "Provide either a general 'lr' or specific 'base_lr' and 'head_lr'.")

        for i, param_group in enumerate(self.optimizer.param_groups):
             if i == 0: # Assuming the first group is the backbone
                 param_group['lr'] = base_lr
             else:
                 param_group['lr'] = head_lr
                 
        print(f"Learning rates adjusted for the new stage: Backbone LR = {base_lr}, Head LR = {head_lr}")

    def _train_one_epoch(self, dataloader):
        """Runs a single training epoch."""
        self.model.train()
        self.train_metrics.reset()
        train_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for images, labels, distractor_images in progress_bar:
            images, labels, distractor_images = images.to(self.config.DEVICE), labels.to(self.config.DEVICE), distractor_images.to(self.config.DEVICE)

            def forward_pass():
                return self.model(images, labels, distractor_images)

            if isinstance(self.optimizer, SAM):
                outputs = forward_pass()
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                outputs_2 = forward_pass()
                loss_2 = self.loss_fn(outputs_2, labels)
                loss_2.backward()
                self.optimizer.second_step(zero_grad=True)
                final_outputs, loss_val = outputs_2, loss_2.item()
            else:
                self.optimizer.zero_grad()
                outputs = forward_pass()
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                final_outputs, loss_val = outputs, loss.item()

            preds = final_outputs['logits'].detach()
            self.train_metrics.update(preds, labels)
            train_loss += loss_val
            progress_bar.set_postfix(loss=loss_val)

        avg_loss = train_loss / len(dataloader)
        metrics_output = self.train_metrics.compute()
        scalar_metrics = {k: v.item() for k, v in metrics_output.items()}
        return avg_loss, scalar_metrics

    def _validate_one_epoch(self, dataloader):
        """Runs a single validation epoch."""
        self.model.eval()
        self.val_metrics.reset()
        val_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, labels, distractor_images in progress_bar:
                images, labels, distractor_images = (
                    images.to(self.config.DEVICE),
                    labels.to(self.config.DEVICE),
                    distractor_images.to(self.config.DEVICE)
                )

                outputs = self.model(images, labels, distractor_images)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()

                preds = outputs['logits'].detach()
                self.val_metrics.update(preds, labels)

        avg_loss = val_loss / len(dataloader)
        metrics_output = self.val_metrics.compute()
        scalar_metrics = {k: v.item() for k, v in metrics_output.items()}
        return avg_loss, scalar_metrics

    def _print_metrics(self, stage_name, epoch, num_epochs, train_loss, train_metrics, val_loss, val_metrics):
        """Prints a formatted summary of the epoch's metrics."""
        print(f"\n--- {stage_name} | Epoch {epoch+1}/{num_epochs} ---")
        print(f"  Train -> Loss: {train_loss:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"Precision: {train_metrics['precision']:.4f} | "
              f"Recall: {train_metrics['recall']:.4f} | "
              f"F1: {train_metrics['f1_score']:.4f}")
        print(f"  Valid -> Loss: {val_loss:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1_score']:.4f}")
        print("-" * (len(stage_name) + 20))

    def run_training_curriculum(self):
        """Executes the full multi-stage training curriculum."""
        print("--- Starting Training Curriculum ---")

        # --- Stage 1: Head Warm-up ---
        # print("\n--- STAGE 1: Head Warm-up ---")
        stage1_config = {
            'epochs': self.config.STAGE1_EPOCHS,
            'lr': self.config.STAGE1_LR,
            'img_size': self.config.STAGE1_IMG_SIZE,
            'batch_size': self.config.STAGE1_BATCH_SIZE,
            'aug_strength': self.config.STAGE1_AUG_STRENGTH
        }
        # train_loader, val_loader = create_dataloaders(self.config, stage1_config)
        # self.model.freeze_backbone()
        
        self.optimizer = self._get_optimizer(stage1_config)
        
        # for epoch in range(stage1_config['epochs']):
        #     train_loss, train_metrics = self._train_one_epoch(train_loader)
        #     val_loss, val_metrics = self._validate_one_epoch(val_loader)
        #     self._print_metrics("STAGE 1: Head Warm-up", epoch, stage1_config['epochs'], train_loss, train_metrics, val_loss, val_metrics)

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
        
        self._adjust_learning_rate(stage2_config)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=stage2_config['epochs'])

        for epoch in range(stage2_config['epochs']):
            train_loss, train_metrics = self._train_one_epoch(train_loader)
            val_loss, val_metrics = self._validate_one_epoch(val_loader)
            self.scheduler.step()
            self._print_metrics("STAGE 2: Early Fine-Tuning", epoch, stage2_config['epochs'], train_loss, train_metrics, val_loss, val_metrics)

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
        
        self._adjust_learning_rate(stage3_config)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=stage3_config['epochs'])

        for epoch in range(stage3_config['epochs']):
            train_loss, train_metrics = self._train_one_epoch(train_loader)
            val_loss, val_metrics = self._validate_one_epoch(val_loader)
            self.scheduler.step()
            self._print_metrics("STAGE 3: Final Polishing", epoch, stage3_config['epochs'], train_loss, train_metrics, val_loss, val_metrics)

        print("\n--- Training Finished ---")