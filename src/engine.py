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
# Matplotlib is imported inside the plotting function to make it an optional dependency
# import matplotlib.pyplot as plt

class Trainer:
    """
    Handles the model training curriculum, including multi-stage training,
    optimization, validation, metric reporting, and checkpointing with resume capability.
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

        self.train_metrics = self._create_metrics_collection().to(config.DEVICE)
        self.val_metrics = self._create_metrics_collection().to(config.DEVICE)

        self.best_val_accuracy = 0.0
        self.checkpoint_dir = getattr(config, 'CHECKPOINT_DIR', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"✅ Checkpoints will be saved to '{self.checkpoint_dir}'")

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def _save_checkpoint(self, stage_num, epoch, val_metrics):
        """Saves the model checkpoint if the current validation accuracy is the best so far."""
        current_accuracy = val_metrics['accuracy']
        if current_accuracy >= self.best_val_accuracy:
            self.best_val_accuracy = current_accuracy
            checkpoint_filename = f"best_model_stage{stage_num}_epoch{epoch+1}_acc{current_accuracy:.4f}.pth"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            
            checkpoint = {
                'stage_num': stage_num,
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_accuracy': self.best_val_accuracy,
                'history': self.history
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"\n✨ New best model found! Accuracy improved to {current_accuracy:.4f}. Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self):
        """Loads a checkpoint for resuming training."""
        if not (self.config.RESUME and self.config.RESUME_CHECKPOINT_PATH):
            print("\n>> Starting training from scratch.")
            return None
        
        if not os.path.exists(self.config.RESUME_CHECKPOINT_PATH):
            print(f"\n⚠️  Warning: Checkpoint file not found at '{self.config.RESUME_CHECKPOINT_PATH}'. Starting from scratch.")
            return None

        print(f"\n>> Resuming training from checkpoint: {self.config.RESUME_CHECKPOINT_PATH}")
        checkpoint = torch.load(self.config.RESUME_CHECKPOINT_PATH, map_location=self.config.DEVICE)
        
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            self.history = checkpoint.get('history', self.history)
            print(f"✅ Model weights, best accuracy ({self.best_val_accuracy:.4f}), and history loaded.")
            return checkpoint
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}. Starting from scratch.")
            return None

    def _create_metrics_collection(self):   
        """Helper function to create a standardized collection of metrics."""
        return MetricCollection({
            'accuracy': MulticlassAccuracy(num_classes=self.config.NUM_CLASSES, average='macro'),
            'precision': MulticlassPrecision(num_classes=self.config.NUM_CLASSES, average='macro'),
            'recall': MulticlassRecall(num_classes=self.config.NUM_CLASSES, average='macro'),
            'f1_score': MulticlassF1Score(num_classes=self.config.NUM_CLASSES, average='macro')
        })

    def _get_optimizer(self, stage_config):
        """Initializes the optimizer with differential learning rates."""
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': stage_config.get('base_lr', stage_config['lr'])},
            {'params': self.model.cbam_stages.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
            {'params': self.model.pwca.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
            {'params': self.model.embedding_head.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
            {'params': self.model.arcface_head.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
        ]
        base_optimizer = optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)
        if 'base_lr' in stage_config:
            print("Using SAM optimizer.")
            return SAM(self.model.parameters(), base_optimizer, rho=self.config.SAM_RHO)
        else:
            print("Using AdamW optimizer.")
            return base_optimizer

    def _adjust_learning_rate(self, stage_config):
        """Dynamically adjusts the learning rate for a new training stage."""
        general_lr = stage_config.get('lr')
        base_lr = stage_config.get('base_lr', general_lr)
        head_lr = stage_config.get('head_lr', general_lr)
        if base_lr is None or head_lr is None:
            raise ValueError("Learning rate configuration is missing.")
        for i, param_group in enumerate(self.optimizer.param_groups):
             param_group['lr'] = base_lr if i == 0 else head_lr
        print(f"Learning rates adjusted: Backbone LR = {base_lr}, Head LR = {head_lr}")

    def _train_one_epoch(self, dataloader, accumulation_steps=1): # Add accumulation_steps as an argument
        """Runs a single training epoch."""
        self.model.train()
        self.train_metrics.reset()
        train_loss = 0.0
        #  Initialize the optimizer gradients
        self.optimizer.zero_grad() 
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for i, (images, labels, distractor_images) in enumerate(progress_bar):
            images, labels, distractor_images = images.to(self.config.DEVICE), labels.to(self.config.DEVICE), distractor_images.to(self.config.DEVICE)

            def forward_pass():
                return self.model(images, labels, distractor_images)

            # --- Loss Calculation ---
            # The forward and loss calculation logic remains the same
            if isinstance(self.optimizer, SAM):
                # SAM requires two forward/backward passes
                outputs = forward_pass()
                loss = self.loss_fn(outputs, labels)
                loss = loss / accumulation_steps # Normalize loss
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                outputs_2 = forward_pass()
                loss_2 = self.loss_fn(outputs_2, labels)
                loss_2 = loss_2 / accumulation_steps # Normalize loss
                loss_2.backward()
                final_outputs, loss_val = outputs_2, loss_2.item() * accumulation_steps # Scale loss back for logging
            else:
                outputs = forward_pass()
                loss = self.loss_fn(outputs, labels)
                loss = loss / accumulation_steps # Normalize loss
                loss.backward()
                final_outputs, loss_val = outputs, loss.item() * accumulation_steps

            # --- Gradient Accumulation Step ---
            if (i + 1) % accumulation_steps == 0:
                if isinstance(self.optimizer, SAM):
                    self.optimizer.second_step(zero_grad=True)
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()

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
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1_score']:.4f}")
        print(f"  Valid -> Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_score']:.4f}")
        print("-" * (len(stage_name) + 20))

    def _plot_and_save_history(self):
        """Plots the training and validation loss and accuracy and saves them as SVG files."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n⚠️  Matplotlib not found. Skipping plot generation. Install with 'pip install matplotlib'.")
            return

        epochs = range(1, len(self.history['train_loss']) + 1)
        # Plot Loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Training Loss', color='blue', marker='.')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss', color='orange', marker='.')
        plt.title('Loss Over Epochs')
        plt.xlabel('Total Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Training Accuracy', color='green', marker='.')
        plt.plot(epochs, self.history['val_acc'], label='Validation Accuracy', color='red', marker='.')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Total Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_history.svg'), format='svg')
        plt.close()

    def run_training_curriculum(self):
        """Executes the full multi-stage training curriculum with resume capability."""
        print("--- Starting Training Curriculum ---")
        checkpoint = self._load_checkpoint()
        start_stage = self.config.RESUME_STAGE if self.config.RESUME and checkpoint else 1
        start_epoch = self.config.RESUME_EPOCH if self.config.RESUME and checkpoint else 0

        stages = [
            {
                "name": "STAGE 1: Head Warm-up", "stage_num": 1,
                "config": {'epochs': self.config.STAGE1_EPOCHS, 'lr': self.config.STAGE1_LR, 'img_size': self.config.STAGE1_IMG_SIZE, 'batch_size': self.config.STAGE1_BATCH_SIZE, 'aug_strength': self.config.STAGE1_AUG_STRENGTH},
                "accumulation_steps": self.config.STAGE1_ACCUMULATION_STEPS, "setup_fn": self.model.freeze_backbone, "use_scheduler": False
            },
            {
                "name": "STAGE 2: Early Full Fine-Tuning", "stage_num": 2,
                "config": {'epochs': self.config.STAGE2_EPOCHS, 'base_lr': self.config.STAGE2_BASE_LR, 'head_lr': self.config.STAGE2_HEAD_LR, 'img_size': self.config.STAGE2_IMG_SIZE, 'batch_size': self.config.STAGE2_BATCH_SIZE, 'aug_strength': self.config.STAGE2_AUG_STRENGTH},
                "accumulation_steps": self.config.STAGE2_ACCUMULATION_STEPS, "setup_fn": self.model.unfreeze_backbone, "use_scheduler": True
            },
            {
                "name": "STAGE 3: Final High-Res Polishing", "stage_num": 3,
                "config": {'epochs': self.config.STAGE3_EPOCHS, 'base_lr': self.config.STAGE3_BASE_LR, 'head_lr': self.config.STAGE3_HEAD_LR, 'img_size': self.config.STAGE3_IMG_SIZE, 'batch_size': self.config.STAGE3_BATCH_SIZE, 'aug_strength': self.config.STAGE3_AUG_STRENGTH},
                "accumulation_steps": self.config.STAGE3_ACCUMULATION_STEPS, "setup_fn": None, "use_scheduler": True
            }
        ]

        for stage_data in stages:
            stage_num, stage_name, stage_config = stage_data["stage_num"], stage_data["name"], stage_data["config"]
            if stage_num < start_stage:
                print(f"\n--- Skipping {stage_name} ---")
                continue
            
            print(f"\n--- {stage_name} ---")
            if stage_data["setup_fn"]: stage_data["setup_fn"]()
            train_loader, val_loader = create_dataloaders(self.config, stage_config)
            
            if self.optimizer is None: self.optimizer = self._get_optimizer(stage_config)
            else: self._adjust_learning_rate(stage_config)

            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=stage_config['epochs']) if stage_data["use_scheduler"] else None

            current_stage_start_epoch = 0
            if stage_num == start_stage and start_epoch > 0:
                if checkpoint and 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✅ Optimizer state loaded.")
                if self.scheduler and checkpoint and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("✅ Scheduler state loaded.")
                current_stage_start_epoch = start_epoch
                print(f">> Starting this stage from epoch {current_stage_start_epoch + 1}")

            for epoch in range(current_stage_start_epoch, stage_config['epochs']):
                train_loss, train_metrics = self._train_one_epoch(train_loader, stage_data["accumulation_steps"])
                val_loss, val_metrics = self._validate_one_epoch(val_loader)
                if self.scheduler: self.scheduler.step()
                self._print_metrics(stage_name, epoch, stage_config['epochs'], train_loss, train_metrics, val_loss, val_metrics)
                self._save_checkpoint(stage_num, epoch, val_metrics)
                self.history['train_loss'].append(train_loss); self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['val_loss'].append(val_loss); self.history['val_acc'].append(val_metrics['accuracy'])

        print("\n--- Training Finished ---")
        print("📊 Generating training history plots...")
        self._plot_and_save_history()
        print(f"✅ Plots saved to '{self.checkpoint_dir}'")
