import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import torch.optim as optim
from src.optimizer.sam import SAM
from src.data.dataset import create_dataloaders
from src.models.embedding_model import EmbeddingModel
from src.losses.hybrid_loss import HybridLoss

# --- MODIFIED IMPORTS ---
# We still use these for the TRAINING loop proxy task
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchmetrics import MetricCollection
# NEW: Import the correct retrieval metrics for VALIDATION
from torchmetrics.retrieval import RetrievalRecall, RetrievalMAP
# --- END MODIFIED IMPORTS ---

# Matplotlib is imported inside the plotting function
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
        self.loss_fn = criterion.to(config.DEVICE)
        self.optimizer = None
        self.scheduler = None

        # --- METRICS MODIFIED ---
        # 1. Train metrics: Still use classification as a proxy task monitor
        self.train_metrics = self._create_metrics_collection().to(config.DEVICE)
        
        # 2. Validation metrics: Use proper retrieval metrics
        # We need separate instances for each K in Recall@K
        self.val_r1 = RetrievalRecall(top_k=1).to(config.DEVICE)
        self.val_r5 = RetrievalRecall(top_k=5).to(config.DEVICE)
        self.val_r10 = RetrievalRecall(top_k=10).to(config.DEVICE)
        self.val_map = RetrievalMAP().to(config.DEVICE)
        # --- END METRICS MODIFIED ---

        self.best_val_accuracy = 0.0 # This will now store the best R@1
        self.checkpoint_dir = getattr(config, 'CHECKPOINT_DIR', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"✅ Checkpoints will be saved to '{self.checkpoint_dir}'")

        # --- HISTORY MODIFIED ---
        self.history = {
            'train_loss': [], 'train_acc': [], # Proxy metrics
            'val_loss': [], 
            'val_r1': [], 'val_r5': [], 'val_r10': [], 'val_map': []
        }
        # --- END HISTORY MODIFIED ---

    def _save_checkpoint(self, stage_num, epoch, val_metrics):
        """Saves the model checkpoint if the current R@1 is the best so far."""
        # --- MODIFIED TO USE R@1 ---
        current_r1 = val_metrics['r1']
        if current_r1 >= self.best_val_accuracy:
            self.best_val_accuracy = current_r1
            # Update filename to reflect R@1
        checkpoint_filename = f"{self.config.BASE_MODEL}_best_model_stage{stage_num}_epoch{epoch+1}_R1@{current_r1:.4f}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        
        checkpoint = {
            'stage_num': stage_num,
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_accuracy': self.best_val_accuracy, # Stores best R@1
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\n✨ New best model found! Recall@1 improved to {current_r1:.4f}. Checkpoint saved to {checkpoint_path}")
        # --- END MODIFIED ---

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
            # best_val_accuracy will hold the best R@1 from the checkpoint
            self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0) 
            self.history = checkpoint.get('history', self.history)
            print(f"✅ Model weights, best R@1 ({self.best_val_accuracy:.4f}), and history loaded.")
            return checkpoint
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}. Starting from scratch.")
            return None

    def _create_metrics_collection(self):   
        """Helper function to create a standardized collection of metrics (for training proxy task)."""
        return MetricCollection({
            'accuracy': MulticlassAccuracy(num_classes=self.config.NUM_CLASSES, average='macro'),
            'precision': MulticlassPrecision(num_classes=self.config.NUM_CLASSES, average='macro'),
            'recall': MulticlassRecall(num_classes=self.config.NUM_CLASSES, average='macro'),
            'f1_score': MulticlassF1Score(num_classes=self.config.NUM_CLASSES, average='macro')
        })

    # --- FUNCTION WITH FIX ---
    def _get_optimizer(self, stage_config):
        """Initializes the optimizer with differential learning rates."""
        
        # (1) Define the parameter groups with differential LRs
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': stage_config.get('base_lr', stage_config['lr'])},
            {'params': self.model.cbam_stages.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
            {'params': self.model.pwca.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
            {'params': self.model.embedding_head.parameters(), 'lr': stage_config.get('head_lr', stage_config['lr'])},
        ]
        
        # (2) Define the base optimizer CLASS, not an instance
        base_optimizer_class = optim.AdamW
        
        # (3) Collect kwargs for the base optimizer
        optimizer_kwargs = {'weight_decay': self.config.WEIGHT_DECAY}

        if 'base_lr' in stage_config: # True for Stages 2 & 3
            print("Using SAM optimizer.")
            
            # (4) Pass the DLR groups, the optimizer CLASS, and the kwargs to SAM
            return SAM(
                param_groups,                 # <--- FIX 1: Pass DLR groups as `params`
                base_optimizer_class,         # <--- FIX 2: Pass AdamW class as `base_optimizer`
                rho=self.config.SAM_RHO,
                **optimizer_kwargs            # <--- FIX 3: Pass kwargs to SAM
            )
        else: # Stage 1
            print("Using AdamW optimizer.")
            
            # (5) Create the AdamW instance normally for stage 1
            return base_optimizer_class(param_groups, **optimizer_kwargs)
    # --- END FUNCTION WITH FIX ---

    def _adjust_learning_rate(self, stage_config):
        """Dynamically adjusts the learning rate for a new training stage."""
        general_lr = stage_config.get('lr')
        base_lr = stage_config.get('base_lr', general_lr)
        head_lr = stage_config.get('head_lr', general_lr)
        if base_lr is None or head_lr is None:
            raise ValueError("Learning rate configuration is missing.")
        
        # For SAM, the base_optimizer.param_groups holds the DLRs
        optimizer_param_groups = self.optimizer.base_optimizer.param_groups \
            if isinstance(self.optimizer, SAM) \
            else self.optimizer.param_groups

        for i, param_group in enumerate(optimizer_param_groups):
             param_group['lr'] = base_lr if i == 0 else head_lr
        print(f"Learning rates adjusted: Backbone LR = {base_lr}, Head LR = {head_lr}")

    def _train_one_epoch(self, dataloader, accumulation_steps=1):
        """
        Runs a single training epoch.
        We still use classification metrics here as a FAST, BATCH-WISE PROXY
        to monitor if the ArcFace head is learning to separate classes.
        """
        self.model.train()
        self.train_metrics.reset()
        train_loss = 0.0
        self.optimizer.zero_grad() 
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for i, (images, labels, distractor_images) in enumerate(progress_bar):
            images, labels, distractor_images = images.to(self.config.DEVICE), labels.to(self.config.DEVICE), distractor_images.to(self.config.DEVICE)

            def forward_pass():
                return self.model(images, labels, distractor_images)

            if isinstance(self.optimizer, SAM):
                outputs = forward_pass()
                loss = self.loss_fn(outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                outputs_2 = forward_pass()
                loss_2 = self.loss_fn(outputs_2, labels)
                loss_2 = loss_2 / accumulation_steps
                loss_2.backward()
                final_outputs, loss_val = outputs_2, loss_2.item() * accumulation_steps
            else:
                outputs = forward_pass()
                loss = self.loss_fn(outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
                final_outputs, loss_val = outputs, loss.item() * accumulation_steps

            if (i + 1) % accumulation_steps == 0:
                if isinstance(self.optimizer, SAM):
                    self.optimizer.second_step(zero_grad=True)
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Update proxy classification metrics
            preds = final_outputs['logits'].detach()
            self.train_metrics.update(preds, labels)
            train_loss += loss_val
            progress_bar.set_postfix(loss=loss_val)

        avg_loss = train_loss / len(dataloader)
        metrics_output = self.train_metrics.compute()
        scalar_metrics = {k: v.item() for k, v in metrics_output.items()}
        return avg_loss, scalar_metrics

    def _validate_one_epoch(self, dataloader):
        """
        Runs a single validation epoch, evaluating based on
        retrieval metrics (R@k, mAP).
        """
        self.model.eval()
        
        # Reset all retrieval metrics
        self.val_r1.reset()
        self.val_r5.reset()
        self.val_r10.reset()
        self.val_map.reset()
        
        val_loss = 0.0
        all_embeddings = []
        all_labels = []

        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = (
                    images.to(self.config.DEVICE),
                    labels.to(self.config.DEVICE)
                )

                # Get model output
                outputs = self.model(images, labels, x_distractor=None)
                
                # We can still compute loss for monitoring
                if outputs['logits'] is not None:
                    try:
                        loss = self.loss_fn(outputs, labels)
                        val_loss += loss.item()
                    except Exception as e:
                        # This might fail if val labels are not in train set
                        # which is fine, we just care about embeddings
                        pass 

                # Store embeddings and labels for metric calculation
                all_embeddings.append(outputs['embedding'].detach())
                all_labels.append(labels.detach())

        avg_loss = val_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        
        if not all_embeddings:
            print("Validation set empty. Skipping retrieval metrics.")
            return avg_loss, {'r1': 0.0, 'r5': 0.0, 'r10': 0.0, 'map': 0.0}

        # --- Retrieval Metric Calculation ---
        # 1. Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        N = all_labels.shape[0]
        device = all_labels.device
        
        # 2. Normalize embeddings for cosine similarity
        all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
        
        # 3. Compute pairwise cosine similarity matrix (N, D) x (D, N) -> (N, N)
        sim_matrix = torch.matmul(all_embeddings_norm, all_embeddings_norm.T)
        
        # 4. Create the boolean target matrix (N, N)
        target_matrix = all_labels.unsqueeze(1) == all_labels.unsqueeze(0)
        
        # 5. Mask out diagonal (self-retrieval)
        sim_matrix.fill_diagonal_(-float('inf'))
        target_matrix.fill_diagonal_(False)
        
        # 6. Flatten matrices for per-query evaluation
        # For retrieval metrics, we need to process each query separately
        # Create indexes that identify which query each prediction belongs to
        
        # Create a flattened index tensor: [0,0,0,...,1,1,1,...,2,2,2,...]
        # where each query i appears N times (once for each potential match)
        indexes_flat = torch.arange(N, device=device).repeat_interleave(N)
        
        # Flatten similarity scores and targets
        preds_flat = sim_matrix.flatten()
        target_flat = target_matrix.flatten()
        
        # 7. Update all metrics with flattened tensors
        self.val_r1.update(preds_flat, target_flat, indexes=indexes_flat)
        self.val_r5.update(preds_flat, target_flat, indexes=indexes_flat)
        self.val_r10.update(preds_flat, target_flat, indexes=indexes_flat)
        self.val_map.update(preds_flat, target_flat, indexes=indexes_flat)
        
        # 8. Compute final scalar values
        scalar_metrics = {
            'r1': self.val_r1.compute().item(),
            'r5': self.val_r5.compute().item(),
            'r10': self.val_r10.compute().item(),
            'map': self.val_map.compute().item(),
        }
        return avg_loss, scalar_metrics
    
    def _print_metrics(self, stage_name, epoch, num_epochs, train_loss, train_metrics, val_loss, val_metrics):
        """Prints a formatted summary of the epoch's metrics."""
        print(f"\n--- {stage_name} | Epoch {epoch+1}/{num_epochs} ---")
        # Train metrics are still proxy classification metrics
        print(f"  Train -> Loss: {train_loss:.4f} | Acc (proxy): {train_metrics['accuracy']:.4f} | F1 (proxy): {train_metrics['f1_score']:.4f}")
        # Valid metrics are now proper retrieval metrics
        print(f"  Valid -> Loss: {val_loss:.4f} | R@1: {val_metrics['r1']:.4f} | R@5: {val_metrics['r5']:.4f} | R@10: {val_metrics['r10']:.4f} | mAP: {val_metrics['map']:.4f}")
        print("-" * (len(stage_name) + 20))

    def _plot_and_save_history(self):
        """Plots the training/validation loss and train_acc/validation_R1."""
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
        plt.plot(epochs, self.history['train_acc'], label='Training Accuracy (Proxy)', color='green', marker='.')
        # Plot R@1 instead of 'val_acc'
        plt.plot(epochs, self.history['val_r1'], label='Validation Recall@1', color='red', marker='.')
        plt.title('Accuracy (Proxy) / Recall@1 Over Epochs')
        plt.xlabel('Total Epochs'); plt.ylabel('Accuracy / R@1'); plt.legend(); plt.grid(True)
        
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
                
                # --- MODIFIED HISTORY APPENDING ---
                # Append proxy train metrics
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_metrics['accuracy'])
                
                # Append validation retrieval metrics
                self.history['val_loss'].append(val_loss)
                self.history['val_r1'].append(val_metrics['r1'])
                self.history['val_r5'].append(val_metrics['r5'])
                self.history['val_r10'].append(val_metrics['r10'])
                self.history['val_map'].append(val_metrics['map'])
                # --- END MODIFIED ---
                
                # Print and save
                self._print_metrics(stage_name, epoch, stage_config['epochs'], train_loss, train_metrics, val_loss, val_metrics)
                self._save_checkpoint(stage_num, epoch, val_metrics) # This now uses R@1

        print("\n--- Training Finished ---")
        print("📊 Generating training history plots...")
        self._plot_and_save_history() # This now plots R@1
        print(f"✅ Plots saved to '{self.checkpoint_dir}'")
