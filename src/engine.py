import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import torch.optim as optim
from src.optimizer.sam import SAM
from src.data.dataset import create_dataloaders
from src.models.embedding_model import EmbeddingModel
from src.losses.hybrid_loss import HybridLoss
# === CHANGED: Removed all classification metrics ===
from torchmetrics import MetricCollection
import torch.nn.functional as F
from torchmetrics.retrieval import RetrievalRecall
# ----------------------------------------
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

        # === CHANGED: Use RetrievalRecall for training metrics ===
        self.train_metrics = MetricCollection({
            'Recall@1': RetrievalRecall(top_k=1),
            'Recall@5': RetrievalRecall(top_k=5),
            'Recall@10': RetrievalRecall(top_k=10)
        }).to(config.DEVICE)
        
        # === Validation metrics ===
        self.val_metric1 = RetrievalRecall(top_k=1).to(config.DEVICE)
        self.val_metric5 = RetrievalRecall(top_k=5).to(config.DEVICE)
        self.val_metric10 = RetrievalRecall(top_k=10).to(config.DEVICE)
        # --------------------------------------------------

        # === CHANGED: Renamed best metric variables ===
        self.best_val_r1 = 0.0 # Stores best Recall@1
        self.best_val_r5 = 0.0 # Stores best Recall@5
        self.best_val_r10 = 0.0 # Stores best Recall@10
        # ---------------------------------------------
        
        self.checkpoint_dir = getattr(config, 'CHECKPOINT_DIR', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"✅ Checkpoints will be saved to '{self.checkpoint_dir}'")

        # === CHANGED: Updated history dictionary keys ===
        self.history = {
            'train_loss': [], 'train_r1': [], 'train_r5': [], 'train_r10': [],
            'val_loss': [], 'val_r1': [], 'val_r5': [], 'val_r10': []
        }
        # -----------------------------------------------

    def _save_checkpoint(self, stage_num, epoch, val_metrics):
        """Saves the model checkpoint every epoch and tracks the best metrics."""
        
        current_r1 = val_metrics['Recall@1']
        current_r5 = val_metrics['Recall@5']
        current_r10 = val_metrics['Recall@10']
        
        is_best = False
        if current_r1 > self.best_val_r1:
            self.best_val_r1 = current_r1
            is_best = True # R@1 is the primary metric for "best"
            
        if current_r5 > self.best_val_r5:
            self.best_val_r5 = current_r5
            
        if current_r10 > self.best_val_r10:
            self.best_val_r10 = current_r10
        
        # === CHANGED: Save model every epoch, removed 'if' condition ===
        checkpoint_filename = f"model_stage{stage_num}_epoch{epoch+1}_r1{current_r1:.4f}_r5{current_r5:.4f}_r10{current_r10:.4f}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        
        checkpoint = {
            'stage_num': stage_num,
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_r1': self.best_val_r1, # Track best R@1
            'best_val_r5': self.best_val_r5, # Track best R@5
            'best_val_r10': self.best_val_r10, # Track best R@10
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved to {checkpoint_path}")
        if is_best:
            print(f"✨ New best model found! Recall@1 improved to {current_r1:.4f}.")
        # -------------------------------------------------------------
        

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
            # === CHANGED: Load best R@1, R@5, R@10 ===
            self.best_val_r1 = checkpoint.get('best_val_r1', 0.0)
            self.best_val_r5 = checkpoint.get('best_val_r5', 0.0)
            self.best_val_r10 = checkpoint.get('best_val_r10', 0.0)
            # ----------------------------------------
            self.history = checkpoint.get('history', self.history)
            print(f"✅ Model weights, best R@1 ({self.best_val_r1:.4f}), and history loaded.")
            return checkpoint
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}. Starting from scratch.")
            return None

    # === CHANGED: Removed _create_metrics_collection as it's no longer needed ===

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
        # Use SAM optimizer only in stage 2 and 3 (when differential LR is specified)
        if 'base_lr' in stage_config:
            print("Using SAM optimizer.")
            # Note: SAM's base_optimizer expects param_groups, not a constructed optimizer
            base_optimizer = optim.AdamW
            return SAM(param_groups, base_optimizer, rho=self.config.SAM_RHO, lr=stage_config['lr'], weight_decay=self.config.WEIGHT_DECAY)
        else:
            print("Using AdamW optimizer.")
            return base_optimizer # This is the AdamW optimizer created earlier


    def _adjust_learning_rate(self, stage_config):
        """Dynamically adjusts the learning rate for a new training stage."""
        general_lr = stage_config.get('lr')
        base_lr = stage_config.get('base_lr', general_lr)
        head_lr = stage_config.get('head_lr', general_lr)

        if base_lr is None or head_lr is None:
            raise ValueError("Learning rate configuration is missing.")
            
        # Get the optimizer, which might be SAM
        optimizer = self.optimizer.base_optimizer if isinstance(self.optimizer, SAM) else self.optimizer

        for i, param_group in enumerate(optimizer.param_groups):
             param_group['lr'] = base_lr if i == 0 else head_lr
        print(f"Learning rates adjusted: Backbone LR = {base_lr}, Head LR = {head_lr}")


    def _train_one_epoch(self, dataloader, accumulation_steps=1):
        """Runs a single training epoch and computes in-batch retrieval metrics."""
        self.model.train()
        self.train_metrics.reset()
        train_loss = 0.0
        self.optimizer.zero_grad() 
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for i, (images, labels, distractor_images) in enumerate(progress_bar):
            images, labels, distractor_images = images.to(self.config.DEVICE), labels.to(self.config.DEVICE), distractor_images.to(self.config.DEVICE)

            def forward_pass():
                # Pass all required inputs
                return self.model(images, labels, distractor_images)

            if isinstance(self.optimizer, SAM):
                # First forward/backward pass
                outputs = forward_pass()
                loss = self.loss_fn(outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                # Second forward/backward pass
                outputs_2 = forward_pass()
                loss_2 = self.loss_fn(outputs_2, labels)
                loss_2 = loss_2 / accumulation_steps
                loss_2.backward()
                self.optimizer.second_step(zero_grad=False) # zero_grad=False to accumulate
                
                final_outputs, loss_val = outputs_2, loss_2.item() * accumulation_steps
            else:
                outputs = forward_pass()
                loss = self.loss_fn(outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
                final_outputs, loss_val = outputs, loss.item() * accumulation_steps

            if (i + 1) % accumulation_steps == 0:
                if not isinstance(self.optimizer, SAM): # SAM's step is handled above
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # === Calculate in-batch retrieval metrics ===
            emb_batch = final_outputs['embedding'].detach()
            labels_batch = labels.detach()
            
            B = labels_batch.size(0)
            if B <= 1: # Need at least 2 samples to compare
                continue
                
            device = emb_batch.device

            # Normalize embeddings and compute (B, B) similarity matrix
            emb_norm = F.normalize(emb_batch, p=2, dim=1)
            sim_matrix = emb_norm @ emb_norm.t() # (B, B)

            # Build (B, B) target matrix (True if same label and not unseen)
            labels_row = labels_batch.unsqueeze(0) # (1, B)
            labels_col = labels_batch.unsqueeze(1) # (B, 1)
            target_matrix = (labels_col == labels_row) & (labels_col >= 0) & (labels_row >= 0)
            
            # Exclude self-matches
            diag_mask = torch.eye(B, dtype=torch.bool, device=device)
            target_matrix = target_matrix & (~diag_mask)

            # Flatten and create indexes
            preds_flat = sim_matrix.flatten() # (B*B,)
            targets_flat = target_matrix.flatten() # (B*B,)
            indexes = torch.arange(B, device=device).repeat_interleave(B) # (B*B,)
            
            # Update metrics if there are any valid positive pairs in the batch
            if targets_flat.any():
                self.train_metrics.update(preds_flat, targets_flat, indexes=indexes)
            # ----------------------------------------------------

            train_loss += loss_val
            progress_bar.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = train_loss / len(dataloader)
        metrics_output = self.train_metrics.compute()
        # === CHANGED: Ensure all metrics are returned ===
        scalar_metrics = {k: v.item() for k, v in metrics_output.items()}
        scalar_metrics.setdefault('Recall@1', 0.0)
        scalar_metrics.setdefault('Recall@5', 0.0)
        scalar_metrics.setdefault('Recall@10', 0.0)
        # -----------------------------------------------
        return avg_loss, scalar_metrics

    # === FUNCTION REWRITTEN FOR RETRIEVAL ===
    def _validate_one_epoch(self, dataloader):
        """Runs a single validation epoch and computes retrieval metrics."""
        self.model.eval()
        self.val_metric1.reset()
        self.val_metric5.reset()
        self.val_metric10.reset()
        val_loss = 0.0

        all_embeddings = []
        all_labels = []

        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        with torch.no_grad():
            # Validation loader now returns (images, labels)
            for images, labels in progress_bar:
                images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)

                # 1. Get Embeddings
                outputs = self.model(images)         # model(images) returns {'embedding': ...}
                all_embeddings.append(outputs['embedding'])
                all_labels.append(labels)

                # 2. (Optional) Compute loss for "seen" classes only
                seen_mask = labels >= 0
                if seen_mask.any():
                    seen_images = images[seen_mask]
                    seen_labels = labels[seen_mask]

                    # Re-run model on seen data *with* labels to get logits
                    loss_outputs = self.model(seen_images, seen_labels)
                    loss = self.loss_fn(loss_outputs, seen_labels)
                    val_loss += loss.item()

        avg_loss = val_loss / len(dataloader)  # Note: This is loss for SEEN classes only

        # ===== Build all-vs-all similarity and relevance targets for torchmetrics =====
        all_embeddings = torch.cat(all_embeddings, dim=0)   # shape (N, D)
        all_labels = torch.cat(all_labels, dim=0)           # shape (N,)

        N = all_labels.size(0)
        device = self.config.DEVICE

        if N == 0:
            # nothing to compute
            scalar_metrics = {'Recall@1': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0}
            return avg_loss, scalar_metrics

        # Normalize embeddings and compute similarity matrix (cosine via dot of normalized vectors)
        emb_norm = F.normalize(all_embeddings, p=2, dim=1)
        sim_matrix = emb_norm @ emb_norm.t()   # shape (N, N), higher => more similar

        # Build target matrix: True if same label and label != -1 (unseen)
        labels_row = all_labels.unsqueeze(0)   # (1, N)
        labels_col = all_labels.unsqueeze(1)   # (N, 1)
        target_matrix = (labels_col == labels_row) & (labels_col >= 0) & (labels_row >= 0)
        # Optionally exclude self-matches (common practice)
        exclude_self = True
        if exclude_self:
            diag_mask = torch.eye(N, dtype=torch.bool, device=device)
            target_matrix = target_matrix & (~diag_mask)

        # Flatten preds and targets and create indexes mapping queries
        preds_flat = sim_matrix.flatten()                       # float scores
        targets_flat = target_matrix.flatten()                  # boolean relevance
        # indexes: for each query i there are N entries (its comparisons vs all gallery items)
        indexes = torch.arange(N, device=device).repeat_interleave(N)  # shape (N*N,)

        # Update each retrieval metric (they were instantiated with top_k=1,5,10)
        # IMPORTANT: pass `indexes=indexes` (not query_indexes/gallery_indexes)
        self.val_metric1.update(preds_flat, targets_flat, indexes=indexes)
        r1 = self.val_metric1.compute().item()
        self.val_metric5.update(preds_flat, targets_flat, indexes=indexes)
        r5 = self.val_metric5.compute().item()
        self.val_metric10.update(preds_flat, targets_flat, indexes=indexes)
        r10 = self.val_metric10.compute().item()

        scalar_metrics = {'Recall@1': r1, 'Recall@5': r5, 'Recall@10': r10}
        return avg_loss, scalar_metrics

    # === CHANGED: Print R@1, R@5, R@10 for both train and val ===
    def _print_metrics(self, stage_name, epoch, num_epochs, train_loss, train_metrics, val_loss, val_metrics):
        """Prints a formatted summary of the epoch's metrics."""
        print(f"\n--- {stage_name} | Epoch {epoch+1}/{num_epochs} ---")
        print(f"  Train -> Loss: {train_loss:.4f} | R@1, R@5, R@10: {train_metrics['Recall@1']:.4f}, {train_metrics['Recall@5']:.4f}, {train_metrics['Recall@10']:.4f}")
        print(f"  Valid -> Loss: {val_loss:.4f} | R@1, R@5, R@10: {val_metrics['Recall@1']:.4f}, {val_metrics['Recall@5']:.4f}, {val_metrics['Recall@10']:.4f}")
        print("-" * (len(stage_name) + 20))

    # === CHANGED: Update plot labels for Recall@1 ===
    def _plot_and_save_history(self):
        """Plots the training and validation loss/metrics and saves them as SVG files."""
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
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss (Seen Classes)', color='orange', marker='.')
        plt.title('Loss Over Epochs')
        plt.xlabel('Total Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        # Plot Recall@1
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_r1'], label='Training Recall@1', color='green', marker='.')
        plt.plot(epochs, self.history['val_r1'], label='Validation Recall@1', color='red', marker='.')
        plt.title('Train vs. Validation Recall@1')
        plt.xlabel('Total Epochs'); plt.ylabel('Recall@1'); plt.legend(); plt.grid(True)

        # Plot Recall@5
        plt.subplot(1, 2, 3)
        plt.plot(epochs, self.history['train_r5'], label='Training Recall@5', color='green', marker='.')
        plt.plot(epochs, self.history['val_r5'], label='Validation Recall@5', color='red', marker='.')
        plt.title('Train vs. Validation Recall@5')
        plt.xlabel('Total Epochs'); plt.ylabel('Recall@5'); plt.legend(); plt.grid(True)
        
        # Plot Recall@10
        plt.subplot(1, 2, 4)
        plt.plot(epochs, self.history['train_r10'], label='Training Recall@10', color='green', marker='.')
        plt.plot(epochs, self.history['val_r10'], label='Validation Recall@10', color='red', marker='.')
        plt.title('Train vs. Validation Recall@10')
        plt.xlabel('Total Epochs'); plt.ylabel('Recall@10'); plt.legend(); plt.grid(True)
        
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
                "config": {'epochs': self.config.STAGE1_EPOCHS, 'lr': self.config.STAGE1_LR, 'img_size': self.config.STAGE1_IMG_SIZE, 'batch_size': self.config.STAGE1_BATCH_SIZE, 'aug_strength': 'mild'}, # Use 'mild'
                "accumulation_steps": self.config.STAGE1_ACCUMULATION_STEPS, "setup_fn": self.model.freeze_backbone, "use_scheduler": False
            },
            {
                "name": "STAGE 2: Early Full Fine-Tuning", "stage_num": 2,
                "config": {'epochs': self.config.STAGE2_EPOCHS, 'base_lr': self.config.STAGE2_BASE_LR, 'head_lr': self.config.STAGE2_HEAD_LR, 'img_size': self.config.STAGE2_IMG_SIZE, 'batch_size': self.config.STAGE2_BATCH_SIZE, 'aug_strength': 'moderate'}, # Use 'moderate'
                "accumulation_steps": self.config.STAGE2_ACCUMULATION_STEPS, "setup_fn": self.model.unfreeze_backbone, "use_scheduler": True
            },
            {
                "name": "STAGE 3: Final High-Res Polishing", "stage_num": 3,
                "config": {'epochs': self.config.STAGE3_EPOCHS, 'base_lr': self.config.STAGE3_BASE_LR, 'head_lr': self.config.STAGE3_HEAD_LR, 'img_size': self.config.STAGE3_IMG_SIZE, 'batch_size': self.config.STAGE3_BATCH_SIZE, 'aug_strength': 'aggressive'}, # Use 'aggressive'
                "accumulation_steps": self.config.STAGE3_ACCUMULATION_STEPS, "setup_fn": None, "use_scheduler": True
            }
        ]
        
        # Update stage configs to use string-based aug_strength
        for stage_data in stages:
            if stage_data['stage_num'] == 1: stage_data['config']['aug_strength'] = 'mild' # From STAGE1_AUG_STRENGTH = 0.0
            elif stage_data['stage_num'] == 2: stage_data['config']['aug_strength'] = 'moderate' # From STAGE2_AUG_STRENGTH = 0.2
            elif stage_data['stage_num'] == 3: stage_data['config']['aug_strength'] = 'aggressive' # From STAGE3_AUG_STRENGTH = 1.0


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
                
                # === CHANGED: Save checkpoint every epoch ===
                self._save_checkpoint(stage_num, epoch, val_metrics)
                
                # === CHANGED: Store all R@k metrics in history ===
                self.history['train_loss'].append(train_loss)
                self.history['train_r1'].append(train_metrics['Recall@1'])
                self.history['train_r5'].append(train_metrics['Recall@5'])
                self.history['train_r10'].append(train_metrics['Recall@10'])
                
                self.history['val_loss'].append(val_loss)
                self.history['val_r1'].append(val_metrics['Recall@1'])
                self.history['val_r5'].append(val_metrics['Recall@5'])
                self.history['val_r10'].append(val_metrics['Recall@10'])
                # --------------------------------------------------

        print("\n--- Training Finished ---")
        print("📊 Generating training history plots...")
        self._plot_and_save_history()
        print(f"✅ Plots saved to '{self.checkpoint_dir}'")