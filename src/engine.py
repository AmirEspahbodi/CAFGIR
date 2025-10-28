# FILE: engine.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import torch.optim as optim
from src.data.dataset import create_dataloaders
from torchmetrics.retrieval import RetrievalRecall, RetrievalMAP

class Trainer:
    def __init__(self, model, criterion, config, loss_name):
        self.config = config
        self.model = model.to(config.DEVICE)
        self.loss_fn = criterion.to(config.DEVICE)
        self.optimizer = None
        self.scheduler = None
        self.loss_name = loss_name

        self.val_r1 = RetrievalRecall(top_k=1).to(config.DEVICE)
        self.val_r5 = RetrievalRecall(top_k=5).to(config.DEVICE)
        self.val_r10 = RetrievalRecall(top_k=10).to(config.DEVICE)
        self.val_map = RetrievalMAP().to(config.DEVICE)

        self.best_val_accuracy = 0.0 
        self.checkpoint_dir = getattr(config, 'CHECKPOINT_DIR', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"✅ Checkpoints will be saved to '{self.checkpoint_dir}'")

        self.history = {
            'train_loss': [], 
            'val_loss': [], 
            'val_r1': [], 'val_r5': [], 'val_r10': [], 'val_map': []
        }

    def _save_checkpoint(self, stage_num, epoch, val_metrics):
        """Saves the model checkpoint if the current R@1 is the best so far."""
        current_r1 = val_metrics['r1']
        if current_r1 >= self.best_val_accuracy:
            self.best_val_accuracy = current_r1
        checkpoint_filename = f"{self.config.BASE_MODEL}_{self.loss_name}_{stage_num}_epoch{epoch+1}__{self.config.LR}_{self.config.BATCH_SIZE}_R1@{current_r1:.4f}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        
        checkpoint = {
            'stage_num': stage_num,
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss_fn_state_dict': self.loss_fn.state_dict(), 
            'best_val_accuracy': self.best_val_accuracy, # Stores best R@1
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\n✨ New best model found! Recall@1 improved to {current_r1:.4f}. Checkpoint saved to {checkpoint_path}")

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
            self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
            self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0) 
            self.history = checkpoint.get('history', self.history)
            print(f"✅ Model weights, best R@1 ({self.best_val_accuracy:.4f}), and history loaded.")
            return checkpoint
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}. Starting from scratch.")
            return None


    def _get_optimizer(self, stage_config):
        """Initializes the optimizer with differential learning rates."""
        general_lr = stage_config.get('lr')
        base_lr = stage_config.get('base_lr', general_lr)
        head_lr = stage_config.get('head_lr', general_lr)

        if base_lr is None or head_lr is None:
             raise ValueError(
                 "Learning rate not configured. "
                 "Please provide 'lr' or ('base_lr' and 'head_lr') in stage config."
            )

        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': base_lr},
            {'params': self.model.pwca.parameters(), 'lr': head_lr},
            {'params': self.model.embedding_head.parameters(), 'lr': head_lr},
            {'params': self.model.global_pool_m.parameters(), 'lr': head_lr},
            {'params': self.model.global_pool_l.parameters(), 'lr': head_lr},
            {'params': self.loss_fn.parameters(), 'lr': head_lr},
        ]
        
        base_optimizer_class = optim.AdamW
        optimizer_kwargs = {'weight_decay': self.config.WEIGHT_DECAY}

        print(f"Using AdamW optimizer. Backbone LR: {base_lr}, Head/Loss LR: {head_lr}")
        return base_optimizer_class(param_groups, **optimizer_kwargs)

    def _adjust_learning_rate(self, stage_config):
        """Dynamically adjusts the learning rate for a new training stage."""
        general_lr = stage_config.get('lr')
        base_lr = stage_config.get('base_lr', general_lr)
        head_lr = stage_config.get('head_lr', general_lr)
        
        if base_lr is None or head_lr is None:
            raise ValueError("Learning rate configuration is missing.")
        
        optimizer_param_groups = self.optimizer.param_groups

        for i, param_group in enumerate(optimizer_param_groups):
             # This sets the *initial* LR for the new stage.
             # The scheduler will then take over.
             param_group['lr'] = base_lr if i == 0 else head_lr
             
        print(f"Learning rates reset for new stage: Backbone LR = {base_lr}, Head LR = {head_lr}")

    def _train_one_epoch(self, dataloader, accumulation_steps=1):
        self.model.train()
        train_loss = 0.0
        self.optimizer.zero_grad() 
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for i, (images, labels, distractor_images) in enumerate(progress_bar):
            images, labels, distractor_images = images.to(self.config.DEVICE), labels.to(self.config.DEVICE), distractor_images.to(self.config.DEVICE)

            
            outputs = self.model(images, labels, distractor_images)
            loss = self.loss_fn(outputs, labels) 
            
            loss = loss / accumulation_steps
            
            loss.backward()
            loss_val = loss.item() * accumulation_steps 


            if (i + 1) % accumulation_steps == 0:
                self.optimizer.step()
                
                # --- MODIFIED: Step scheduler *after* optimizer step ---
                if self.scheduler:
                    self.scheduler.step()
                # --- END MODIFICATION ---
                
                self.optimizer.zero_grad()

            train_loss += loss_val
            
            # Log current LR from the first param group (backbone)
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=loss_val, lr=f"{current_lr:.1e}")
        
        # Handle last steps if not perfectly divisible by accumulation_steps
        if (len(dataloader) % accumulation_steps) != 0:
            self.optimizer.step()
            # --- MODIFIED: Step scheduler *after* optimizer step ---
            if self.scheduler:
                self.scheduler.step()
            # --- END MODIFICATION ---
            self.optimizer.zero_grad()

        avg_loss = train_loss / len(dataloader)
        return avg_loss, {} 

    def _validate_one_epoch(self, dataloader):
        self.model.eval()
        
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

                outputs = self.model(images, labels, x_distractor=None)
                
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()

                all_embeddings.append(outputs['embedding'].detach())
                all_labels.append(labels.detach())

        avg_loss = val_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        
        if not all_embeddings:
            print("Validation set empty. Skipping retrieval metrics.")
            return avg_loss, {'r1': 0.0, 'r5': 0.0, 'r10': 0.0, 'map': 0.0}

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
        indexes_flat = torch.arange(N, device=device).repeat_interleave(N)
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
        print(f"  Train -> Loss: {train_loss:.4f}")
        print(f"  Valid -> Loss: {val_loss:.4f} | R@1: {val_metrics['r1']:.4f} | R@5: {val_metrics['r5']:.4f} | R@10: {val_metrics['r10']:.4f} | mAP: {val_metrics['map']:.4f}")
        print("-" * (len(stage_name) + 20))

    def _plot_and_save_history(self):
        """Plots the training/validation loss."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n⚠️  Matplotlib not found. Skipping plot generation. Install with 'pip install matplotlib'.")
            return

        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, self.history['train_loss'], label='Training Loss', color='blue', marker='.')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss', color='orange', marker='.')
        plt.title('Loss Over Epochs')
        plt.xlabel('Total Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_history.svg'), format='svg')
        plt.close()

    def run_training_curriculum(self):
        """Executes the full multi-stage training curriculum with resume capability."""
        print("--- Starting Training Curriculum ---")
        checkpoint = self._load_checkpoint()
        start_stage = self.config.RESUME_STAGE if self.config.RESUME and checkpoint else 1
        start_epoch = self.config.RESUME_EPOCH if self.config.RESUME and checkpoint else 0

        # --- MODIFIED: Added sampler and warmup configs ---
        stages = [
            {
                "name": "STAGE 1: Head Warm-up", "stage_num": 1,
                "config": {
                    'epochs': self.config.STAGE1_EPOCHS, 'lr': self.config.STAGE1_LR, 
                    'img_size': self.config.STAGE1_IMG_SIZE, 'batch_size': self.config.STAGE1_BATCH_SIZE,
                },
                "accumulation_steps": self.config.STAGE1_ACCUMULATION_STEPS, "setup_fn": self.model.freeze_backbone, "use_scheduler": False
            },
            {
                "name": "STAGE 2: Early Full Fine-Tuning", "stage_num": 2,
                "config": {
                    'epochs': self.config.STAGE2_EPOCHS, 'base_lr': self.config.STAGE2_BASE_LR, 'head_lr': self.config.STAGE2_HEAD_LR, 
                    'img_size': self.config.STAGE2_IMG_SIZE, 'batch_size': self.config.STAGE2_BATCH_SIZE,
                    'sampler_p': self.config.STAGE2_SAMPLER_P, 'sampler_k': self.config.STAGE2_SAMPLER_K,
                    'warmup_epochs': self.config.STAGE2_WARMUP_EPOCHS
                },
                "accumulation_steps": self.config.STAGE2_ACCUMULATION_STEPS, "setup_fn": self.model.unfreeze_backbone, "use_scheduler": True
            },
            {
                "name": "STAGE 3: Final High-Res Polishing", "stage_num": 3,
                "config": {
                    'epochs': self.config.STAGE3_EPOCHS, 'base_lr': self.config.STAGE3_BASE_LR, 'head_lr': self.config.STAGE3_HEAD_LR, 
                    'img_size': self.config.STAGE3_IMG_SIZE, 'batch_size': self.config.STAGE3_BATCH_SIZE,
                    'sampler_p': self.config.STAGE3_SAMPLER_P, 'sampler_k': self.config.STAGE3_SAMPLER_K,
                    'warmup_epochs': self.config.STAGE3_WARMUP_EPOCHS
                },
                "accumulation_steps": self.config.STAGE3_ACCUMULATION_STEPS, "setup_fn": None, "use_scheduler": True
            }
        ]
        # --- END MODIFICATION ---

        for stage_data in stages:
            stage_num, stage_name, stage_config = stage_data["stage_num"], stage_data["name"], stage_data["config"]
            if stage_num < start_stage:
                print(f"\n--- Skipping {stage_name} ---")
                continue
            
            print(f"\n--- {stage_name} ---")
            if stage_data["setup_fn"]: stage_data["setup_fn"]()
            
            # Create dataloaders FIRST (needed for scheduler calculation)
            train_loader, val_loader = create_dataloaders(self.config, stage_config)
            
            # Setup optimizer
            if self.optimizer is None: self.optimizer = self._get_optimizer(stage_config)
            else: self._adjust_learning_rate(stage_config)

            # --- MODIFIED: New Scheduler Logic (step-based, with warmup by EPOCH) ---
            if stage_data["use_scheduler"]:
                # Total optimizer steps = (num_batches / accum_steps) * num_epochs
                # Note: len(train_loader) IS num_batches (from batch_sampler or dataloader)
                optimizer_steps_per_epoch = len(train_loader) // stage_data["accumulation_steps"]
                total_optimizer_steps = optimizer_steps_per_epoch * stage_config['epochs']
                
                # Get warmup epochs from config
                warmup_epochs = stage_config.get('warmup_epochs', 0)
                warmup_steps = optimizer_steps_per_epoch * warmup_epochs

                if warmup_steps > 0:
                    cosine_steps = total_optimizer_steps - warmup_steps
                    
                    if cosine_steps <= 0:
                         print(f"⚠️  Warmup epochs ({warmup_epochs}) is >= total epochs ({stage_config['epochs']}). Defaulting to CosineAnnealingLR for all steps.")
                         self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer, T_max=total_optimizer_steps, eta_min=1e-7
                         )
                    else:
                        scheduler1 = optim.lr_scheduler.LinearLR(
                            self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
                        )
                        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer, T_max=cosine_steps, eta_min=1e-7
                        )
                        self.scheduler = optim.lr_scheduler.SequentialLR(
                            self.optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps]
                        )
                        print(f"✅ Using SequentialLR: {warmup_epochs} epoch(s) warm-up ({warmup_steps} steps) + {cosine_steps} cosine decay steps.")
                else:
                    print("✅ Using CosineAnnealingLR (no warmup).")
                    self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=total_optimizer_steps, eta_min=1e-7
                    )
            else:
                print("No scheduler used for this stage.")
                self.scheduler = None
            # --- END MODIFICATION ---

            current_stage_start_epoch = 0
            if stage_num == start_stage and start_epoch > 0:
                if checkpoint and 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✅ Optimizer state loaded.")
                if self.scheduler and checkpoint and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                    # This is tricky, as we've redefined the scheduler.
                    # It's safer to restart the scheduler but fast-forward it.
                    print("Scheduler state found, but re-initializing and fast-forwarding.")
                    num_steps_to_skip = optimizer_steps_per_epoch * start_epoch
                    for _ in range(num_steps_to_skip):
                        if self.scheduler: self.scheduler.step()
                    print(f"✅ Scheduler fast-forwarded by {num_steps_to_skip} steps.")

                current_stage_start_epoch = start_epoch
                print(f">> Starting this stage from epoch {current_stage_start_epoch + 1}")

            for epoch in range(current_stage_start_epoch, stage_config['epochs']):
                train_loss, _ = self._train_one_epoch(train_loader, stage_data["accumulation_steps"])
                
                val_loss, val_metrics = self._validate_one_epoch(val_loader)
                
                # --- MODIFIED: Scheduler step is now in _train_one_epoch ---
                
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_r1'].append(val_metrics['r1'])
                self.history['val_r5'].append(val_metrics['r5'])
                self.history['val_r10'].append(val_metrics['r10'])
                self.history['val_map'].append(val_metrics['map'])
                self._print_metrics(stage_name, epoch, stage_config['epochs'], train_loss, None, val_loss, val_metrics)
                self._save_checkpoint(stage_num, epoch, val_metrics)

        print("\n--- Training Finished ---")
        print("📊 Generating training history plots...")
        self.plot_and_save_history_svg() # Use SVG for clarity
        print(f"✅ Plots saved to '{self.checkpoint_dir}'")

    # --- Added SVG plotting function ---
    def plot_and_save_history_svg(self):
        """Plots the training/validation loss and R@1, saving as SVG."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n⚠️  Matplotlib not found. Skipping plot generation. Install with 'pip install matplotlib'.")
            return

        total_epochs = len(self.history['train_loss'])
        if total_epochs == 0:
            print("No history to plot.")
            return
            
        epochs = range(1, total_epochs + 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot Loss
        ax1.plot(epochs, self.history['train_loss'], label='Training Loss', color='blue', marker='.', linestyle='--')
        ax1.plot(epochs, self.history['val_loss'], label='Validation Loss', color='orange', marker='.')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.7)
        ax1.set_title('Training & Validation Loss')

        # Plot Recall@1
        ax2.plot(epochs, self.history['val_r1'], label='R@1', color='green', marker='.')
        ax2.plot(epochs, self.history['val_map'], label='mAP', color='red', marker='.', linestyle=':')
        ax2.set_xlabel('Total Epochs')
        ax2.set_ylabel('Metric')
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.7)
        ax2.set_title('Validation Retrieval Metrics (R@1, mAP)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_history.svg'), format='svg')
        plt.close()

