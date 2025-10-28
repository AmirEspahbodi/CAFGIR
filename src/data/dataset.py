import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, datasets
import os
import random
from PIL import Image
import numpy as np

def get_transforms(image_size=224):
    return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class SeatCoverDistractorDataset(Dataset):
    """
    Custom dataset that provides an anchor image, its label, and a
    distractor image from a different class for PWCA training.
    (Used ONLY for training)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root_dir)
        self.samples = self.dataset.samples
        self.class_to_indices = self._get_class_indices()
        
        self.class_to_idx = self.dataset.class_to_idx
        self.classes = self.dataset.classes

    def _get_class_indices(self):
        class_to_indices = {i: [] for i in range(len(self.dataset.classes))}
        for i, (_, label) in enumerate(self.samples):
            class_to_indices[label].append(i)
        return class_to_indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Get anchor image and label
        anchor_path, anchor_label = self.samples[index]
        anchor_img = Image.open(anchor_path).convert('RGB')

        # Get distractor image from a different class
        distractor_label = random.choice([l for l in range(len(self.classes)) if l != anchor_label])
        distractor_index = random.choice(self.class_to_indices[distractor_label])
        distractor_path, _ = self.samples[distractor_index]
        distractor_img = Image.open(distractor_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            distractor_img = self.transform(distractor_img)
            
        return anchor_img, anchor_label, distractor_img

class ClassAwareBatchSampler(Sampler):
    """
    P-K Sampler Implementation (as a BatchSampler).
    Selects P classes and K images from each class for each batch.
    
    Args:
        dataset: A dataset object that MUST have `class_to_indices` attribute.
        num_classes_per_batch (P): The number of distinct classes in a batch.
        num_images_per_class (K): The number of images per class in a batch.
    """
    def __init__(self, dataset, num_classes_per_batch, num_images_per_class):
        self.dataset = dataset
        self.num_classes_per_batch = num_classes_per_batch
        self.num_images_per_class = num_images_per_class
        self.batch_size = self.num_classes_per_batch * self.num_images_per_class
        
        if not hasattr(dataset, 'class_to_indices'):
            raise ValueError("Dataset must have a 'class_to_indices' attribute.")
            
        self.class_to_indices = dataset.class_to_indices
        self.class_ids = list(self.class_to_indices.keys())
        
        # Calculate number of batches
        self.num_samples = len(self.dataset)
        self.num_batches = self.num_samples // self.batch_size
        
        print(f"✅ Initialized ClassAwareBatchSampler:")
        print(f"  P (Classes per Batch): {self.num_classes_per_batch}")
        print(f"  K (Images per Class):  {self.num_images_per_class}")
        print(f"  Batch Size (P*K):      {self.batch_size}")
        print(f"  Total Samples:         {self.num_samples}")
        print(f"  Batches per Epoch:     {self.num_batches}")

    def __iter__(self):
        all_batches_indices = []
        for _ in range(self.num_batches):
            batch_indices = []
            
            # 1. Randomly select P classes
            selected_classes = np.random.choice(self.class_ids, self.num_classes_per_batch, replace=False)
            
            # 2. For each class, randomly select K images
            for class_id in selected_classes:
                indices_for_class = self.class_to_indices[class_id]
                
                # Use `random.choices` (with replacement) to allow K > num_images
                # This is more robust if some classes have few images
                selected_indices_for_class = random.choices(indices_for_class, k=self.num_images_per_class)
                
                batch_indices.extend(selected_indices_for_class)
            
            # Shuffle indices within the batch
            random.shuffle(batch_indices)
            all_batches_indices.append(batch_indices)
        
        # Shuffle the batches themselves
        random.shuffle(all_batches_indices)
        
        # Yield batches one by one
        for batch in all_batches_indices:
            yield batch

    def __len__(self):
        return self.num_batches

def create_dataloaders(config, stage_config):
    """Creates training and validation dataloaders for a given stage."""
    train_transform = get_transforms(
        image_size=stage_config['img_size']
    )
    val_transform = get_transforms(
        image_size=stage_config['img_size']
    )

    train_dir = os.path.join(config.DATA_DIR, 'train')
    val_dir = os.path.join(config.DATA_DIR, 'validation')

    # Use the distractor dataset for training
    train_dataset = SeatCoverDistractorDataset(root_dir=train_dir, transform=train_transform)

    # Use a standard ImageFolder for validation.
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    
    # --- New: Check for Class-Aware Sampling ---
    sampler_p = stage_config.get('sampler_p')
    sampler_k = stage_config.get('sampler_k')
    stage_batch_size = stage_config.get('batch_size')

    if sampler_p and sampler_k:
        print(f"Using ClassAwareBatchSampler for Training (P={sampler_p}, K={sampler_k})")
        
        # Sanity check
        expected_batch_size = sampler_p * sampler_k
        if stage_batch_size != expected_batch_size:
            raise ValueError(
                f"Configuration error: Stage BATCH_SIZE ({stage_batch_size}) "
                f"does not match P*K ({expected_batch_size})."
            )

        train_batch_sampler = ClassAwareBatchSampler(
            dataset=train_dataset,
            num_classes_per_batch=sampler_p,
            num_images_per_class=sampler_k
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            # We MUST NOT pass batch_size, shuffle, sampler, or drop_last
            # when batch_sampler is provided, as they are mutually exclusive.
            # Passing `batch_size=None` fails on older PyTorch versions.
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
    else:
        print("Using standard random shuffling for Training.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=stage_batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=stage_batch_size, # Can use larger batch for val
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, val_loader

