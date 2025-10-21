import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
import random
from PIL import Image
import json # Added for label map

def get_augmentations(strength='mild', image_size=224):
    """Returns a transform pipeline based on strength and image size."""
    
    # === CHANGED: Augmentations are now un-commented ===
    if strength == 'mild':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif strength == 'moderate':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # transforms.RandomHorizontalFlip(),
            # transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif strength == 'aggressive':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandAugment(),
            # transforms.RandomErasing(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else: # For validation/test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class SeatCoverDistractorDataset(Dataset):
    """
    Custom dataset that provides an anchor image, its label, and a
    distractor image from a different class for PWCA training.
    
    === CHANGED: This dataset now requires a class_to_idx map ===
    """
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        if class_to_idx is None:
            raise ValueError("A class_to_idx map must be provided.")
            
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.classes = list(class_to_idx.keys())
        self.num_classes = len(self.classes)
        
        self.samples = self._make_dataset()
        self.class_to_indices = self._get_class_indices()

    def _make_dataset(self):
        samples = []
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir) or class_name not in self.class_to_idx:
                continue
            
            label = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    samples.append((img_path, label))
        return samples

    def _get_class_indices(self):
        class_to_indices = {i: [] for i in range(self.num_classes)}
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
        distractor_label = random.choice([l for l in range(self.num_classes) if l != anchor_label])
        distractor_index = random.choice(self.class_to_indices[distractor_label])
        distractor_path, _ = self.samples[distractor_index]
        distractor_img = Image.open(distractor_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            distractor_img = self.transform(distractor_img)
            
        return anchor_img, anchor_label, distractor_img


# === NEW: Simple dataset for validation/retrieval ===
class RetrievalDataset(Dataset):
    """
    Loads images and labels for retrieval evaluation.
    Uses a provided class_to_idx map and assigns -1 to unseen classes.
    """
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        if class_to_idx is None:
            raise ValueError("A class_to_idx map must be provided.")
            
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = self._make_dataset(root_dir)

    def _make_dataset(self, root_dir):
        samples = []
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # Use the map to get the label.
            # Assign -1 if the class is not in the map (i.e., it's unseen)
            label = self.class_to_idx.get(class_name, -1) 
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    samples.append((img_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(config, stage_config):
    """Creates training and validation dataloaders for a given stage."""
    
    # --- CHANGED: Load or create the standardized class_to_idx map ---
    map_path = os.path.join(config.OUTPUT_DIR, 'class_to_idx.json')
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    train_dir = os.path.join(config.DATA_DIR, 'train')
    val_dir = os.path.join(config.DATA_DIR, 'validation')

    if os.path.exists(map_path):
        print(f"Loading existing class map from {map_path}")
        with open(map_path, 'r') as f:
            class_to_idx = json.load(f)
    else:
        print("Creating new class map...")
        temp_train_dataset = datasets.ImageFolder(root=train_dir)
        class_to_idx = temp_train_dataset.class_to_idx
        with open(map_path, 'w') as f:
            json.dump(class_to_idx, f)
        print(f"Saved class map to {map_path}")
    # -----------------------------------------------------------------

    train_transform = get_augmentations(
        strength=stage_config['aug_strength'],
        image_size=stage_config['img_size']
    )
    val_transform = get_augmentations(
        strength='none',
        image_size=stage_config['img_size']
    )

    # === CHANGED: Pass class_to_idx to both datasets ===
    train_dataset = SeatCoverDistractorDataset(
        root_dir=train_dir, 
        transform=train_transform,
        class_to_idx=class_to_idx
    )
    
    val_dataset = RetrievalDataset(
        root_dir=val_dir, 
        transform=val_transform,
        class_to_idx=class_to_idx
    )
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=stage_config['batch_size'],
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=stage_config['batch_size'] * 2, # Can use larger batch for validation
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, val_loader