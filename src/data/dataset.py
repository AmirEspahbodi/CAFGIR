import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
import random
from PIL import Image

def get_augmentations(strength='mild', image_size=224):
    """Returns a transform pipeline based on strength and image size."""
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
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root_dir)
        self.samples = self.dataset.samples
        self.class_to_indices = self._get_class_indices()

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
        distractor_label = random.choice([l for l in range(len(self.dataset.classes)) if l != anchor_label])
        distractor_index = random.choice(self.class_to_indices[distractor_label])
        distractor_path, _ = self.samples[distractor_index]
        distractor_img = Image.open(distractor_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            distractor_img = self.transform(distractor_img)
            
        return anchor_img, anchor_label, distractor_img

def create_dataloaders(config, stage_config):
    """Creates training and validation dataloaders for a given stage."""
    train_transform = get_augmentations(
        strength=stage_config['aug_strength'], 
        image_size=stage_config['img_size']
    )
    val_transform = get_augmentations(
        strength='none', 
        image_size=stage_config['img_size']
    )

    train_dir = os.path.join(config.DATA_DIR, 'train')
    val_dir = os.path.join(config.DATA_DIR, 'validation')
    
    # Use the special distractor dataset for training
    train_dataset = SeatCoverDistractorDataset(root_dir=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

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
        batch_size=stage_config['batch_size'] * 2, # Larger batch for faster validation
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader
