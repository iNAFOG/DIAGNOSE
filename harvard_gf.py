import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from config import Config

class HarvardGFDataset(Dataset):
    def __init__(self, dataframe, data_path, sensitive_attr='Race_binary', transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_path = data_path
        self.sensitive_attr = sensitive_attr
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        split_folder = row['use'] # 'training', 'validation', 'test'
        
        # Map to capitalized folder names: Training, Validation, Test
        folder_map = {
            'training': 'Training',
            'validation': 'Validation',
            'test': 'Test'
        }
        # Use the map if the key exists (case-insensitive lookup), otherwise keep original
        split_folder = folder_map.get(str(split_folder).lower(), split_folder)
        
        npz_path = os.path.join(self.data_path, 'images', split_folder, filename)
        
        # Debugging: Check if file exists, if not try to find it
        if not os.path.exists(npz_path):
            # Try case-insensitive search for folder
            parent_dir = os.path.join(self.data_path, 'images')
            if os.path.exists(parent_dir):
                # Check if 'training' vs 'Training' issue
                for folder in os.listdir(parent_dir):
                    if folder.lower() == split_folder.lower():
                        # Found matching folder (ignoring case)
                        correct_folder = folder
                        # Check if file exists in this folder
                        potential_path = os.path.join(parent_dir, correct_folder, filename)
                        if os.path.exists(potential_path):
                            npz_path = potential_path
                            break
            
            # If still not found, try checking if filename needs extension or has extra extension
            if not os.path.exists(npz_path):
                 # Try adding .npz if missing
                 if not filename.endswith('.npz') and os.path.exists(npz_path + '.npz'):
                     npz_path += '.npz'
        
        try:
            # Load NPZ
            # Keys: ['oct_bscans', 'rnflt', 'md', 'glaucoma', 'tds', 'race', 'male', 'hispanic', 'language', 'maritalstatus', 'age']
            with np.load(npz_path) as data:
                # Use RNFLT map (200x200) instead of 3D OCT volume
                img_2d = data['rnflt']
                
                # Handle NaN values if any (RNFLT maps might have NaNs for invalid regions)
                if np.isnan(img_2d).any():
                    img_2d = np.nan_to_num(img_2d, nan=0.0)

                # Normalize to 0-255 uint8 using Global Scaling
                # RNFLT is thickness in microns. Glaucoma is defined by thinning.
                # We use a fixed max of 350 microns to preserve absolute thickness information.
                if img_2d.dtype != np.uint8:
                    img_2d = np.clip(img_2d, 0, 350) # Clip to max expected thickness
                    img_2d = (img_2d / 350.0) * 255
                    img_2d = img_2d.round().astype(np.uint8)
                
                # Convert to PIL Image (Grayscale)
                image = Image.fromarray(img_2d, mode='L').convert('RGB')

        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            # Print directory listing to help debug
            dir_to_check = os.path.dirname(npz_path)
            if os.path.exists(dir_to_check):
                print(f"Contents of {dir_to_check} (first 5): {os.listdir(dir_to_check)[:5]}")
            else:
                print(f"Directory {dir_to_check} does not exist!")
            raise e

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(int(row['binaryLabel']), dtype=torch.long)
        group = torch.tensor(int(row[self.sensitive_attr]), dtype=torch.long)
        
        return {'image': image, 'label': label, 'group': group}

    def get_groups(self):
        """
        Extract groups for FairDI training.
        Returns a dictionary {group_id: [indices]}
        """
        groups = {}
        # Use the dataframe for fast group extraction
        group_column = self.df[self.sensitive_attr].values
        for idx, gid in enumerate(group_column):
            gid = int(gid)
            groups.setdefault(gid, []).append(idx)
            
        print(f'Found {len(groups)} demographic groups')
        print(f'Total samples: {len(self)}')
        for gid, indices in groups.items():
            print(f'Group {gid}: {len(indices)} samples')
        return groups
    
def get_harvard_transforms(is_training=True):
    """
    Get transforms for HarvardGF3300 dataset.
    """
    img_size = Config.backbone_model.get('img_size', 224)
    
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])

def prepare_harvard_data(data_path):
    """
    Load and preprocess HarvardGF3300 metadata.
    """
    csv_path = os.path.join(data_path, 'HarvardGF_metadata.csv')
    if not os.path.exists(csv_path):
        csv_path = 'HarvardGF_metadata.csv'
        
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Binary Label (Glaucoma)
    # 'yes' -> 1, 'no' -> 0
    df['binaryLabel'] = df['glaucoma'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    
    # 2. Sensitive Attributes
    
    # Sex_binary: male -> 1, female -> 0
    df['Sex_binary'] = df['gender'].apply(lambda x: 1 if str(x).lower() == 'male' else 0)
    
    # Age_binary: > 60 -> 1, else 0
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['Age_binary'] = (df['age'] > 60).astype(int)
    
    # Race_binary
    # Group 0: 'black'
    # Group 1: 'asian', 'white'
    def map_race(r):
        r = str(r).lower()
        if r == 'black':
            return 0
        elif r in ['asian', 'white']:
            return 1
        else:
            return -1 # Exclude others
            
    df['Race_binary'] = df['race'].apply(map_race)
    
    # Filter unknown races
    print(f"Original samples: {len(df)}")
    df = df[df['Race_binary'] != -1]
    print(f"Samples after filtering race: {len(df)}")
    
    # 3. Split based on 'use' column
    train_df = df[df['use'] == 'training']
    val_df = df[df['use'] == 'validation']
    test_df = df[df['use'] == 'test']
    
    print(f"Split sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")
    
    # Create Datasets
    sensitive_attr = Config.dataset_config.get('sensitive_attr', 'Race_binary')
    
    train_transform = get_harvard_transforms(is_training=True)
    val_transform = get_harvard_transforms(is_training=False)
    
    train_dataset = HarvardGFDataset(train_df, data_path, sensitive_attr=sensitive_attr, transform=train_transform)
    val_dataset = HarvardGFDataset(val_df, data_path, sensitive_attr=sensitive_attr, transform=val_transform)
    test_dataset = HarvardGFDataset(test_df, data_path, sensitive_attr=sensitive_attr, transform=val_transform)
    
    return train_dataset, val_dataset, test_dataset
