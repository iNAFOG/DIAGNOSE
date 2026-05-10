import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from config import Config
from sklearn.model_selection import train_test_split
import glob

try:
    import nibabel as nib
except ImportError:
    nib = None

class OASISDataset(Dataset):
    def __init__(self, dataframe, data_path, sensitive_attr='Sex_binary', transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_path = data_path
        self.sensitive_attr = sensitive_attr
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Define the directory where images are stored
        img_dir = os.path.join(self.data_path, 'images')
        
        # Search for a file that starts with the ID and ends with .nii.
        # Example: /data/images/OASIS_ID_123_processed.nii
        search_pattern = os.path.join(img_dir, f"{row['ID']}*.nii")
        matches = glob.glob(search_pattern)
        
        if not matches:
            raise FileNotFoundError(f"No .nii file found starting with {row['ID']} in {img_dir}")
            
        # Take the first match (assumes one file per ID).
        nii_path = matches[0] 
        
        try:
            if nib is None:
                raise ImportError("nibabel is not installed. Please install it to use OASIS dataset.")
                
            # Load NIfTI
            img = nib.load(nii_path)
            # Get data as float32 to avoid overflow during normalization
            vol = img.get_fdata(dtype=np.float32) 
            
            # Shape is usually (H, W, D) or (H, W, D, 1)
            if len(vol.shape) == 4:
                vol = vol[:, :, :, 0]
                
            depth = vol.shape[2]
            mid_idx = depth // 2
            
            # Get relative slice offset from dataframe
            offset = int(row['slice_offset'])
            
            # Calculate actual slice index
            slice_idx = mid_idx + offset
            
            # Clip to valid range just in case
            slice_idx = max(0, min(slice_idx, depth - 1))
            
            # Extract slice (Axial is usually 3rd dim)
            img_2d = vol[:, :, slice_idx]
            
            # Normalize (Robust min-max: 1st-99th percentile)
            p1 = np.percentile(img_2d, 1)
            p99 = np.percentile(img_2d, 99)
            
            # Clip and scale
            img_2d = np.clip(img_2d, p1, p99)
            if (p99 - p1) > 1e-6:
                img_2d = (img_2d - p1) / (p99 - p1) * 255
            else:
                img_2d = np.zeros_like(img_2d)
                
            img_2d = img_2d.astype(np.uint8)
            
            # Convert to PIL
            image = Image.fromarray(img_2d, mode='L').convert('RGB')

        except Exception as e:
            print(f"Error loading {nii_path}: {e}")
            raise e

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(int(row['binaryLabel']), dtype=torch.long)
        group = torch.tensor(int(row[self.sensitive_attr]), dtype=torch.long)
        
        return {'image': image, 'label': label, 'group': group}

    def get_groups(self):
        groups = {}
        group_column = self.df[self.sensitive_attr].values
        for idx, gid in enumerate(group_column):
            gid = int(gid)
            groups.setdefault(gid, []).append(idx)
        return groups

def get_oasis_transforms(is_training=True):
    img_size = Config.backbone_model.get('img_size', 224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])

def prepare_oasis_data(data_path):
    csv_path = os.path.join(data_path, 'oasis_cross-sectional.csv')
    if not os.path.exists(csv_path):
        # Fallback to local if not in data_path
        csv_path = 'oasis_cross-sectional.csv'
        
    print(f"Loading OASIS metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Binary Label Mapping
    # Category 0: CDR 0, 0.5, 1
    # Category 1: CDR 2
    def map_label(cdr):
        try:
            cdr = float(cdr)
        except:
            return -1 # Invalid
        if cdr in [0.0, 0.5, 1.0]:
            return 0
        elif cdr == 2.0:
            return 1
        else:
            return -1 # Exclude (e.g. CDR 3 if any, or NaN)
            
    df['binaryLabel'] = df['CDR'].apply(map_label)
    df = df[df['binaryLabel'] != -1]
    
    # 2. Sensitive Attributes
    # Sex_binary: M -> 1, F -> 0
    df['Sex_binary'] = df['M/F'].apply(lambda x: 1 if str(x).upper() == 'M' else 0)
    
    print("DEBUG: Sex_binary distribution in original df:")
    print(df['Sex_binary'].value_counts())
    print("DEBUG: Label x Sex distribution in original df:")
    print(pd.crosstab(df['binaryLabel'], df['Sex_binary']))
    
    # Age_binary: > 60 -> 1, else 0
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Age_binary'] = (df['Age'] > 60).astype(int)
    
    # 3. Slice Expansion Logic
    # Category 0: 10% of central slices
    # Category 1: 25% of central slices
    # Assuming standard depth of 176 slices (OASIS-1 T1w)
    # 10% of 176 = 17.6 -> 18 slices (+/- 9)
    # 25% of 176 = 44 slices (+/- 22)
    
    expanded_rows = []
    
    for _, row in df.iterrows():
        label = row['binaryLabel']
        
        if label == 0:
            # 10% -> ~18 slices -> range -9 to +8
            offsets = range(-9, 9)
        else:
            # 25% -> ~44 slices -> range -22 to +22
            offsets = range(-22, 22)
            
        for offset in offsets:
            new_row = row.copy()
            new_row['slice_offset'] = offset
            expanded_rows.append(new_row)
            
    expanded_df = pd.DataFrame(expanded_rows)
    print(f"Expanded dataset from {len(df)} patients to {len(expanded_df)} slices.")
    
    print("DEBUG: Sex_binary distribution in expanded_df:")
    print(expanded_df['Sex_binary'].value_counts())
    print("DEBUG: Label x Sex distribution in expanded_df:")
    print(pd.crosstab(expanded_df['binaryLabel'], expanded_df['Sex_binary']))
    
    # 4. Split
    # We should split by PATIENT, not by slice, to avoid data leakage.
    # We use Stratified Split to ensure representation of (Label, Sex) groups in all sets.
    
    # Create a stratification key: Label_Sex
    # We use the original df (one row per patient)
    df['stratify_key'] = df['binaryLabel'].astype(str) + "_" + df['Sex_binary'].astype(str)
    
    print("Stratification Groups Distribution:")
    print(df['stratify_key'].value_counts())
    
    unique_ids = df['ID'].values
    stratify_labels = df['stratify_key'].values
    
    try:
        # Split Train (80%) and Temp (20%)
        train_ids, temp_ids, _, temp_stratify = train_test_split(
            unique_ids, stratify_labels,
            test_size=0.2, 
            stratify=stratify_labels, 
            random_state=42
        )
        
        # Split Temp into Val (10% total) and Test (10% total) -> 50% of Temp
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=0.5,
            stratify=temp_stratify,
            random_state=42
        )
        print("✅ Successfully performed stratified split on (Label, Sex).")
        
    except ValueError as e:
        print(f"⚠️ Stratified split failed (likely due to small groups): {e}")
        print("Falling back to stratification by Label only...")
        try:
            stratify_labels = df['binaryLabel'].values
            train_ids, temp_ids, _, temp_stratify = train_test_split(
                unique_ids, stratify_labels,
                test_size=0.2, 
                stratify=stratify_labels, 
                random_state=42
            )
            val_ids, test_ids = train_test_split(
                temp_ids,
                test_size=0.5,
                stratify=temp_stratify,
                random_state=42
            )
            print("✅ Successfully performed stratified split on Label only.")
        except ValueError:
             print("⚠️ Stratification failed completely. Falling back to random split.")
             np.random.seed(42)
             shuffled_ids = np.random.permutation(unique_ids)
             n_total = len(shuffled_ids)
             n_train = int(0.8 * n_total)
             n_val = int(0.1 * n_total)
             train_ids = shuffled_ids[:n_train]
             val_ids = shuffled_ids[n_train:n_train+n_val]
             test_ids = shuffled_ids[n_train+n_val:]

    train_ids = set(train_ids)
    val_ids = set(val_ids)
    test_ids = set(test_ids)
    
    train_df = expanded_df[expanded_df['ID'].isin(train_ids)]
    val_df = expanded_df[expanded_df['ID'].isin(val_ids)]
    test_df = expanded_df[expanded_df['ID'].isin(test_ids)]
    
    print(f"Split sizes (slices): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Create Datasets
    sensitive_attr = Config.dataset_config.get('sensitive_attr', 'Sex_binary')
    
    train_transform = get_oasis_transforms(is_training=True)
    val_transform = get_oasis_transforms(is_training=False)
    
    train_dataset = OASISDataset(train_df, data_path, sensitive_attr=sensitive_attr, transform=train_transform)
    val_dataset = OASISDataset(val_df, data_path, sensitive_attr=sensitive_attr, transform=val_transform)
    test_dataset = OASISDataset(test_df, data_path, sensitive_attr=sensitive_attr, transform=val_transform)
    
    return train_dataset, val_dataset, test_dataset
