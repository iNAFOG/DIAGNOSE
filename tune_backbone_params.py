import optuna
import torch
import numpy as np
import os
from config import Config
from main import get_datasets, set_seed
from backbone import BackboneModel
from backbone_trainer import BackboneTrainer
from torch.utils.data import DataLoader

def objective(trial):
    # 1. Suggest Hyperparameters
    # alpha: Weight for Class 0. Class 1 weight will be 1 - alpha.
    # Range: 0.01 to 0.99 to avoid zero weights
    alpha = trial.suggest_float('alpha', 0.01, 0.99)
    
    # c: FIS Loss parameter (fusion weight)
    # Range: 0.0 to 1.0
    fis_c = trial.suggest_float('fis_c', 0.0, 1.0)
    
    # 2. Setup Device and Data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Note: We load datasets inside objective or globally? 
    # Better globally to avoid reloading every trial, but for simplicity/safety let's assume they are passed or global.
    # We'll use the global variables defined in main block.
    
    # 3. Prepare Class Weights
    # Normalized: [alpha, 1-alpha]
    weights = [alpha, 1.0 - alpha]
    class_weights = torch.tensor(weights).float().to(device)
    
    # 4. Initialize Model
    num_classes = Config.dataset_config.get('num_classes', 2)
    backbone_name = Config.backbone_model.get('name', 'vit_base_patch16_224')
    
    model = BackboneModel(num_classes=num_classes, backbone=backbone_name).to(device)
    
    # 5. Train Backbone
    # We use a smaller number of epochs for tuning to be faster, or full epochs?
    # User didn't specify, but usually tuning uses fewer epochs or same.
    # Let's use the config's num_epochs but maybe we can override if needed.
    # For now, we stick to config.
    
    trainer = BackboneTrainer()
    
    # We need to capture the result. The train method returns the trained model.
    # But we need the validation metric.
    # BackboneTrainer.train currently prints metrics but doesn't return the best metric.
    # We need to modify BackboneTrainer to return the best metric or we can parse it?
    # Or better, we can subclass or modify BackboneTrainer to return the metric.
    # Actually, looking at BackboneTrainer.train, it returns 'model'.
    # It tracks 'best_worst_case_auc' internally for early stopping.
    # I should modify BackboneTrainer.train to return (model, best_worst_case_auc) or similar.
    # OR, I can just re-evaluate the returned model on the validation set.
    # Since the model returned is the one with best early stopping performance (if enabled),
    # re-evaluating it should give the best result.
    
    trained_model = trainer.train(
        model, 
        train_loader, 
        groups, 
        device=device, 
        val_loader=val_loader, 
        class_weights=class_weights,
        fis_c=fis_c
    )
    
    # 6. Evaluate
    # We re-evaluate to get the exact metric to return
    worst_case_auc = trainer._validate_worst_case(trained_model, val_loader, groups, device=device)
    
    return worst_case_auc

if __name__ == "__main__":
    # Setup
    set_seed(42)
    data_path = Config.data_path
    dataset_name = Config.dataset_config.get('name', 'HAM10000')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Tuning Backbone for Dataset: {dataset_name}")
    print(f"Backbone: {Config.backbone_model.get('name')}")
    
    # Load Data
    try:
        train_dataset, val_dataset, test_dataset = get_datasets(dataset_name, data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
        
    # Prepare Loaders
    batch_size = Config.data_config['batch_size']
    num_workers = Config.data_config['num_workers']
    pin_memory = Config.data_config['pin_memory']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    
    # Get Groups
    groups = train_dataset.get_groups()
    
    # Optuna Study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # Run 20 trials
    
    print("\n" + "="*30)
    print("TUNING COMPLETE")
    print("="*30)
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value (Worst-Case AUC): {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Calculate derived weights
    best_alpha = study.best_params['alpha']
    print(f"Derived Class Weights: [{best_alpha:.4f}, {1-best_alpha:.4f}]")
