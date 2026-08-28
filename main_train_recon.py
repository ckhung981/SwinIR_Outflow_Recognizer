import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import time
from datetime import datetime
import shutil
import random
import gc
import re
import torchvision.transforms as T

# Ensure this imports the recognition version of SwinIR
from models.network_swinir_recong import SwinIR 

def clear_gpu_memory():
    """
    Forces garbage collection and clears CUDA cache, 
    then prints the current VRAM usage to verify it is clean.
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print("=" * 40)
        print("GPU Memory Status After Clearing:")
        print(f"Allocated: {allocated_memory:.2f} MB")
        print(f"Reserved:  {reserved_memory:.2f} MB")
        print("=" * 40)
        
        if allocated_memory > 100:
            print("[WARNING] GPU memory is not completely clean. Check for zombie processes.")

# ==========================================
# 0. Set Random Seed for Reproducibility
# ==========================================
def set_seed(seed):
    """Fix all random seeds to ensure reproducibility when a specific seed is used."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # Force CuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

# ==========================================
# 1. Dataset 
# ==========================================
class ScientificRecognitionDataset(Dataset):
    def __init__(self, root_dir, in_chans=1, transform=None):
        self.root_dir = root_dir
        self.in_chans = in_chans
        self.transform = transform
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            img = np.zeros((8, 8), dtype=np.float32)
            
        img = img.astype(np.float32) / 255.0 
        
        if self.in_chans == 1:
            img = torch.from_numpy(img).unsqueeze(0)
        else:
            img = torch.from_numpy(img).transpose(2, 0, 1)
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
def save_augmented_samples(dataloader, save_dir, num_samples=10):
    """
    Extracts a batch from the dataloader and saves the augmented images to disk
    so the user can visually verify the data augmentation results.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Outputting {num_samples} augmented sample images to {save_dir} ...")
    
    # Grab one batch of data
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    
    # Determine how many to save
    save_count = min(num_samples, inputs.size(0))
    
    for i in range(save_count):
        img_tensor = inputs[i]
        
        # Move back to CPU and convert to numpy
        img_np = img_tensor.cpu().numpy()
        
        # Handle grayscale vs multi-channel
        if img_np.shape[0] == 1:
            img_np = img_np.squeeze(0)  # (1, H, W) -> (H, W)
        else:
            img_np = np.transpose(img_np, (1, 2, 0)) # (C, H, W) -> (H, W, C)
            
        # Un-normalize from [0.0, 1.0] back to [0, 255]
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        
        # Save to file
        file_path = os.path.join(save_dir, f"aug_sample_{i+1:02d}_label_{labels[i].item()}.png")
        cv2.imwrite(file_path, img_np)
        
    print("Augmented sample images output complete")


# ==========================================
# 2. Validation
# ==========================================
def validate(model, test_loader, device, criterion):
    """
    Validates the model and computes Top-1, Top-2, Top-3 accuracies,
    as well as Grouped Accuracies (4 main groups, 4 classes each).
    """
    model.eval()
    val_loss = 0.0
    
    correct_top1, correct_top2, correct_top3 = 0, 0, 0
    grp_correct_top1, grp_correct_top2, grp_correct_top3 = 0, 0, 0
    grp2_correct_top1, grp2_correct_top2, grp2_correct_top3 = 0, 0, 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            total += labels.size(0)
            
            # Get Top-3 predictions
            _, top3_preds = outputs.topk(3, 1, True, True)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                preds = top3_preds[i].tolist()
                
                # Standard Accuracies
                if label == preds[0]:
                    correct_top1 += 1
                if label in preds[:2]:
                    correct_top2 += 1
                if label in preds[:3]:
                    correct_top3 += 1
                    
                # Group Accuracies 
                # Dividing class index by 4 maps 0-15 into groups 0, 1, 2, 3
                grp_label = label // 4
                grp_preds = [p // 4 for p in preds]
                
                if grp_label == grp_preds[0]:
                    grp_correct_top1 += 1
                if grp_label in grp_preds[:2]:
                    grp_correct_top2 += 1
                if grp_label in grp_preds[:3]:
                    grp_correct_top3 += 1

                # Group2 Accuracy (m6, m30, m60, m90): index % 4 maps 0-15 into groups 0, 1, 2, 3
                grp2_label = label % 4
                grp2_preds = [p % 4 for p in preds]

                if grp2_label == grp2_preds[0]:
                    grp2_correct_top1 += 1
                if grp2_label in grp2_preds[:2]:
                    grp2_correct_top2 += 1
                if grp2_label in grp2_preds[:3]:
                    grp2_correct_top3 += 1

    metrics = {
        'loss': val_loss / len(test_loader),
        'top1': 100. * correct_top1 / total,
        'top2': 100. * correct_top2 / total,
        'top3': 100. * correct_top3 / total,
        'grp_top1': 100. * grp_correct_top1 / total,
        'grp_top2': 100. * grp_correct_top2 / total,
        'grp_top3': 100. * grp_correct_top3 / total,
        'grp2_top1': 100. * grp2_correct_top1 / total,
        'grp2_top2': 100. * grp2_correct_top2 / total,
        'grp2_top3': 100. * grp2_correct_top3 / total,
    }
    return metrics
# ==========================================
# 3. Main
# ==========================================
def main(manual_seed=None):
    #DEBUG 1
    torch.autograd.set_detect_anomaly(True)
    
    clear_gpu_memory()
    # --- Dynamic Seed Generation ---
    if manual_seed is None:
        actual_seed = random.randint(0, 2**32 - 1)
    else:
        actual_seed = manual_seed

    # Apply the seed immediately
    set_seed(actual_seed)

    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise EnvironmentError("CUDA is not available. Please check your GPU setup.")
        
    # --- Training Parameters ---
    params = {
        "seed": actual_seed, 
        "device": 'cuda',
        "batch_size": 6,
        "start_epoch": 0,      # Default value, will be auto-updated if resuming
        "epochs": 1200,         
        "lr": 1e-4,
        "num_classes": 16,
        "in_chans": 1,
        "test_every": 5,
        "embed_dim": 96,
        "depths": [6, 6, 6, 6],
        "num_heads": [4, 4, 4, 4],

        # --- Auxiliary Multi-Task Loss Settings ---
        # n is already ~solved (96.9% top-1), m is the bottleneck (60.9% top-1),
        # so the m auxiliary head is weighted higher to push extra gradient signal there.
        "aux_weight_n": 0.2,
        "aux_weight_m": 0.5,

        # --- Resume Training Settings ---
        "resume_path": "model_weights/20260720_140225/model_epoch_1200_acc_60.9.pth", # Path to the checkpoint to resume training
        "resume_append_dir": False  # Whether to append to the existing run directory when resuming
    }

    
    start_wall_time = time.time()
    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    device = torch.device(params["device"])
    
    # --- Determine Run Directory and Auto-detect Epoch ---
    is_resuming = params["resume_path"] is not None and os.path.exists(params["resume_path"])
    
    if is_resuming and params["resume_append_dir"]:
        # Extract the directory of the existing checkpoint
        run_dir = os.path.dirname(params["resume_path"])
        print(f"Appending to existing run directory: {run_dir}")
        
        # Auto-detect epoch from filename using regex
        filename = os.path.basename(params["resume_path"])
        match = re.search(r'model_epoch_(\d+)', filename)
        if match:
            params["start_epoch"] = int(match.group(1))
            print(f"Auto-detected starting epoch: {params['start_epoch']} (Next training epoch will be {params['start_epoch'] + 1})")
        else:
            print("[WARNING] Could not auto-detect epoch from filename. Using default start_epoch.")
    else:
        # Create a completely new directory
        run_dir = os.path.join('model_weights', current_time_str)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created new run directory: {run_dir}")
    
    info_file = os.path.join(run_dir, 'training_info.txt')
    
    # --- Source Code Backup (Only if creating a new directory) ---
    if not (is_resuming and params["resume_append_dir"]):
        try:
            current_script = __file__
            shutil.copy2(current_script, os.path.join(run_dir, os.path.basename(current_script)))
            model_file = 'models/network_swinir_recong.py'
            if os.path.exists(model_file):
                shutil.copy2(model_file, os.path.join(run_dir, 'network_swinir_recong_backup.py'))
        except NameError:
            print("Warning: Could not identify source file for backup.")
            
        # Write initial parameter log
        with open(info_file, 'w') as f:
            f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("--- Hyperparameters ---\n")
            for k, v in params.items():
                f.write(f"{k}: {v}\n")
            f.write("------------------------\n\n")

    # Log resume action
    if is_resuming:
        with open(info_file, 'a') as f:
            f.write(f"\n--- Resumed Training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Loaded Checkpoint: {params['resume_path']}\n")
            f.write(f"Starting from Epoch: {params['start_epoch']}\n\n")

    # --- 1. Prepare Data ---
    #Data Augmentation
    train_transform = T.Compose([
        T.RandomAffine(
            degrees=180,           # random rotation between -180 and 180 degrees
            translate=(0.15, 0.15),  # random translation up to 30% of image dimensions
            scale=(0.7, 1.4),      # random scaling 
            fill=0                 # fill the regions outside the image with 0
        ),
        T.RandomHorizontalFlip(p=0.5), # 50% probability of horizontal flip
        T.RandomVerticalFlip(p=0.5)    # 50% probability of vertical flip
    ])
        
    train_set = ScientificRecognitionDataset('data/train', in_chans=params["in_chans"], transform=train_transform)
    test_set = ScientificRecognitionDataset('data/test', in_chans=params["in_chans"], transform=None)
    
    train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
     
    sample_dir = os.path.join(run_dir, 'augmented_samples')
    save_augmented_samples(train_loader, sample_dir, num_samples=10)
    # --- 2. Initialize Model ---
    model = SwinIR(
        img_size=64, #if the number is not compatible with intput image size, it will be automatically adjusted in the model's forward method
        in_chans=params["in_chans"],
        num_classes=params["num_classes"],
        num_n_groups=4,
        num_m_groups=4,
        window_size=8,
        depths=params["depths"], 
        embed_dim=params["embed_dim"], 
        num_heads=params["num_heads"],
        mlp_ratio=2, 
        upsampler=''
    ).to(device)

    # --- 2.5 Load Resume Checkpoint ---
    if is_resuming:
        print(f"Loading weights from checkpoint: {params['resume_path']}")
        # strict=False: older checkpoints saved before the aux heads existed won't have
        # head_n/head_m weights; those simply stay randomly initialized.
        missing_unexpected = model.load_state_dict(torch.load(params["resume_path"], map_location=device), strict=False)
        print(f"Checkpoint loaded. Missing keys: {missing_unexpected.missing_keys}, Unexpected keys: {missing_unexpected.unexpected_keys}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4, eps=1e-8)
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Initialize ReduceLROnPlateau Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.75, 
        patience=5, 
        min_lr=1e-6
    )
    
    print(f"Training execution starts. Results will be saved to: {run_dir}")
    
# --- 3. Training Loop ---
    # The loop will start from start_epoch and continue until epochs
    for epoch in range(params["start_epoch"], params["epochs"]):
        model.train()
        train_loss_main = 0.0
        train_loss_aux = 0.0
        train_correct = 0
        train_total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            # n-group = label // 4 (n1,n2,n4,n6), m-group = label % 4 (m6,m30,m60,m90);
            # consistent with the grouping already used in validate()'s grp/grp2 metrics.
            n_labels = labels // 4
            m_labels = labels % 4

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs, outputs_n, outputs_m = model(inputs, return_aux=True)

                loss_main = criterion(outputs, labels)
                loss_n = criterion(outputs_n, n_labels)
                loss_m = criterion(outputs_m, m_labels)
                loss = loss_main + params["aux_weight_n"] * loss_n + params["aux_weight_m"] * loss_m

            scaler.scale(loss).backward()

            #DEBUG 5
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss_main += loss_main.item()
            train_loss_aux += (loss_n.item() + loss_m.item())
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss_main / len(train_loader)
        avg_train_aux_loss = train_loss_aux / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation and Checkpoint Saving
        if (epoch + 1) % params["test_every"] == 0:
            val_metrics = validate(model, test_loader, device, criterion)
            val_loss = val_metrics['loss']
            
            # Fetch current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f">>> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Epoch {epoch+1}] ")
            print(f"    Train Loss: {avg_train_loss:.4f} (Aux n+m: {avg_train_aux_loss:.4f}) | Train Acc: {train_acc:.2f}% | LR: {current_lr:.2e}")
            print(f"    Val Loss:   {val_loss:.4f}")
            print(f"    n&m Acc -> Top1: {val_metrics['top1']:.2f}% | Top2: {val_metrics['top2']:.2f}% | Top3: {val_metrics['top3']:.2f}%")
            print(f"    n Acc -> Top1: {val_metrics['grp_top1']:.2f}% | Top2: {val_metrics['grp_top2']:.2f}% | Top3: {val_metrics['grp_top3']:.2f}%")
            print(f"    m Acc -> Top1: {val_metrics['grp2_top1']:.2f}% | Top2: {val_metrics['grp2_top2']:.2f}% | Top3: {val_metrics['grp2_top3']:.2f}%")
            print("-" * 60)
            
            save_path = os.path.join(run_dir, f'model_epoch_{epoch+1}_acc_{val_metrics["top1"]:.1f}.pth')
            torch.save(model.state_dict(), save_path)
            
            with open(info_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch+1}:\n"
                        f"  Train: Loss={avg_train_loss:.4f} (Aux n+m={avg_train_aux_loss:.4f}), Acc={train_acc:.2f}%, LR={current_lr:.2e}\n"
                        f"  Val:   Loss={val_loss:.4f}\n"
                        f"  n&m Acc: T1={val_metrics['top1']:.2f}%, T2={val_metrics['top2']:.2f}%, T3={val_metrics['top3']:.2f}%\n"
                        f"  n Acc: T1={val_metrics['grp_top1']:.2f}%, T2={val_metrics['grp_top2']:.2f}%, T3={val_metrics['grp_top3']:.2f}%\n"
                        f"  m Acc: T1={val_metrics['grp2_top1']:.2f}%, T2={val_metrics['grp2_top2']:.2f}%, T3={val_metrics['grp2_top3']:.2f}%\n\n")

            # Step the scheduler based on validation loss
            scheduler.step(val_loss)

        else:
            current_lr = optimizer.param_groups[0]['lr']
            print(f">>> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Epoch {epoch+1}] "
                  f"Train Loss: {avg_train_loss:.4f} (Aux n+m: {avg_train_aux_loss:.4f}) | Train Acc: {train_acc:.2f}% | LR: {current_lr:.2e}")
    # --- Finalize Training Time ---
    end_wall_time = time.time()
    total_duration = end_wall_time - start_wall_time
    duration_str = time.strftime("%H:%M:%S", time.gmtime(total_duration))
    
    with open(info_file, 'a') as f:
        f.write(f"\nTraining execution finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session Duration: {duration_str}\n")
    
    print(f"Training complete. Session time: {duration_str}")

if __name__ == '__main__':
    main(manual_seed=None)
