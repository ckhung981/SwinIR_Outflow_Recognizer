'''
# --- Inference & Validation Script for SwinIR Recognition ---
# This script loads a trained SwinIR model and runs inference or validation.
# - If TARGET_PATH is a file: Outputs top-3 predictions and group classification.
# - If TARGET_PATH is a directory: Runs full validation on the dataset (expects class subfolders)
#   and outputs Top-1/2/3 and Group Top-1/2/3 accuracies.
'''

import torch
import cv2
import numpy as np
import os
import ast
from torch.utils.data import DataLoader, Dataset
from models.network_swinir_recong import SwinIR

# ==========================================
# 1. Dataset & Utilities
# ==========================================
class ScientificRecognitionDataset(Dataset):
    def __init__(self, root_dir, in_chans=1):
        self.root_dir = root_dir
        self.in_chans = in_chans
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
        return img, label

def parse_training_info(txt_path):
    """Parses the training_info.txt file to extract model parameters."""
    params = {}
    if not os.path.exists(txt_path):
        print(f"Warning: {txt_path} not found. Using default parameters.")
        return params

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ':' in line:
                key, val = line.split(':', 1)
            elif '=' in line:
                key, val = line.split('=', 1)
            else:
                continue
            
            key = key.strip()
            val = val.strip()
            
            try:
                val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
            params[key] = val
            
    return params

def load_model_dynamically(model_path, classes_list, fallback_in_chans, device):
    """Loads the model architecture and weights based on training_info.txt."""
    model_dir = os.path.dirname(model_path)
    info_path = os.path.join(model_dir, 'training_info.txt')
    parsed_params = parse_training_info(info_path)
    
    model_kwargs = {
        'img_size': parsed_params.get('img_size', 64),
        'in_chans': parsed_params.get('in_chans', fallback_in_chans),
        'num_classes': parsed_params.get('num_classes', len(classes_list)),
        'window_size': parsed_params.get('window_size', 8),
        'depths': parsed_params.get('depths', [6, 6, 6, 6]),
        'embed_dim': parsed_params.get('embed_dim', 96),
        'num_heads': parsed_params.get('num_heads', [6, 6, 6, 6]),
        'mlp_ratio': parsed_params.get('mlp_ratio', 2),
        'upsampler': parsed_params.get('upsampler', '')
    }

    print("Initializing model with the following parameters:")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v}")

    model = SwinIR(**model_kwargs).to(device)
    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    # strict=False: checkpoints saved before the head_n/head_m auxiliary heads existed
    # won't have those keys; that's fine since inference only ever uses the main head.
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, model_kwargs['in_chans']

# ==========================================
# 2. Inference & Validation Logic
# ==========================================
def infer_single_image(model, image_path, classes_list, in_chans, device):
    """Runs inference on a single image and prints Top-3 and Group results."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    img_input = img.astype(np.float32) / 255.0
    
    if in_chans == 1:
        if len(img_input.shape) == 3:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0)
    else:
        if len(img_input.shape) == 2:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0)
    
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = model(img_tensor)
        else:
            output = model(img_tensor)
            
        probabilities = torch.nn.functional.softmax(output.float(), dim=1)
        
        k = min(3, len(classes_list))
        topk_confs, topk_indices = torch.topk(probabilities, k=k, dim=1)
        
        topk_confs_list = topk_confs[0].tolist()
        topk_indices_list = topk_indices[0].tolist()

    # Define the 4 main groups based on your class structure
    group_names = ['n1 (Group 0)', 'n2 (Group 1)', 'n4 (Group 2)', 'n6 (Group 3)']

    print("-" * 40)
    print(f"Inference Results for Single File: {os.path.basename(image_path)}")
    print("-" * 40)
    
    for rank, (idx, conf) in enumerate(zip(topk_indices_list, topk_confs_list), start=1):
        class_name = classes_list[idx]
        grp_idx = idx // 4
        grp_name = group_names[grp_idx]
        
        print(f" {rank}. {class_name} (Index: {idx})")
        print(f"    Confidence: {conf*100:.2f}% | Belongs to: {grp_name}")
        if rank == 1:
            print("    " + "-"*30)

def validate_directory(model, dir_path, in_chans, device):
    """Runs a full validation loop over a directory structured with class subfolders."""
    print(f"Starting directory validation on: {dir_path}")
    dataset = ScientificRecognitionDataset(dir_path, in_chans=in_chans)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    correct_top1, correct_top2, correct_top3 = 0, 0, 0
    grp_correct_top1, grp_correct_top2, grp_correct_top3 = 0, 0, 0
    grp2_correct_top1, grp2_correct_top2, grp2_correct_top3 = 0, 0, 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
            
            total += labels.size(0)
            _, top3_preds = outputs.topk(3, 1, True, True)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                preds = top3_preds[i].tolist()
                
                # Standard Accuracies
                if label == preds[0]: correct_top1 += 1
                if label in preds[:2]: correct_top2 += 1
                if label in preds[:3]: correct_top3 += 1
                    
                # Group Accuracies (0-3: Grp0, 4-7: Grp1, 8-11: Grp2, 12-15: Grp3)
                grp_label = label // 4
                grp_preds = [p // 4 for p in preds]
                
                if grp_label == grp_preds[0]: grp_correct_top1 += 1
                if grp_label in grp_preds[:2]: grp_correct_top2 += 1
                if grp_label in grp_preds[:3]: grp_correct_top3 += 1
                
                #Group2 Accuracy (m6, m30, m60, m90)
                grp2_label = label % 4
                grp2_preds = [p % 4 for p in preds]
                if grp2_label == grp2_preds[0]: grp2_correct_top1 += 1
                if grp2_label in grp2_preds[:2]: grp2_correct_top2 += 1
                if grp2_label in grp2_preds[:3]: grp2_correct_top3 += 1
                
    
    print("-" * 50)
    print(f"Validation Results for Directory: {os.path.basename(dir_path)}")
    print(f"Total Samples Processed: {total}")
    print("-" * 50)
    print("Standard Class Accuracy (16 Classes):")
    print(f"  Top-1: {100. * correct_top1 / total:.2f}%")
    print(f"  Top-2: {100. * correct_top2 / total:.2f}%")
    print(f"  Top-3: {100. * correct_top3 / total:.2f}%")
    print("-" * 50)
    print("Grouped Accuracy (4 Main Groups: n1, n2, n4, n6):")
    print(f"  Top-1: {100. * grp_correct_top1 / total:.2f}%")
    print(f"  Top-2: {100. * grp_correct_top2 / total:.2f}%")
    print(f"  Top-3: {100. * grp_correct_top3 / total:.2f}%")
    print("-" * 50)
    print("Grouped Accuracy (4 Sub-Groups: m6, m30, m60, m90):")
    print(f"  Top-1: {100. * grp2_correct_top1 / total:.2f}%")
    print(f"  Top-2: {100. * grp2_correct_top2 / total:.2f}%")
    print(f"  Top-3: {100. * grp2_correct_top3 / total:.2f}%")
    print("-" * 50)


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == '__main__':
    MODEL_WEIGHTS = 'model_weights/20260720_140225/model_epoch_1200_acc_60.9.pth' 
    
    # You can set TARGET_PATH to either a specific .tif file OR a directory like 'data/test'
    TARGET_PATH = 'data/test'  # Example for single file inference

    # Class order MUST match the label index order ScientificRecognitionDataset assigns during
    # training, i.e. sorted(os.listdir(<train_dir>)). A hand-typed "natural" order like
    # ['n1_m6','n1_m30',...] silently mismatches that (e.g. "n1_m30" < "n1_m6" lexicographically,
    # since '3' < '6'), causing predicted indices to be labeled with the wrong class name. Deriving
    # it from the actual training folder avoids that class of bug entirely.
    CLASSES_SOURCE_DIR = 'data/train'
    CLASSES = sorted(os.listdir(CLASSES_SOURCE_DIR))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load the model once
        model, in_chans = load_model_dynamically(
            model_path=MODEL_WEIGHTS, 
            classes_list=CLASSES, 
            fallback_in_chans=1, 
            device=device
        )
        
        # Route execution based on path type
        if os.path.isfile(TARGET_PATH):
            infer_single_image(model, TARGET_PATH, CLASSES, in_chans, device)
        elif os.path.isdir(TARGET_PATH):
            validate_directory(model, TARGET_PATH, in_chans, device)
        else:
            print(f"Error: TARGET_PATH '{TARGET_PATH}' does not exist.")
            
    except Exception as e:
        print(f"Execution failed: {e}")