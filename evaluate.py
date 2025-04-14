import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Import local modules
from data.ucf101 import get_dataloaders
from models.video_models import get_model

def evaluate(model, dataloader, device):
    """Evaluate the model on the validation set."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            # Move data to target device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            # Store predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate accuracy
    accuracy = (all_preds == all_targets).mean() * 100
    
    return accuracy, all_preds, all_targets

def main():
    parser = argparse.ArgumentParser(description='Evaluate action recognition models on UCF101')
    
    # Dataset parameters
    parser.add_argument('--data_root', default='./data/UCF101', type=str, help='Path to UCF101 videos')
    parser.add_argument('--anno_dir', default='./data/ucfTrainTestlist', type=str, help='Path to annotations')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--clip_len', default=16, type=int, help='Number of frames per clip')
    parser.add_argument('--frame_rate', default=2, type=int, help='Frame sampling rate')
    parser.add_argument('--crop_size', default=224, type=int, help='Height and width of frames')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading')
    
    # Model parameters
    parser.add_argument('--model', default='slowfast_r50', type=str, help='Model name (slowfast_r50, x3d_m)')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--device', default='cuda', type=str, help='Device (cuda, cpu)')
    parser.add_argument('--output_dir', default='./results', type=str, help='Path to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_to_idx = checkpoint['class_to_idx']
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Create dataloaders (only need validation)
    _, val_loader, _ = get_dataloaders(
        root_dir=args.data_root,
        annotation_dir=args.anno_dir,
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        frame_sample_rate=args.frame_rate,
        crop_size=args.crop_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=False
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    accuracy, all_preds, all_targets = evaluate(model, val_loader, device)
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Generate classification report
    class_names = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(
        all_targets, all_preds, 
        target_names=class_names, 
        digits=3
    )
    
    # Save results
    np.save(os.path.join(args.output_dir, f"{args.model}_confusion_matrix.npy"), cm)
    
    with open(os.path.join(args.output_dir, f"{args.model}_classification_report.txt"), 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
