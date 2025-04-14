import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Import local modules
from data.ucf101 import get_dataloaders
from models.video_models import get_model

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for i, (inputs, targets) in enumerate(progress_bar):
        # Move data to target device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (i + 1),
            'acc': 100. * correct / total
        })
    
    # Epoch statistics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Log to TensorBoard
    if writer:
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, epoch, writer=None):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(progress_bar):
            # Move data to target device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (i + 1),
                'acc': 100. * correct / total
            })
    
    # Epoch statistics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Log to TensorBoard
    if writer:
        writer.add_scalar('Loss/val', epoch_loss, epoch)
        writer.add_scalar('Accuracy/val', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description='Train action recognition models on UCF101')
    
    # Dataset parameters
    parser.add_argument('--data_root', default='./data/UCF101', type=str, help='Path to UCF101 videos')
    parser.add_argument('--anno_dir', default='./data/ucfTrainTestlist', type=str, help='Path to annotations')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--clip_len', default=16, type=int, help='Number of frames per clip')
    parser.add_argument('--frame_rate', default=2, type=int, help='Frame sampling rate')
    parser.add_argument('--crop_size', default=224, type=int, help='Height and width of frames')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading')
    
    # Model parameters
    parser.add_argument('--model', default='slowfast_r50', type=str, help='Model name (slowfast_r50, x3d_m)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone and train only the head')
    
    # Training parameters
    parser.add_argument('--epochs', default=30, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--device', default='cuda', type=str, help='Device (cuda, cpu)')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str, help='Path to save checkpoints')
    parser.add_argument('--log_dir', default='./logs', type=str, help='Path to save logs')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, class_to_idx = get_dataloaders(
        root_dir=args.data_root,
        annotation_dir=args.anno_dir,
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        frame_sample_rate=args.frame_rate,
        crop_size=args.crop_size,
        num_workers=args.num_workers
    )
    
    num_classes = len(class_to_idx)
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained
    )
    
    if args.freeze_backbone:
        model.freeze_backbone()
        print("Backbone frozen! Only training the classification head.")
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use different learning rates for backbone and head if backbone is frozen
    if args.freeze_backbone:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"))
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f"{args.model}_best.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_to_idx': class_to_idx
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path} with accuracy: {val_acc:.2f}%")
        
        # Print epoch results
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 40)
    
    writer.close()
    
    # Save final model
    final_checkpoint_path = os.path.join(
        args.checkpoint_dir, 
        f"{args.model}_final.pth"
    )
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx
    }, final_checkpoint_path)
    print(f"Saved final model checkpoint to {final_checkpoint_path}")

if __name__ == "__main__":
    main()
