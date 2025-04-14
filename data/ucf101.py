import os
import glob
import cv2
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class UCF101Dataset(Dataset):
    """UCF101 video dataset for action recognition."""
    
    def __init__(self, 
                 root_dir, 
                 annotation_path,
                 split='train',
                 transform=None,
                 clip_len=16,
                 frame_sample_rate=2,
                 crop_size=224,
                 temporal_jitter=True):
        """
        Args:
            root_dir (string): Directory with all the video files.
            annotation_path (string): Path to the annotation file.
            split (string): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
            clip_len (int): Number of frames in a clip.
            frame_sample_rate (int): Sample every nth frame.
            crop_size (int): Height and width of inputs.
            temporal_jitter (bool): Whether to apply temporal jittering.
        """
        self.root_dir = root_dir
        self.annotation_path = annotation_path
        self.split = split
        self.transform = transform
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.temporal_jitter = temporal_jitter
        
        # Read annotations file
        self.annotations = self._parse_annotations()
        
        # Create class mapping
        self.class_to_idx = {}
        for i, (_, label) in enumerate(self.annotations):
            if label not in self.class_to_idx:
                self.class_to_idx[label] = len(self.class_to_idx)
                
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def _parse_annotations(self):
        """Parse the annotation file to get video paths and labels."""
        annotations = []
        
        with open(self.annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    video_file, class_idx = line.split(' ')
                    class_name = video_file.split('/')[0]  # Extract class name from path
                    
                    # Only include videos for the current split
                    if (self.split == 'train' and int(class_idx) == 1) or \
                       (self.split == 'test' and int(class_idx) == 2):
                        annotations.append((os.path.join(self.root_dir, video_file), class_name))
        
        return annotations

    def __len__(self):
        return len(self.annotations)

    def _load_video(self, video_path):
        """
        Load video frames from file.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frames to sample
        if self.temporal_jitter:
            # Temporal jittering: randomly select starting frame
            if total_frames > (self.clip_len * self.frame_sample_rate):
                max_start = total_frames - (self.clip_len * self.frame_sample_rate)
                start_frame = random.randint(0, max_start)
            else:
                start_frame = 0
        else:
            # Uniform sampling: evenly sample frames across video
            if total_frames >= self.clip_len * self.frame_sample_rate:
                skip_frames = (total_frames - self.clip_len * self.frame_sample_rate) // 2
                start_frame = skip_frames
            else:
                start_frame = 0
        
        # Set starting frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read frames
        frame_count = 0
        while len(frames) < self.clip_len and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample every nth frame
            if frame_count % self.frame_sample_rate == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to crop_size
                frame = cv2.resize(frame, (self.crop_size, self.crop_size))
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        # If we didn't get enough frames, loop the video
        while len(frames) < self.clip_len:
            frames.append(frames[-1])
        
        # Stack frames together
        clip = np.stack(frames)
        
        return clip

    def __getitem__(self, idx):
        video_path, label = self.annotations[idx]
        
        # Load video frames
        clip = self._load_video(video_path)
        
        # Apply transforms if specified
        if self.transform:
            clip = self.transform(clip)
        else:
            # Convert to tensor, normalize to [0, 1]
            clip = torch.from_numpy(clip.astype(np.float32) / 255.0)
            # Reshape to [C, T, H, W]
            clip = clip.permute(3, 0, 1, 2)
        
        # Get label index
        label_idx = self.class_to_idx[label]
        
        return clip, label_idx

def download_ucf101():
    """Download UCF101 dataset using Kaggle API."""
    try:
        import kaggle
        print("Downloading UCF101 dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('pevogam/ucf101', path='./data', unzip=True)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have the Kaggle API credentials set up.")
        print("Create a kaggle.json file in ~/.kaggle/ with your Kaggle username and API key.")

def prepare_ucf101(root_dir='./data/UCF101', annotation_dir='./data/ucfTrainTestlist'):
    """
    Prepare UCF101 dataset by organizing files and creating splits.
    Returns paths to train and test annotation files.
    """
    # Ensure directories exist
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    
    # Check if dataset is downloaded
    if not os.path.exists(root_dir) or len(os.listdir(root_dir)) == 0:
        download_ucf101()
    
    # Create class mapping (class name -> index)
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {class_names[i]: i+1 for i in range(len(class_names))}
    
    # Create annotation file paths
    train_file = os.path.join(annotation_dir, 'trainlist01.txt')
    test_file = os.path.join(annotation_dir, 'testlist01.txt')
    
    return train_file, test_file, class_to_idx

def get_dataloaders(root_dir='./data/UCF101', 
                   annotation_dir='./data/ucfTrainTestlist',
                   batch_size=8,
                   clip_len=16,
                   frame_sample_rate=2,
                   crop_size=224,
                   num_workers=4):
    """
    Create DataLoaders for training and testing.
    """
    from torchvision import transforms
    
    # Prepare dataset
    train_file, test_file, _ = prepare_ucf101(root_dir, annotation_dir)
    
    # Define transforms
    train_transform = transforms.Compose([
        # Add data augmentation here if needed
    ])
    
    test_transform = transforms.Compose([
        # Test transforms if needed
    ])
    
    # Create datasets
    train_dataset = UCF101Dataset(
        root_dir=root_dir,
        annotation_path=train_file,
        split='train',
        transform=train_transform,
        clip_len=clip_len,
        frame_sample_rate=frame_sample_rate,
        crop_size=crop_size,
        temporal_jitter=True
    )
    
    test_dataset = UCF101Dataset(
        root_dir=root_dir,
        annotation_path=test_file,
        split='test',
        transform=test_transform,
        clip_len=clip_len,
        frame_sample_rate=frame_sample_rate,
        crop_size=crop_size,
        temporal_jitter=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.class_to_idx

if __name__ == "__main__":
    # Test data module
    train_loader, test_loader, class_to_idx = get_dataloaders(
        batch_size=4,
        clip_len=16,
        frame_sample_rate=2,
        crop_size=224
    )
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Number of classes: {len(class_to_idx)}")
    
    # Sample a batch
    for clips, labels in train_loader:
        print(f"Batch shape: {clips.shape}")
        print(f"Labels: {labels}")
        break
