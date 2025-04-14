import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Import local modules
from models.video_models import get_model

def load_video(video_path, clip_len=16, frame_sample_rate=2, crop_size=224):
    """
    Load video frames from file for inference.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get total frame count and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frames to sample
    sample_points = np.linspace(0, total_frames - 1, clip_len * frame_sample_rate, dtype=int)
    
    # Read frames
    frame_idx = 0
    progress_bar = tqdm(total=len(sample_points), desc="Loading video")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in sample_points:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to crop_size
            frame = cv2.resize(frame, (crop_size, crop_size))
            frames.append(frame)
            progress_bar.update(1)
            
            # Exit if we've collected enough frames
            if len(frames) >= clip_len:
                break
                
        frame_idx += 1
    
    cap.release()
    progress_bar.close()
    
    # Check if we have enough frames
    if len(frames) < clip_len:
        print(f"Warning: Only loaded {len(frames)} frames, expected {clip_len}")
        # Pad with the last frame if needed
        while len(frames) < clip_len:
            frames.append(frames[-1] if frames else np.zeros((crop_size, crop_size, 3), dtype=np.uint8))
    
    # Stack frames together
    clip = np.stack(frames)
    
    return clip, fps

def preprocess_clip(clip):
    """
    Preprocess video clip for model input.
    """
    # Convert to tensor, normalize to [0, 1]
    clip = torch.from_numpy(clip.astype(np.float32) / 255.0)
    # Reshape to [C, T, H, W]
    clip = clip.permute(3, 0, 1, 2)
    # Add batch dimension
    clip = clip.unsqueeze(0)
    
    return clip

def main():
    parser = argparse.ArgumentParser(description='Run inference with action recognition models')
    
    # Input parameters
    parser.add_argument('--video', required=True, type=str, help='Path to input video')
    
    # Model parameters
    parser.add_argument('--model', default='slowfast_r50', type=str, help='Model name (slowfast_r50, x3d_m)')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--clip_len', default=16, type=int, help='Number of frames per clip')
    parser.add_argument('--frame_rate', default=2, type=int, help='Frame sampling rate')
    parser.add_argument('--crop_size', default=224, type=int, help='Height and width of frames')
    parser.add_argument('--device', default='cuda', type=str, help='Device (cuda, cpu)')
    parser.add_argument('--top_k', default=5, type=int, help='Show top-k predictions')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_to_idx = checkpoint['class_to_idx']
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Create model
    model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=False
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess video
    print(f"Loading video: {args.video}")
    clip, fps = load_video(
        args.video,
        clip_len=args.clip_len,
        frame_sample_rate=args.frame_rate,
        crop_size=args.crop_size
    )
    
    if clip is None:
        print("Failed to load video. Exiting.")
        return
    
    print(f"Video FPS: {fps}, Processed {args.clip_len} frames")
    
    # Preprocess clip
    clip_tensor = preprocess_clip(clip)
    clip_tensor = clip_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(clip_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_prob, top_class = torch.topk(probabilities, args.top_k)
        
        top_prob = top_prob.squeeze().cpu().numpy()
        top_class = top_class.squeeze().cpu().numpy()
    
    # Print results
    print("\nPredictions:")
    print("-" * 50)
    for i in range(args.top_k):
        class_name = idx_to_class[top_class[i]]
        probability = top_prob[i] * 100
        print(f"{i+1}. {class_name}: {probability:.2f}%")

if __name__ == "__main__":
    main()
