import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import pytorchvideo.models
from pytorchvideo.models.head import create_res_basic_head

class ActionRecognitionModel(nn.Module):
    """
    Wrapper for video models (SlowFast, X3D) for action recognition.
    """
    def __init__(self, 
                 model_name='slowfast_r50', 
                 pretrained=True, 
                 num_classes=101):
        super(ActionRecognitionModel, self).__init__()
        
        self.model_name = model_name
        
        # Initialize model based on name
        if model_name == 'slowfast_r50':
            # SlowFast R50 model
            self.model = pytorchvideo.models.slowfast.create_slowfast(
                input_channels=(3, 3),
                model_depth=50,
                model_num_class=num_classes,
                dropout_rate=0.5,
                pretrained=pretrained
            )
            
            # If pretrained, replace the classification head
            if pretrained:
                # Modify the head to match our number of classes
                self.model.blocks[-1] = create_res_basic_head(
                    in_features=self.model.blocks[-1].proj.in_features,
                    out_features=num_classes,
                    pool=nn.AvgPool3d,
                    dropout_rate=0.5
                )
                
        elif model_name == 'x3d_m':
            # X3D-M model
            self.model = pytorchvideo.models.x3d.create_x3d(
                input_clip_length=16,
                input_crop_size=224,
                model_num_class=num_classes,
                dropout_rate=0.5,
                pretrained=pretrained
            )
            
            # If pretrained, replace the classification head
            if pretrained:
                # Modify the head to match our number of classes
                in_features = self.model.blocks[-1].proj.in_features
                self.model.blocks[-1].proj = nn.Linear(in_features, num_classes)
                
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W]
                For SlowFast, requires separate inputs for slow and fast pathways
        """
        if self.model_name == 'slowfast_r50':
            # SlowFast requires separate inputs for slow and fast pathways
            # Split the input along the temporal dimension
            fast_pathway = x
            
            # For slow pathway, we sample frames at a lower temporal resolution
            # Typically 1/4 or 1/8 of the fast pathway
            t = x.shape[2]
            stride = 4  # Assuming 1/4 temporal sampling for slow pathway
            slow_indices = torch.linspace(0, t-1, t//stride).long()
            slow_pathway = torch.index_select(x, 2, slow_indices.to(x.device))
            
            # Forward pass with both pathways
            return self.model([slow_pathway, fast_pathway])
        else:
            # Standard forward pass for X3D and other models
            return self.model(x)
    
    def freeze_backbone(self, freeze=True):
        """
        Freeze or unfreeze the backbone of the model.
        Only train the classification head.
        """
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = not freeze
            
        # Unfreeze the classification head
        if self.model_name == 'slowfast_r50':
            for param in self.model.blocks[-1].parameters():
                param.requires_grad = True
        elif self.model_name == 'x3d_m':
            for param in self.model.blocks[-1].parameters():
                param.requires_grad = True

def get_model(model_name, num_classes=101, pretrained=True):
    """
    Factory function to create a model with the given name.
    
    Args:
        model_name (str): Name of the model (slowfast_r50, x3d_m)
        num_classes (int): Number of classes for classification
        pretrained (bool): Whether to load pretrained weights
        
    Returns:
        model (nn.Module): The created model
    """
    return ActionRecognitionModel(model_name=model_name, 
                                 pretrained=pretrained,
                                 num_classes=num_classes)
