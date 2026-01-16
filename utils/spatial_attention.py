import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

def visualize_spatial_attention(encoder, obs, episode, save_dir, layer_idx=3):
    """
    Visualize spatial attention maps overlaid on the original observation.
    """
    encoder.eval()
    
    # Handle batch dimension
    if obs.dim() == 3:
        obs_input = obs.unsqueeze(0)
    else:
        obs_input = obs
    
    with torch.no_grad():
        attention_maps = encoder.observation_encoder.spatial_attention(obs_input)
    
    # Convert observation to numpy for visualization
    obs_np = obs_input[0].cpu().numpy() 
    obs_np = np.transpose(obs_np, (1, 2, 0))  
    obs_np = np.clip(obs_np, 0, 1)  

    num_layers = len(attention_maps)
    print(num_layers)
    
    if layer_idx is not None:
        attention_np = attention_maps[layer_idx][0].cpu().numpy()
        attention_resized = F.interpolate(
            torch.from_numpy(attention_np).unsqueeze(0).unsqueeze(0),
            size=obs_np.shape[:2],
            mode='bilinear',
            align_corners=False
        )[0, 0].numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(obs_np)
        axes[0].set_title('Original Observation')
        axes[0].axis('off')
        
        im = axes[1].imshow(attention_resized, cmap='jet')
        axes[1].set_title(f'Attention Map (Layer {layer_idx})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        axes[2].imshow(obs_np)
        axes[2].imshow(attention_resized, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'spatial_attention_map_' + episode + '.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    else:
        fig, axes = plt.subplots(2, num_layers + 1, figsize=(5 * (num_layers + 1), 10))
        
        # # Original image in first column
        # for i in range(2):
        #     axes[i, 0].imshow(obs_np)
        #     axes[i, 0].set_title('Original')
        #     axes[i, 0].axis('off')
        
        # # Each layer's attention
        # for layer_idx in range(num_layers):
        #     attention_np = attention_maps[layer_idx][0].cpu().numpy()
        #     attention_resized = F.interpolate(
        #         torch.from_numpy(attention_np).unsqueeze(0).unsqueeze(0),
        #         size=obs_np.shape[:2],
        #         mode='bilinear',
        #         align_corners=False
        #     )[0, 0].numpy()
            
        #     # Heatmap only
        #     im = axes[0, layer_idx + 1].imshow(attention_resized, cmap='jet')
        #     axes[0, layer_idx + 1].set_title(f'Layer {layer_idx + 1}')
        #     axes[0, layer_idx + 1].axis('off')
        #     plt.colorbar(im, ax=axes[0, layer_idx + 1], fraction=0.046)
            
        #     # Overlay
        #     axes[1, layer_idx + 1].imshow(obs_np)
        #     axes[1, layer_idx + 1].imshow(attention_resized, cmap='jet', alpha=0.5)
        #     axes[1, layer_idx + 1].set_title(f'Overlay {layer_idx + 1}')
        #     axes[1, layer_idx + 1].axis('off')
        
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, f'attention_all_layers_step{step}.png'), 
        #             dpi=150, bbox_inches='tight')
        # plt.close()