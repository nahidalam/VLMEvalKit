"""
Spatial Coherence Analysis for Vision Encoders

Quantifies how well vision encoders preserve spatial locality by comparing
similarity of nearby vs distant patch features.
"""

import torch
import numpy as np
from PIL import Image
import os.path as osp
from os.path import expanduser
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoModel, AutoProcessor,
    Dinov2Model, AutoImageProcessor
)
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ENCODER WRAPPERS
# ============================================================================

class EncoderWrapper:
    """Base wrapper for vision encoders."""
    
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
    
    def extract_patch_features(self, image_path: str) -> np.ndarray:
        """
        Extract patch features from image.
        
        Returns:
            np.ndarray of shape [H*W, D] where H, W are spatial dims, D is feature dim
        """
        raise NotImplementedError
    
    def get_spatial_dims(self) -> Tuple[int, int]:
        """Return (H, W) spatial dimensions of the feature map."""
        raise NotImplementedError


class CLIPEncoder(EncoderWrapper):
    """CLIP vision encoder."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", device: str = 'cuda'):
        super().__init__(model_name, device)
        print(f"Loading CLIP: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Will be determined from actual feature extraction
        self.grid_size = None
        
    def extract_patch_features(self, image_path: str) -> np.ndarray:
        """Extract patch features from CLIP vision encoder."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Get vision model outputs
            vision_outputs = self.model.vision_model(
                pixel_values=inputs.pixel_values,
                output_hidden_states=True
            )
            # Last hidden state: [batch, num_patches + 1, hidden_dim]
            # CLIP has CLS token - remove it (first token)
            patch_features = vision_outputs.last_hidden_state[:, 1:, :]
        
        # Shape: [1, num_patches, hidden_dim] -> [num_patches, hidden_dim]
        features = patch_features[0].cpu().numpy()
        
        # Infer grid size from number of patches
        num_patches = features.shape[0]
        grid_size = int(np.sqrt(num_patches))
        if self.grid_size is None:
            self.grid_size = grid_size
        
        return features
    
    def get_spatial_dims(self) -> Tuple[int, int]:
        """Get spatial dimensions (will be set after first extraction)."""
        if self.grid_size is None:
            # Default guess, will be updated
            return (14, 14)
        return (self.grid_size, self.grid_size)


class SigLIPEncoder(EncoderWrapper):
    """SigLIP vision encoder."""
    
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = 'cuda'):
        super().__init__(model_name, device)
        print(f"Loading SigLIP: {model_name}")
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Will be determined from actual feature extraction
        self.grid_size = None
        
    def extract_patch_features(self, image_path: str) -> np.ndarray:
        """Extract patch features from SigLIP vision encoder."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Get vision model outputs
            vision_outputs = self.model.vision_model(
                pixel_values=inputs.pixel_values,
                output_hidden_states=True
            )
            # SigLIP may or may not have CLS token - check the shape
            hidden_state = vision_outputs.last_hidden_state
            num_tokens = hidden_state.shape[1]
            
            # Try to detect if there's a CLS token
            # If num_tokens is a perfect square, no CLS token
            # If num_tokens - 1 is a perfect square, has CLS token
            num_patches_with_cls = num_tokens - 1
            grid_with_cls = int(np.sqrt(num_patches_with_cls))
            grid_no_cls = int(np.sqrt(num_tokens))
            
            if grid_no_cls * grid_no_cls == num_tokens:
                # No CLS token
                patch_features = hidden_state
                self.grid_size = grid_no_cls
            elif grid_with_cls * grid_with_cls == num_patches_with_cls:
                # Has CLS token
                patch_features = hidden_state[:, 1:, :]
                self.grid_size = grid_with_cls
            else:
                # Assume no CLS token
                patch_features = hidden_state
                self.grid_size = grid_no_cls
        
        features = patch_features[0].cpu().numpy()
        return features
    
    def get_spatial_dims(self) -> Tuple[int, int]:
        """Get spatial dimensions (will be set after first extraction)."""
        if self.grid_size is None:
            return (14, 14)
        return (self.grid_size, self.grid_size)


class DINOv2Encoder(EncoderWrapper):
    """DINOv2 vision encoder."""
    
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = 'cuda'):
        super().__init__(model_name, device)
        print(f"Loading DINOv2: {model_name}")
        self.model = Dinov2Model.from_pretrained(model_name).to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Will be determined from actual feature extraction
        self.grid_size = None
        
    def extract_patch_features(self, image_path: str) -> np.ndarray:
        """Extract patch features from DINOv2 encoder."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                pixel_values=inputs.pixel_values,
                output_hidden_states=True
            )
            # Last hidden state: [batch, num_patches + 1, hidden_dim]
            # DINOv2 has CLS token - remove it (first token)
            hidden_state = outputs.last_hidden_state
            num_tokens = hidden_state.shape[1]
            
            # Check for CLS token
            num_patches_with_cls = num_tokens - 1
            grid_with_cls = int(np.sqrt(num_patches_with_cls))
            grid_no_cls = int(np.sqrt(num_tokens))
            
            if grid_with_cls * grid_with_cls == num_patches_with_cls:
                # Has CLS token (DINOv2 typically does)
                patch_features = hidden_state[:, 1:, :]
                self.grid_size = grid_with_cls
            elif grid_no_cls * grid_no_cls == num_tokens:
                # No CLS token
                patch_features = hidden_state
                self.grid_size = grid_no_cls
            else:
                # Fallback: assume CLS token
                patch_features = hidden_state[:, 1:, :]
                self.grid_size = grid_with_cls
        
        features = patch_features[0].cpu().numpy()
        return features
    
    def get_spatial_dims(self) -> Tuple[int, int]:
        """Get spatial dimensions (will be set after first extraction)."""
        if self.grid_size is None:
            return (14, 14)
        return (self.grid_size, self.grid_size)


class AIMv2Encoder(EncoderWrapper):
    """AIMv2 vision encoder."""
    
    def __init__(self, model_name: str = "apple/aimv2-large-patch14-224", device: str = 'cuda'):
        super().__init__(model_name, device)
        print(f"Loading AIMv2: {model_name}")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        
        # Will be determined from actual feature extraction
        self.grid_size = None
        
    def extract_patch_features(self, image_path: str) -> np.ndarray:
        """Extract patch features from AIMv2 encoder."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                pixel_values=inputs.pixel_values,
                output_hidden_states=True
            )
            # Get hidden state
            if hasattr(outputs, 'last_hidden_state'):
                hidden_state = outputs.last_hidden_state
            else:
                hidden_state = outputs.hidden_states[-1]
            
            # Check if there's a CLS token
            num_tokens = hidden_state.shape[1]
            num_patches_with_cls = num_tokens - 1
            grid_with_cls = int(np.sqrt(num_patches_with_cls))
            grid_no_cls = int(np.sqrt(num_tokens))
            
            if grid_no_cls * grid_no_cls == num_tokens:
                # No CLS token
                patch_features = hidden_state
                self.grid_size = grid_no_cls
            elif grid_with_cls * grid_with_cls == num_patches_with_cls:
                # Has CLS token
                patch_features = hidden_state[:, 1:, :]
                self.grid_size = grid_with_cls
            else:
                # Assume no CLS token
                patch_features = hidden_state
                self.grid_size = grid_no_cls
        
        features = patch_features[0].cpu().numpy()
        return features
    
    def get_spatial_dims(self) -> Tuple[int, int]:
        """Get spatial dimensions (will be set after first extraction)."""
        if self.grid_size is None:
            return (16, 16)
        return (self.grid_size, self.grid_size)


# ============================================================================
# SPATIAL COHERENCE COMPUTATION
# ============================================================================

def get_patch_coordinates(idx: int, H: int, W: int) -> Tuple[int, int]:
    """Convert flat patch index to 2D coordinates."""
    i = idx // W
    j = idx % W
    return (i, j)


def chebyshev_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
    """Compute Chebyshev distance (L-infinity) between two coordinates."""
    return max(abs(coord1[0] - coord2[0]), abs(coord1[1] - coord2[1]))


def compute_local_neighbors(idx: int, H: int, W: int, max_dist: int = 1) -> List[int]:
    """Get indices of local neighbors within Chebyshev distance."""
    i, j = get_patch_coordinates(idx, H, W)
    neighbors = []
    
    for di in range(-max_dist, max_dist + 1):
        for dj in range(-max_dist, max_dist + 1):
            if di == 0 and dj == 0:
                continue  # Skip self
            
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                neighbor_idx = ni * W + nj
                neighbors.append(neighbor_idx)
    
    return neighbors


def compute_distant_patches(idx: int, H: int, W: int, min_dist: int = None) -> List[int]:
    """Get indices of distant patches."""
    if min_dist is None:
        min_dist = min(H, W) // 2
    
    i, j = get_patch_coordinates(idx, H, W)
    distant = []
    
    for other_idx in range(H * W):
        if other_idx == idx:
            continue
        
        other_i, other_j = get_patch_coordinates(other_idx, H, W)
        dist = chebyshev_distance((i, j), (other_i, other_j))
        
        if dist >= min_dist:
            distant.append(other_idx)
    
    return distant


def cosine_similarity_np(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def compute_spatial_coherence(patch_features: np.ndarray, H: int, W: int) -> Dict:
    """
    Compute spatial coherence metrics for patch features.
    
    Args:
        patch_features: [H*W, D] array of patch features
        H, W: spatial dimensions
    
    Returns:
        Dictionary with local_sim, distant_sim, and SCR
    """
    num_patches = H * W
    assert patch_features.shape[0] == num_patches, \
        f"Expected {num_patches} patches, got {patch_features.shape[0]}"
    
    local_sims = []
    distant_sims = []
    
    # Compute for each patch
    for patch_idx in range(num_patches):
        patch_vec = patch_features[patch_idx]
        
        # Get local neighbors
        local_neighbors = compute_local_neighbors(patch_idx, H, W, max_dist=1)
        if local_neighbors:
            local_sim = np.mean([
                cosine_similarity_np(patch_vec, patch_features[n])
                for n in local_neighbors
            ])
            local_sims.append(local_sim)
        
        # Get distant patches
        distant_patches = compute_distant_patches(patch_idx, H, W)
        if distant_patches:
            distant_sim = np.mean([
                cosine_similarity_np(patch_vec, patch_features[d])
                for d in distant_patches
            ])
            distant_sims.append(distant_sim)
    
    # Aggregate
    mean_local_sim = np.mean(local_sims) if local_sims else 0.0
    mean_distant_sim = np.mean(distant_sims) if distant_sims else 0.0
    
    # Compute SCR
    scr = mean_local_sim / mean_distant_sim if mean_distant_sim > 0 else 0.0
    
    return {
        'local_sim': float(mean_local_sim),
        'distant_sim': float(mean_distant_sim),
        'scr': float(scr),
        'num_patches': num_patches
    }


# ============================================================================
# BATCH EVALUATION
# ============================================================================

def evaluate_encoder_on_images(
    encoder: EncoderWrapper,
    image_paths: List[str],
    debug: bool = False
) -> Dict:
    """
    Evaluate spatial coherence for an encoder on multiple images.
    
    Args:
        encoder: Encoder wrapper instance
        image_paths: List of paths to test images
        debug: If True, print detailed debugging info
    
    Returns:
        Dictionary with per-image and aggregated results
    """
    results = {
        'encoder': encoder.model_name,
        'images': [],
        'aggregated': {},
        'errors': []
    }
    
    all_local_sims = []
    all_distant_sims = []
    all_scrs = []
    
    print(f"\nProcessing {len(image_paths)} images with {encoder.model_name}...")
    
    for img_path in tqdm(image_paths, disable=debug):
        try:
            if debug:
                print(f"\n  Processing: {osp.basename(img_path)}")
            
            # Extract features (this will also set grid_size)
            patch_features = encoder.extract_patch_features(img_path)
            
            # Now get spatial dimensions
            H, W = encoder.get_spatial_dims()
            
            if debug:
                print(f"  Extracted features shape: {patch_features.shape}")
                print(f"  Expected spatial dimensions: {H}x{W} = {H*W} patches")
                print(f"  Feature dtype: {patch_features.dtype}")
                print(f"  Feature range: [{patch_features.min():.4f}, {patch_features.max():.4f}]")
            
            # Compute coherence
            coherence = compute_spatial_coherence(patch_features, H, W)
            
            if debug:
                print(f"  Local sim: {coherence['local_sim']:.4f}")
                print(f"  Distant sim: {coherence['distant_sim']:.4f}")
                print(f"  SCR: {coherence['scr']:.4f}")
            
            results['images'].append({
                'image': osp.basename(img_path),
                'local_sim': coherence['local_sim'],
                'distant_sim': coherence['distant_sim'],
                'scr': coherence['scr']
            })
            
            all_local_sims.append(coherence['local_sim'])
            all_distant_sims.append(coherence['distant_sim'])
            all_scrs.append(coherence['scr'])
            
        except Exception as e:
            error_msg = f"Error processing {osp.basename(img_path)}: {str(e)}"
            print(f"\n  ❌ {error_msg}")
            if debug:
                import traceback
                traceback.print_exc()
            results['errors'].append(error_msg)
            continue
    
    # Get final spatial dimensions
    H, W = encoder.get_spatial_dims()
    
    # Aggregate results
    if all_scrs:  # Only aggregate if we have successful results
        results['aggregated'] = {
            'mean_local_sim': float(np.mean(all_local_sims)),
            'std_local_sim': float(np.std(all_local_sims)),
            'mean_distant_sim': float(np.mean(all_distant_sims)),
            'std_distant_sim': float(np.std(all_distant_sims)),
            'mean_scr': float(np.mean(all_scrs)),
            'std_scr': float(np.std(all_scrs)),
            'spatial_dims': f"{H}x{W}",
            'num_successful': len(all_scrs),
            'num_failed': len(results['errors'])
        }
    else:
        # No successful results
        results['aggregated'] = {
            'mean_local_sim': float('nan'),
            'std_local_sim': float('nan'),
            'mean_distant_sim': float('nan'),
            'std_distant_sim': float('nan'),
            'mean_scr': float('nan'),
            'std_scr': float('nan'),
            'spatial_dims': f"{H}x{W}",
            'num_successful': 0,
            'num_failed': len(results['errors'])
        }
        print(f"  ⚠️  WARNING: No successful extractions for {encoder.model_name}")
    
    return results


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    """Main evaluation pipeline."""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Enable debug mode
    DEBUG = True  # Set to True to see detailed errors
    
    # Define encoders to test
    encoders_config = [
        ('CLIP-ViT-B/16', lambda: CLIPEncoder("openai/clip-vit-base-patch16", device)),
        ('SigLIP-Base', lambda: SigLIPEncoder("google/siglip-base-patch16-224", device)),
        ('SigLIP2-Base', lambda: SigLIPEncoder("google/siglip2-base-patch16-224", device)),
        ('DINOv2-Base', lambda: DINOv2Encoder("facebook/dinov2-base", device)),
        ('AIMv2-Large', lambda: AIMv2Encoder("apple/aimv2-large-patch14-224", device)),
    ]
    
    # Get test images
    # You can modify this to use your own image directory
    image_dir = expanduser('~/VLMEvalKit/assets')
    
    # For demo, just use the test image
    # In practice, you'd want a diverse set of images
    test_images = [
        expanduser('~/VLMEvalKit/assets/022.jpg'),
    ]
    
    # You can add more images from a directory:
    # import glob
    # test_images = glob.glob(osp.join(image_dir, '*.jpg'))[:20]  # Use first 20 images
    
    print(f"\nEvaluating {len(encoders_config)} encoders on {len(test_images)} images")
    print("=" * 80)
    
    # Evaluate each encoder
    all_results = []
    
    for encoder_name, encoder_fn in encoders_config:
        print(f"\n{'='*80}")
        print(f"Evaluating: {encoder_name}")
        print(f"{'='*80}")
        
        try:
            encoder = encoder_fn()
            results = evaluate_encoder_on_images(encoder, test_images, debug=DEBUG)
            all_results.append(results)
            
            # Print summary
            agg = results['aggregated']
            print(f"\nResults for {encoder_name}:")
            print(f"  Mean Local Similarity:   {agg['mean_local_sim']:.4f} ± {agg['std_local_sim']:.4f}")
            print(f"  Mean Distant Similarity: {agg['mean_distant_sim']:.4f} ± {agg['std_distant_sim']:.4f}")
            print(f"  Mean SCR:                {agg['mean_scr']:.4f} ± {agg['std_scr']:.4f}")
            print(f"  Spatial Dims:            {agg['spatial_dims']}")
            
            # Clean up
            del encoder
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error evaluating {encoder_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_file = expanduser('~/VLMEvalKit/spatial_coherence_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Encoder':<30} {'Local Sim':<12} {'Distant Sim':<12} {'SCR':<10}")
    print("-" * 80)
    
    for result in all_results:
        encoder_name = result['encoder'].split('/')[-1][:28]
        agg = result['aggregated']
        print(f"{encoder_name:<30} {agg['mean_local_sim']:>11.4f} {agg['mean_distant_sim']:>11.4f} {agg['mean_scr']:>9.4f}")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
