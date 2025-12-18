import sys
import os.path as osp
from os.path import expanduser
import torch
import numpy as np
from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.config import *
from vlmeval.smp import *
import json

PTH = osp.realpath(__file__)
IMAGE_PTH = expanduser('~/VLMEvalKit/assets/022.jpg')

# Global PE mode control
PE_MODE = 'normal'
SHUFFLE_SEED = None

class PEModifier:
    """Handles PE modification at different architectural levels"""
    
    def __init__(self, mode='normal', seed=None):
        self.mode = mode
        self.seed = seed
        self.hooks = []
    
    def modify_embeddings(self, embeddings):
        """Apply PE modification to embeddings tensor"""
        if self.mode == 'normal':
            return embeddings
        
        # Handle different tensor shapes
        if embeddings.dim() == 3:  # [batch, seq_len, hidden]
            batch_size, seq_len, hidden_dim = embeddings.shape
        elif embeddings.dim() == 2:  # [seq_len, hidden]
            embeddings = embeddings.unsqueeze(0)
            batch_size, seq_len, hidden_dim = embeddings.shape
        else:
            return embeddings
        
        if self.mode == 'shuffled':
            # Use seed for reproducibility if provided
            if self.seed is not None:
                generator = torch.Generator(device=embeddings.device)
                generator.manual_seed(self.seed)
            else:
                generator = None
            
            for b in range(batch_size):
                if generator:
                    perm = torch.randperm(seq_len, generator=generator, device=embeddings.device)
                else:
                    perm = torch.randperm(seq_len, device=embeddings.device)
                embeddings[b] = embeddings[b][perm]
        
        elif self.mode == 'constant':
            # Replace with mean (removes positional information)
            mean_embed = embeddings.mean(dim=1, keepdim=True)
            embeddings = mean_embed.expand_as(embeddings)
        
        return embeddings
    
    def create_forward_hook(self):
        """Create hook function"""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                return self.modify_embeddings(output)
            elif isinstance(output, tuple):
                modified = self.modify_embeddings(output[0])
                return (modified,) + output[1:]
            return output
        return hook_fn
    
    def create_forward_pre_hook(self):
        """Create pre-hook for modifying inputs"""
        def pre_hook_fn(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    modified = self.modify_embeddings(input[0])
                    return (modified,) + input[1:]
            return input
        return pre_hook_fn


def find_and_hook_vision_encoder(model, pe_modifier):
    """
    Comprehensive search for vision encoder and hook injection.
    Tries multiple architectural patterns.
    """
    hooks = []
    hooked_modules = []
    
    # Pattern 1: Direct model attributes
    vision_paths = [
        'model.vision_tower',
        'model.visual',
        'model.vision_model',
        'vision_tower',
        'visual',
        'vision_model',
        'model.model.vision_tower',
    ]
    
    vision_encoder = None
    for path in vision_paths:
        try:
            parts = path.split('.')
            obj = model
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                vision_encoder = obj
                print(f"✓ Found vision encoder at: {path}")
                break
        except:
            continue
    
    if vision_encoder is None:
        print("✗ Could not find vision encoder via standard paths")
        # Try to find any module with 'vision' or 'visual' in name
        for name, module in model.named_modules():
            if 'vision' in name.lower() or 'visual' in name.lower():
                print(f"  Found potential vision module: {name}")
                vision_encoder = module
                break
    
    if vision_encoder is None:
        print("✗ Warning: Could not locate vision encoder. PE modification will not work.")
        return hooks, hooked_modules
    
    # Hook injection strategies - try multiple layers
    hook_targets = []
    
    # Strategy 1: Hook embeddings/patch_embed layer (earliest)
    for attr_name in ['embeddings', 'patch_embed', 'conv1', 'patch_embedding']:
        if hasattr(vision_encoder, attr_name):
            module = getattr(vision_encoder, attr_name)
            hook_targets.append((f'vision_encoder.{attr_name}', module))
    
    # Strategy 2: Hook encoder layers
    if hasattr(vision_encoder, 'encoder'):
        encoder = vision_encoder.encoder
        if hasattr(encoder, 'layers') and len(encoder.layers) > 0:
            hook_targets.append(('vision_encoder.encoder.layers[0]', encoder.layers[0]))
    
    # Strategy 3: Hook the entire vision encoder output
    hook_targets.append(('vision_encoder', vision_encoder))
    
    # Register hooks
    for name, module in hook_targets:
        try:
            hook = module.register_forward_hook(pe_modifier.create_forward_hook())
            hooks.append(hook)
            hooked_modules.append(name)
            print(f"✓ Registered hook on: {name} ({type(module).__name__})")
        except Exception as e:
            print(f"✗ Failed to hook {name}: {e}")
    
    if not hooks:
        print("✗ Warning: No hooks were successfully registered")
    
    return hooks, hooked_modules


def CHECK_with_PE(val, msg, pe_mode='normal', seed=None):
    """Run inference with specified PE mode"""
    
    if val not in supported_VLM:
        return (val, "Model not found", pe_mode)
    
    print(f"\n  Loading model: {val}")
    model = supported_VLM[val]()
    
    # Create PE modifier
    pe_modifier = PEModifier(mode=pe_mode, seed=seed)
    
    # Inject hooks
    print(f"  Injecting PE modification hooks (mode={pe_mode})...")
    hooks, hooked_modules = find_and_hook_vision_encoder(model, pe_modifier)
    
    if not hooks:
        print(f"  ⚠ WARNING: No hooks injected - results may not reflect PE modifications!")
    
    try:
        print(f"  Running inference...")
        res = model.generate(msg)
        result = (val, res, pe_mode, hooked_modules)
    except Exception as e:
        print(f"  ✗ Error during inference: {e}")
        result = (val, f"Error: {str(e)}", pe_mode, hooked_modules)
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        print(f"  Cleaned up {len(hooks)} hooks")
    
    return result


def run_pe_experiment(model_name, messages, pe_modes=['normal', 'shuffled', 'constant'], 
                      num_trials=3):
    """Run counterfactual PE test across multiple conditions"""
    
    results = {mode: [] for mode in pe_modes}
    
    print(f"\n{'='*70}")
    print(f"Running PE Experiment on: {model_name}")
    print(f"PE Modes: {pe_modes}")
    print(f"Queries: {len(messages)}")
    print(f"Trials per shuffled mode: {num_trials}")
    print(f"{'='*70}")
    
    for mode in pe_modes:
        print(f"\n{'─'*70}")
        print(f"PE Mode: {mode.upper()}")
        print(f"{'─'*70}")
        
        # Run multiple trials for shuffled (due to randomness)
        trials = num_trials if mode == 'shuffled' else 1
        
        for trial in range(trials):
            if trials > 1:
                print(f"\n  Trial {trial + 1}/{trials}")
            
            trial_results = []
            for idx, msg in enumerate(messages):
                query_text = msg[1]['value']
                print(f"\n  Query {idx+1}: {query_text[:60]}...")
                
                # Use different seed for each trial
                seed = trial * 1000 + idx if mode == 'shuffled' else None
                result = CHECK_with_PE(model_name, msg, pe_mode=mode, seed=seed)
                
                trial_results.append(result)
                print(f"  → Output: {result[1][:100]}{'...' if len(result[1]) > 100 else ''}")
                if len(result) > 3 and result[3]:
                    print(f"  → Hooked modules: {result[3]}")
            
            results[mode].append(trial_results)
    
    return results


def compute_pe_sensitivity(results, ground_truth=None):
    """Compute ΔPE metric and detailed analysis"""
    
    metrics = {}
    
    # Extract outputs for comparison
    normal_outputs = [r[1] for r in results['normal'][0]]
    
    # For shuffled, compare across trials
    shuffled_all_trials = [[r[1] for r in trial] for trial in results['shuffled']]
    shuffled_first = shuffled_all_trials[0] if shuffled_all_trials else []
    
    constant_outputs = [r[1] for r in results['constant'][0]]
    
    def exact_match(outputs1, outputs2):
        """Exact string match (case-insensitive, stripped)"""
        if len(outputs1) != len(outputs2):
            return 0.0
        matches = sum(1 for o1, o2 in zip(outputs1, outputs2) 
                     if o1.strip().lower() == o2.strip().lower())
        return matches / len(outputs1)
    
    def token_overlap(outputs1, outputs2):
        """Token-level Jaccard similarity"""
        scores = []
        for o1, o2 in zip(outputs1, outputs2):
            tokens1 = set(o1.lower().split())
            tokens2 = set(o2.lower().split())
            if not tokens1 and not tokens2:
                scores.append(1.0)
            elif not tokens1 or not tokens2:
                scores.append(0.0)
            else:
                intersection = len(tokens1 & tokens2)
                union = len(tokens1 | tokens2)
                scores.append(intersection / union if union > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0
    
    # Compute metrics
    metrics['exact_match_shuffled'] = exact_match(normal_outputs, shuffled_first)
    metrics['exact_match_constant'] = exact_match(normal_outputs, constant_outputs)
    
    metrics['token_overlap_shuffled'] = token_overlap(normal_outputs, shuffled_first)
    metrics['token_overlap_constant'] = token_overlap(normal_outputs, constant_outputs)
    
    # ΔPE based on token overlap (more sensitive than exact match)
    metrics['delta_PE_shuffled'] = 1 - metrics['token_overlap_shuffled']
    metrics['delta_PE_constant'] = 1 - metrics['token_overlap_constant']
    
    # Inter-trial consistency for shuffled
    if len(shuffled_all_trials) > 1:
        trial_pairs = []
        for i in range(len(shuffled_all_trials)):
            for j in range(i+1, len(shuffled_all_trials)):
                consistency = token_overlap(shuffled_all_trials[i], shuffled_all_trials[j])
                trial_pairs.append(consistency)
        metrics['shuffled_trial_consistency'] = sum(trial_pairs) / len(trial_pairs) if trial_pairs else 1.0
    else:
        metrics['shuffled_trial_consistency'] = 1.0
    
    # Print detailed results
    print(f"\n{'='*70}")
    print("PE SENSITIVITY METRICS:")
    print(f"{'='*70}")
    print(f"\nExact Match Scores:")
    print(f"  Normal vs Shuffled: {metrics['exact_match_shuffled']:.3f}")
    print(f"  Normal vs Constant: {metrics['exact_match_constant']:.3f}")
    print(f"\nToken Overlap Scores:")
    print(f"  Normal vs Shuffled: {metrics['token_overlap_shuffled']:.3f}")
    print(f"  Normal vs Constant: {metrics['token_overlap_constant']:.3f}")
    print(f"\nΔPE (Sensitivity to PE modifications):")
    print(f"  ΔPE(Shuffled):  {metrics['delta_PE_shuffled']:.3f}")
    print(f"  ΔPE(Constant):  {metrics['delta_PE_constant']:.3f}")
    print(f"\nShuffled Trial Consistency: {metrics['shuffled_trial_consistency']:.3f}")
    print(f"\nInterpretation:")
    print(f"  ΔPE > 0.5:  Strong dependence on positional alignment")
    print(f"  ΔPE 0.2-0.5: Moderate PE dependence")
    print(f"  ΔPE < 0.2:  Robust spatial representations (PE-invariant)")
    print(f"{'='*70}\n")
    
    # Detailed per-query breakdown
    print(f"{'='*70}")
    print("Per-Query Analysis:")
    print(f"{'='*70}")
    for idx in range(len(normal_outputs)):
        print(f"\nQuery {idx+1}:")
        print(f"  Normal:   {normal_outputs[idx][:80]}")
        print(f"  Shuffled: {shuffled_first[idx][:80] if idx < len(shuffled_first) else 'N/A'}")
        print(f"  Constant: {constant_outputs[idx][:80] if idx < len(constant_outputs) else 'N/A'}")
    
    return metrics


if __name__ == "__main__":
    # Define spatial test queries
    spatial_queries = [
        [
            dict(type='image', value=IMAGE_PTH),
            dict(type='text', value="Are the chop sticks to the left or right of the bowl?")
        ],
        [
            dict(type='image', value=IMAGE_PTH),
            dict(type='text', value="What object is closest to the top of the image?")
        ],
        [
            dict(type='image', value=IMAGE_PTH),
            dict(type='text', value="Describe the spatial arrangement of objects.")
        ]
    ]
    
    # Models to test
    model_list = [
        "llava_v1.5_7b",
        # Add more models as needed
    ]
    
    # Run experiment
    all_results = {}
    all_metrics = {}
    
    for model_name in model_list:
        print(f"\n{'#'*70}")
        print(f"# Testing Model: {model_name}")
        print(f"{'#'*70}")
        
        results = run_pe_experiment(
            model_name, 
            spatial_queries,
            pe_modes=['normal', 'shuffled', 'constant'],
            num_trials=3
        )
        
        metrics = compute_pe_sensitivity(results)
        
        all_results[model_name] = results
        all_metrics[model_name] = metrics
    
    # Save results
    output_file = 'pe_counterfactual_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'metrics': all_metrics,
            'outputs': {
                model: {
                    mode: [[r[1] for r in trial] for trial in trials]
                    for mode, trials in model_results.items()
                }
                for model, model_results in all_results.items()
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY - Cross-Model Comparison:")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'ΔPE(Shuf)':<12} {'ΔPE(Const)':<12} {'Interpretation'}")
    print(f"{'-'*70}")
    for model, metrics in all_metrics.items():
        delta_s = metrics['delta_PE_shuffled']
        delta_c = metrics['delta_PE_constant']
        
        if delta_s > 0.5 or delta_c > 0.5:
            interp = "Alignment-dependent"
        elif delta_s < 0.2 and delta_c < 0.2:
            interp = "Robust (PE-invariant)"
        else:
            interp = "Mixed"
        
        print(f"{model:<30} {delta_s:<12.3f} {delta_c:<12.3f} {interp}")
    print(f"{'='*70}\n")
