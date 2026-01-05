import sys
import os
import os.path as osp
from os.path import expanduser
import torch
import numpy as np
from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.config import *
from vlmeval.smp import *
import json
from PIL import Image
from scipy.stats import wilcoxon, binom

# Import for original LLaVA loading
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    LLAVA_AVAILABLE = True
except ImportError:
    print("Warning: LLaVA library not found. Install with: pip install git+https://github.com/haotian-liu/LLaVA.git")
    LLAVA_AVAILABLE = False

PTH = osp.realpath(__file__)
IMAGE_PTH = '/home/ubuntu/VLMEvalKit/assets/022.jpg'

# Directory containing custom LLaVA models
CUSTOM_MODELS_DIR = expanduser('/home/ubuntu/custom_llava_models')

# Global PE mode control
PE_MODE = 'normal'
SHUFFLE_SEED = None


# ============================================================================
# CUSTOM MODEL LOADING (Using Original LLaVA Library)
# ============================================================================

class CustomLLaVAModel:
    """Wrapper for loading and running custom local LLaVA models using the original LLaVA library."""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize custom LLaVA model using the LLaVA library's load_pretrained_model.
        """
        if not LLAVA_AVAILABLE:
            raise ImportError("LLaVA library required. Install with: pip install git+https://github.com/haotian-liu/LLaVA.git")
        
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.device = device
        
        print(f"Loading custom model: {self.model_name}")
        print(f"  Path: {model_path}")
        
        # Use LLaVA's native loading function
        model_name = get_model_name_from_path(model_path)
        print(f"  Model name detected: {model_name}")
        
        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            print(f"  ✓ Loaded successfully using LLaVA library")
        except Exception as e:
            print(f"  Failed to load with model_base=None: {e}")
            try:
                print(f"  Trying with base model: liuhaotian/llava-v1.5-7b")
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    model_path=model_path,
                    model_base="liuhaotian/llava-v1.5-7b",
                    model_name=model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                print(f"  ✓ Loaded with base model")
            except Exception as e2:
                print(f"  Failed to load model: {e2}")
                raise RuntimeError(f"Cannot load model from {model_path}: {e2}")
        
        self.model.eval()
        
        # Determine conversation template
        if 'llama-2' in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif 'mistral' in model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif 'v1.6' in model_name.lower() or 'v1.5' in model_name.lower():
            self.conv_mode = "llava_v1"
        else:
            self.conv_mode = "llava_v1"
        
        print(f"  Using conversation mode: {self.conv_mode}")
    
    def generate(self, messages):
        """Generate response for given messages."""
        image_path = None
        text_prompt = None
        
        for msg in messages:
            if msg['type'] == 'image':
                image_path = msg['value']
            elif msg['type'] == 'text':
                text_prompt = msg['value']
        
        if image_path is None or text_prompt is None:
            return "Error: Missing image or text in messages"
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model.config)
            
            if isinstance(image_tensor, list):
                image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            
            conv = conv_templates[self.conv_mode].copy()
            
            if self.model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text_prompt
            else:
                question = DEFAULT_IMAGE_TOKEN + '\n' + text_prompt
            
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.unsqueeze(0).to(self.model.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=False,
                    max_new_tokens=256,
                    use_cache=True,
                )
            
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            response = response.strip()
            if conv.sep2 and conv.sep2 in response:
                response = response.split(conv.sep2)[-1].strip()
            
            return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}"
    
    def get_vision_tower(self):
        """Return the vision tower for hook injection."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
            return self.model.model.vision_tower
        elif hasattr(self.model, 'vision_tower'):
            return self.model.vision_tower
        return None


# Cache for loaded custom models
_custom_model_cache = {}

def get_custom_model(model_name):
    """Get or load a custom model by name."""
    if model_name not in _custom_model_cache:
        model_path = os.path.join(CUSTOM_MODELS_DIR, model_name)
        if os.path.exists(model_path):
            _custom_model_cache[model_name] = CustomLLaVAModel(model_path)
        else:
            raise FileNotFoundError(f"Custom model not found: {model_path}")
    return _custom_model_cache[model_name]


# List of custom model names
CUSTOM_MODEL_NAMES = [
    'llava-v1.5-7b-finetune-aimv2',
    'llava-v1.5-7b-finetune-mrope-aimv2',
    'llava-v1.5-7b-finetune-mrope-clip',
    'llava-v1.5-7b-finetune-mrope-siglip-base-patch16-256',
    'llava-v1.5-7b-finetune-mrope-siglip2-base-patch16-256',
    'llava-v1.5-7b-finetune-siglip',
    'llava-v1.5-7b-finetune-siglip2',
]


# ============================================================================
# PE MODIFIER
# ============================================================================

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


# ============================================================================
# HOOK INJECTION
# ============================================================================

def find_and_hook_vision_encoder(model, pe_modifier, is_custom=False):
    """
    Comprehensive search for vision encoder and hook injection.
    Works for both VLMEval models and custom LLaVA models.
    """
    hooks = []
    hooked_modules = []
    
    # For custom models, get the underlying model
    if is_custom and hasattr(model, 'model'):
        actual_model = model.model
    else:
        actual_model = model
    
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
            obj = actual_model
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                vision_encoder = obj
                print(f"  ✓ Found vision encoder at: {path}")
                break
        except:
            continue
    
    if vision_encoder is None:
        print("  ✗ Could not find vision encoder via standard paths")
        # Try to find any module with 'vision' or 'visual' in name
        for name, module in actual_model.named_modules():
            if 'vision' in name.lower() or 'visual' in name.lower():
                print(f"    Found potential vision module: {name}")
                vision_encoder = module
                break
    
    if vision_encoder is None:
        print("  ✗ Warning: Could not locate vision encoder. PE modification will not work.")
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
            print(f"  ✓ Registered hook on: {name} ({type(module).__name__})")
        except Exception as e:
            print(f"  ✗ Failed to hook {name}: {e}")
    
    if not hooks:
        print("  ✗ Warning: No hooks were successfully registered")
    
    return hooks, hooked_modules


# ============================================================================
# INFERENCE WITH PE MODIFICATION
# ============================================================================

def CHECK_with_PE(model_name, msg, pe_mode='normal', seed=None):
    """Run inference with specified PE mode - works for both VLMEval and custom models"""
    
    is_custom = model_name in CUSTOM_MODEL_NAMES
    
    print(f"\n  Loading model: {model_name} ({'custom' if is_custom else 'vlmeval'})")
    
    try:
        if is_custom:
            model = get_custom_model(model_name)
            actual_model = model.model if hasattr(model, 'model') else model
        else:
            if model_name not in supported_VLM:
                return (model_name, "Model not found", pe_mode, [])
            model = supported_VLM[model_name]()
            actual_model = model
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return (model_name, f"Error loading model: {str(e)}", pe_mode, [])
    
    # Create PE modifier
    pe_modifier = PEModifier(mode=pe_mode, seed=seed)
    
    # Inject hooks
    print(f"  Injecting PE modification hooks (mode={pe_mode})...")
    hooks, hooked_modules = find_and_hook_vision_encoder(actual_model, pe_modifier, is_custom=is_custom)
    
    if not hooks:
        print(f"  ⚠ WARNING: No hooks injected - results may not reflect PE modifications!")
    
    try:
        print(f"  Running inference...")
        res = model.generate(msg)
        result = (model_name, res, pe_mode, hooked_modules)
    except Exception as e:
        print(f"  ✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        result = (model_name, f"Error: {str(e)}", pe_mode, hooked_modules)
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
    """Compute ΔPE metric and detailed analysis with per-sample statistical tests"""
    
    metrics = {}
    
    # Extract outputs for comparison
    normal_outputs = [r[1] for r in results['normal'][0]]
    
    # For shuffled, compare across trials - use first trial for main comparison
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
    
    # Build per-sample accuracy vectors (0/1 per sample)
    # Accuracy = 1 if output matches normal, 0 otherwise
    acc_normal = np.ones(len(normal_outputs))  # Normal always matches itself
    acc_shuf = np.array([1 if o1.strip().lower() == o2.strip().lower() 
                         else 0 for o1, o2 in zip(normal_outputs, shuffled_first)])
    acc_const = np.array([1 if o1.strip().lower() == o2.strip().lower() 
                          else 0 for o1, o2 in zip(normal_outputs, constant_outputs)])
    
    # Compute per-sample token overlap scores for bootstrap CI
    def token_overlap_single(o1, o2):
        """Token-level Jaccard similarity for a single pair"""
        tokens1 = set(o1.lower().split())
        tokens2 = set(o2.lower().split())
        if not tokens1 and not tokens2:
            return 1.0
        elif not tokens1 or not tokens2:
            return 0.0
        else:
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            return intersection / union if union > 0 else 0.0
    
    # Per-sample ΔPE values (1 - token_overlap)
    delta_pe_shuf_samples = np.array([1 - token_overlap_single(o1, o2) 
                                       for o1, o2 in zip(normal_outputs, shuffled_first)])
    delta_pe_const_samples = np.array([1 - token_overlap_single(o1, o2) 
                                        for o1, o2 in zip(normal_outputs, constant_outputs)])
    
    # Compute aggregate metrics
    metrics['exact_match_shuffled'] = exact_match(normal_outputs, shuffled_first)
    metrics['exact_match_constant'] = exact_match(normal_outputs, constant_outputs)
    
    metrics['token_overlap_shuffled'] = token_overlap(normal_outputs, shuffled_first)
    metrics['token_overlap_constant'] = token_overlap(normal_outputs, constant_outputs)
    
    # ΔPE based on token overlap (more sensitive than exact match)
    metrics['delta_PE_shuffled'] = 1 - metrics['token_overlap_shuffled']
    metrics['delta_PE_constant'] = 1 - metrics['token_overlap_constant']
    
    # Bootstrap CIs for ΔPE
    ci_shuf = bootstrap_ci(delta_pe_shuf_samples)
    ci_const = bootstrap_ci(delta_pe_const_samples)
    metrics['delta_PE_shuffled_ci_lower'] = ci_shuf[0]
    metrics['delta_PE_shuffled_ci_upper'] = ci_shuf[1]
    metrics['delta_PE_constant_ci_lower'] = ci_const[0]
    metrics['delta_PE_constant_ci_upper'] = ci_const[1]
    
    # Per-sample accuracies (for statistical tests)
    metrics['acc_normal'] = acc_normal.tolist()
    metrics['acc_shuf'] = acc_shuf.tolist()
    metrics['acc_const'] = acc_const.tolist()
    
    # Statistical tests
    try:
        # Test: normal vs shuffled (Wilcoxon signed-rank)
        if len(acc_normal) == len(acc_shuf) and len(acc_normal) > 0:
            stat_shuf, p_shuf = wilcoxon(acc_normal, acc_shuf, alternative='two-sided')
            metrics['wilcoxon_stat_shuf'] = float(stat_shuf)
            metrics['wilcoxon_pvalue_shuf'] = float(p_shuf)
        else:
            metrics['wilcoxon_stat_shuf'] = None
            metrics['wilcoxon_pvalue_shuf'] = None
        
        # Test: normal vs constant (Exact sign test)
        if len(acc_normal) == len(acc_const) and len(acc_normal) > 0:
            n_worse, n_total, p_const = exact_sign_test(acc_normal, acc_const)
            metrics['sign_test_n_worse_const'] = n_worse
            metrics['sign_test_n_total_const'] = n_total
            metrics['sign_test_pvalue_const'] = p_const
        else:
            metrics['sign_test_n_worse_const'] = None
            metrics['sign_test_n_total_const'] = None
            metrics['sign_test_pvalue_const'] = None
    except Exception as e:
        print(f"  ⚠️ Error in statistical tests: {str(e)}")
        metrics['wilcoxon_stat_shuf'] = None
        metrics['wilcoxon_pvalue_shuf'] = None
        metrics['sign_test_n_worse_const'] = None
        metrics['sign_test_n_total_const'] = None
        metrics['sign_test_pvalue_const'] = None
    
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
    
    return metrics


def print_pe_metrics(metrics, model_name):
    """Print detailed PE sensitivity metrics."""
    print(f"\n{'='*70}")
    print(f"PE SENSITIVITY METRICS - {model_name}")
    print(f"{'='*70}")
    print(f"\nExact Match Scores:")
    print(f"  Normal vs Shuffled: {metrics['exact_match_shuffled']:.3f}")
    print(f"  Normal vs Constant: {metrics['exact_match_constant']:.3f}")
    print(f"\nToken Overlap Scores:")
    print(f"  Normal vs Shuffled: {metrics['token_overlap_shuffled']:.3f}")
    print(f"  Normal vs Constant: {metrics['token_overlap_constant']:.3f}")
    print(f"\nΔPE (Sensitivity to PE modifications):")
    delta_s = metrics['delta_PE_shuffled']
    delta_c = metrics['delta_PE_constant']
    ci_s_lower = metrics.get('delta_PE_shuffled_ci_lower')
    ci_s_upper = metrics.get('delta_PE_shuffled_ci_upper')
    ci_c_lower = metrics.get('delta_PE_constant_ci_lower')
    ci_c_upper = metrics.get('delta_PE_constant_ci_upper')
    
    if ci_s_lower is not None and ci_s_upper is not None:
        print(f"  ΔPE(Shuffled):  {delta_s:.3f} [{ci_s_lower:.3f}, {ci_s_upper:.3f}]")
    else:
        print(f"  ΔPE(Shuffled):  {delta_s:.3f}")
    
    if ci_c_lower is not None and ci_c_upper is not None:
        print(f"  ΔPE(Constant):  {delta_c:.3f} [{ci_c_lower:.3f}, {ci_c_upper:.3f}]")
    else:
        print(f"  ΔPE(Constant):  {delta_c:.3f}")
    
    # Statistical tests
    print(f"\nStatistical Tests:")
    p_shuf = metrics.get('wilcoxon_pvalue_shuf')
    p_const = metrics.get('sign_test_pvalue_const')
    n_worse = metrics.get('sign_test_n_worse_const')
    n_total = metrics.get('sign_test_n_total_const')
    
    if p_shuf is not None:
        p_str_shuf = format_p_value(p_shuf)
        print(f"  Normal vs Shuffled (Wilcoxon): p = {p_str_shuf}")
    else:
        print(f"  Normal vs Shuffled (Wilcoxon): p = N/A")
    
    if p_const is not None and n_worse is not None and n_total is not None:
        p_str_const = format_p_value(p_const)
        print(f"  Normal vs Constant (Exact sign test): {n_worse}/{n_total} worse, p = {p_str_const}")
    else:
        print(f"  Normal vs Constant (Exact sign test): p = N/A")
    
    print(f"\nShuffled Trial Consistency: {metrics['shuffled_trial_consistency']:.3f}")


def format_p_value(p_value):
    """Format p-value for display."""
    if p_value is None:
        return "N/A"
    if p_value < 0.001:
        return "<0.001"
    elif p_value < 0.01:
        return "<0.01"
    elif p_value < 0.05:
        return "<0.05"
    else:
        return f"{p_value:.3f}"


def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Array-like of values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (None, None)
    
    data = np.array(data)
    n = len(data)
    bootstrap_means = []
    
    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return (float(lower_bound), float(upper_bound))


def exact_sign_test(acc_normal, acc_const):
    """
    Exact sign test for binary outcomes.
    Tests H0: P(worse under Const) = 0.5 vs H1: P(worse under Const) > 0.5
    
    Args:
        acc_normal: Array of accuracies under normal PE (0/1)
        acc_const: Array of accuracies under constant PE (0/1)
    
    Returns:
        (n_worse, n_total, p_value)
    """
    if len(acc_normal) != len(acc_const):
        return (None, None, None)
    
    acc_normal = np.array(acc_normal)
    acc_const = np.array(acc_const)
    
    # Count samples that got worse (1 -> 0) under Const
    worse = (acc_normal == 1) & (acc_const == 0)
    n_worse = np.sum(worse)
    n_total = len(acc_normal)
    
    # Exact binomial test: P(X >= n_worse) where X ~ Binomial(n_total, 0.5)
    # This tests if Const causes significantly more failures
    if n_total == 0:
        return (0, 0, None)
    
    # One-sided test: probability of observing n_worse or more worse cases
    p_value = 1 - binom.cdf(n_worse - 1, n_total, 0.5)
    
    return (int(n_worse), int(n_total), float(p_value))


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import gc
    
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
        ],
        [
            dict(type='image', value=IMAGE_PTH),
            dict(type='text', value="Is the bowl to the left or right of center?")
        ],
        [
            dict(type='image', value=IMAGE_PTH),
            dict(type='text', value="What is at the bottom of the image?")
        ],
    ]
    
    # ========================================================================
    # MODELS TO TEST
    # ========================================================================
    
    # VLMEval models
    vlmeval_models = ["llava_v1.5_7b"]
    
    # Custom local models
    custom_models = [
        'llava-v1.5-7b-finetune-aimv2',
        'llava-v1.5-7b-finetune-mrope-aimv2',
        'llava-v1.5-7b-finetune-mrope-clip',
        'llava-v1.5-7b-finetune-mrope-siglip-base-patch16-256',
        'llava-v1.5-7b-finetune-mrope-siglip2-base-patch16-256',
        'llava-v1.5-7b-finetune-siglip',
        'llava-v1.5-7b-finetune-siglip2',
    ]
    
    # Combine all models
    model_list = vlmeval_models + custom_models
    
    print("\n" + "="*80)
    print("PE COUNTERFACTUAL TEST - COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Models: {len(model_list)} ({len(vlmeval_models)} VLMEval + {len(custom_models)} custom)")
    print(f"  Queries: {len(spatial_queries)}")
    print(f"  PE Modes: normal, shuffled, constant")
    print(f"  Trials per shuffled: 3")
    print(f"\nCustom models directory: {CUSTOM_MODELS_DIR}")
    print(f"Test image: {IMAGE_PTH}")
    print("="*80)
    
    # Run experiments
    all_results = {}
    all_metrics = {}
    
    for model_name in model_list:
        print(f"\n{'#'*70}")
        print(f"# Testing Model: {model_name}")
        print(f"{'#'*70}")
        
        try:
            results = run_pe_experiment(
                model_name, 
                spatial_queries,
                pe_modes=['normal', 'shuffled', 'constant'],
                num_trials=3
            )
            
            metrics = compute_pe_sensitivity(results)
            print_pe_metrics(metrics, model_name)
            
            all_results[model_name] = results
            all_metrics[model_name] = metrics
            
        except Exception as e:
            print(f"✗ Error testing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_metrics[model_name] = {
                'error': str(e),
                'delta_PE_shuffled': -1,
                'delta_PE_constant': -1,
            }
        
        # Clear model cache and GPU memory between models
        if model_name in _custom_model_cache:
            del _custom_model_cache[model_name]
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    output_file = '/home/ubuntu/VLMEvalKit/pe_counterfactual_results.json'
    
    # Prepare serializable results
    serializable_results = {}
    for model, model_results in all_results.items():
        serializable_results[model] = {
            mode: [[r[1] for r in trial] for trial in trials]
            for mode, trials in model_results.items()
        }
    
    # Convert numpy types to native Python types for JSON serialization
    output_data = {
        'metrics': convert_numpy_types(all_metrics),
        'outputs': serializable_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # ========================================================================
    # SUMMARY TABLE (ICML Format)
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("SUMMARY - Cross-Model PE Sensitivity Comparison")
    print(f"{'='*80}")
    print(f"\n{'Model':<35} {'ΔPE (Shuf)':<25} {'p-value':<12} {'ΔPE (Const)':<25} {'p-value':<12}")
    print(f"{'-'*120}")
    
    for model_name in model_list:
        metrics = all_metrics.get(model_name, {})
        
        if 'error' in metrics:
            print(f"{model_name[:33]:<35} {'ERROR':<25} {'ERROR':<12} {'ERROR':<25} {'ERROR':<12}")
            continue
        
        delta_s = metrics.get('delta_PE_shuffled', -1)
        delta_c = metrics.get('delta_PE_constant', -1)
        ci_s_lower = metrics.get('delta_PE_shuffled_ci_lower')
        ci_s_upper = metrics.get('delta_PE_shuffled_ci_upper')
        ci_c_lower = metrics.get('delta_PE_constant_ci_lower')
        ci_c_upper = metrics.get('delta_PE_constant_ci_upper')
        p_shuf = metrics.get('wilcoxon_pvalue_shuf')
        p_const = metrics.get('sign_test_pvalue_const')
        
        p_str_shuf = format_p_value(p_shuf) if p_shuf is not None else "N/A"
        p_str_const = format_p_value(p_const) if p_const is not None else "N/A"
        
        if delta_s < 0 or delta_c < 0:
            print(f"{model_name[:33]:<35} {'N/A':<25} {'N/A':<12} {'N/A':<25} {'N/A':<12}")
        else:
            # Format ΔPE with CI
            if ci_s_lower is not None and ci_s_upper is not None:
                delta_s_str = f"{delta_s:.3f} [{ci_s_lower:.3f}, {ci_s_upper:.3f}]"
            else:
                delta_s_str = f"{delta_s:.3f}"
            
            if ci_c_lower is not None and ci_c_upper is not None:
                delta_c_str = f"{delta_c:.3f} [{ci_c_lower:.3f}, {ci_c_upper:.3f}]"
            else:
                delta_c_str = f"{delta_c:.3f}"
            
            print(f"{model_name[:33]:<35} {delta_s_str:<25} {p_str_shuf:<12} {delta_c_str:<25} {p_str_const:<12}")
    
    # ========================================================================
    # INTERPRETATION SECTION
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    print("\nLarge ΔPE with statistical significance indicates alignment dependence,")
    print("whereas small and non-significant ΔPE indicates representation-dominant spatial reasoning.\n")
    
    print(f"{'Model':<40} {'Sensitivity':<25} {'Statistical Support':<25}")
    print(f"{'-'*95}")
    
    for model_name in model_list:
        metrics = all_metrics.get(model_name, {})
        
        if 'error' in metrics:
            print(f"{model_name[:38]:<40} {'Error':<25} {metrics.get('error', 'Unknown')[:23]}")
            continue
        
        delta_s = metrics.get('delta_PE_shuffled', -1)
        delta_c = metrics.get('delta_PE_constant', -1)
        p_shuf = metrics.get('wilcoxon_pvalue_shuf')
        p_const = metrics.get('sign_test_pvalue_const')
        
        if delta_s < 0 or delta_c < 0:
            sensitivity = "No data"
            stat_support = "N/A"
        else:
            # Determine sensitivity level based on ΔPE magnitude
            avg_delta = (delta_s + delta_c) / 2
            if avg_delta > 0.5:
                sensitivity = "Strong sensitivity (large ΔPE)"
            elif avg_delta > 0.2:
                sensitivity = "Moderate sensitivity"
            else:
                sensitivity = "Low sensitivity"
            
            # Determine statistical support
            sig_shuf = p_shuf is not None and p_shuf < 0.05
            sig_const = p_const is not None and p_const < 0.05
            
            if sig_shuf or sig_const:
                stat_support = "Significant (p < 0.05)"
            else:
                stat_support = "Not significant"
        
        print(f"{model_name[:38]:<40} {sensitivity:<25} {stat_support:<25}")
    
    # ========================================================================
    # DETAILED COMPARISON TABLE
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("DETAILED METRICS TABLE")
    print(f"{'='*80}")
    print(f"\n{'Model':<40} {'ExactShuf':<10} {'ExactConst':<10} {'TokShuf':<10} {'TokConst':<10} {'TrialCons':<10}")
    print(f"{'-'*100}")
    
    for model_name in model_list:
        metrics = all_metrics.get(model_name, {})
        
        if 'error' in metrics:
            print(f"{model_name[:38]:<40} {'ERR':<10} {'ERR':<10} {'ERR':<10} {'ERR':<10} {'ERR':<10}")
            continue
        
        em_s = metrics.get('exact_match_shuffled', 0)
        em_c = metrics.get('exact_match_constant', 0)
        to_s = metrics.get('token_overlap_shuffled', 0)
        to_c = metrics.get('token_overlap_constant', 0)
        tc = metrics.get('shuffled_trial_consistency', 0)
        
        print(f"{model_name[:38]:<40} {em_s:<10.3f} {em_c:<10.3f} {to_s:<10.3f} {to_c:<10.3f} {tc:<10.3f}")
    
    # ========================================================================
    # RANKING BY PE ROBUSTNESS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("MODEL RANKING BY PE ROBUSTNESS (lower ΔPE = more robust)")
    print(f"{'='*80}")
    
    # Calculate average ΔPE for ranking
    model_avg_delta = []
    for model_name in model_list:
        metrics = all_metrics.get(model_name, {})
        if 'error' not in metrics:
            avg_delta = (metrics.get('delta_PE_shuffled', 1) + metrics.get('delta_PE_constant', 1)) / 2
            model_avg_delta.append((model_name, avg_delta, metrics))
    
    # Sort by average ΔPE (lower is better)
    model_avg_delta.sort(key=lambda x: x[1])
    
    print(f"\n{'Rank':<6} {'Model':<50} {'Avg ΔPE':<12} {'Status'}")
    print(f"{'-'*80}")
    
    for rank, (model_name, avg_delta, metrics) in enumerate(model_avg_delta, 1):
        if avg_delta < 0.2:
            status = "🟢 Robust"
        elif avg_delta < 0.4:
            status = "🟡 Moderate"
        else:
            status = "🔴 Sensitive"
        
        bar = "█" * int((1 - avg_delta) * 15) + "░" * (15 - int((1 - avg_delta) * 15))
        print(f"{rank:<6} {model_name[:48]:<50} {avg_delta:<12.3f} {bar} {status}")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print("="*80 + "\n")


