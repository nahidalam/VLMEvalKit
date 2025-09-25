import torch
import torch.nn as nn
import re

class StableMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # This __init__ is PERFECT. It ensures the weights load correctly.
        self.add_module("0", nn.Linear(config.mm_hidden_size, config.hidden_size))
        self.add_module("1", nn.GELU())
        self.add_module("2", nn.Linear(config.hidden_size, config.hidden_size))

    def forward(self, x):
        # Store the original dtype (e.g., float16) to cast back to at the end.
        input_dtype = x.dtype
        
        # --- 1. Input Tensor Stats ---
        # print("\n--- Projector Input (x) ---")
        # print(f"  - Dtype: {x.dtype}")
        # print(f"  - Max: {x.max().item():.4f}, Min: {x.min().item():.4f}, Mean: {x.mean().item():.6f}")

        # --- THE DEFINITIVE FIX ---
        # Temporarily cast the layers to float32.
        layer_0_fp32 = self._modules["0"].to(torch.float32)
        layer_1_fp32 = self._modules["1"].to(torch.float32)
        layer_2_fp32 = self._modules["2"].to(torch.float32)
        
        # Upcast the input to float32 as well.
        x_fp32 = x.to(torch.float32)
        
        # --- 2. Input Tensor after Upcasting to float32 ---
        # print("\n--- Projector Input (x_fp32) ---")
        # print(f"  - Dtype: {x_fp32.dtype}")
        # print(f"  - Max: {x_fp32.max().item():.4f}, Min: {x_fp32.min().item():.4f}, Mean: {x_fp32.mean().item():.6f}")
        
        # Perform ALL projector operations in full 32-bit precision.
        hidden_states = layer_0_fp32(x_fp32)
        
        # --- 3. Hidden States after First Linear Layer ---
        # print("\n--- Hidden States after Linear_1 (in float32) ---")
        # print(f"  - Dtype: {hidden_states.dtype}")
        # print(f"  - Max: {hidden_states.max().item():.4f}, Min: {hidden_states.min().item():.4f}, Mean: {hidden_states.mean().item():.6f}")
        
        hidden_states = layer_1_fp32(hidden_states)
        
        # --- 4. Hidden States after GELU Activation ---
        # print("\n--- Hidden States after GELU (in float32) ---")
        # print(f"  - Dtype: {hidden_states.dtype}")
        # print(f"  - Max: {hidden_states.max().item():.4f}, Min: {hidden_states.min().item():.4f}, Mean: {hidden_states.mean().item():.6f}")
        
        hidden_states = layer_2_fp32(hidden_states)
        
        # --- 5. Hidden States after Second Linear Layer (Final float32 output) ---
        # print("\n--- Hidden States after Linear_2 (in float32) ---")
        # print(f"  - Dtype: {hidden_states.dtype}")
        # print(f"  - Max: {hidden_states.max().item():.4f}, Min: {hidden_states.min().item():.4f}, Mean: {hidden_states.mean().item():.6f}")
        
        # Optional: Clamp before casting if you are forced to use float16 and still see overflow
        finfo = torch.finfo(input_dtype)
        clamped_hidden_states = hidden_states.clamp(min=finfo.min, max=finfo.max)

        # Cast the final, stable result back down to the original dtype
        final_output = clamped_hidden_states.to(input_dtype) # Using clamped version for max safety
        
        # --- 6. Final Output after Downcasting ---
        # print("\n--- Projector Final Output (casted back to original dtype) ---")
        # print(f"  - Dtype: {final_output.dtype}")
        # print(f"  - Max: {final_output.max().item():.4f}, Min: {final_output.min().item():.4f}, Mean: {final_output.mean().item():.6f}")
        
        return final_output
# --- END: Final MLP Class ---



class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        # # --- START: Modification to use the instrumented class ---
        # # We specifically intercept the case for the 2-layer MLP.
        if mlp_depth == 2:
            return StableMLP(config)
        # # --- END: Modification ---
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
