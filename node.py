# Ported from
# https://huggingface.co/ByteDance/Hyper-SD/blob/main/comfyui/ComfyUI-HyperSDXL1StepUnetScheduler/node.py

import comfy.samplers
import comfy.sample
from comfy.k_diffusion import sampling as k_diffusion_sampling
import latent_preview
import torch
import comfy.utils


class HyperSDXL1StepUnetScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps):
        timesteps = torch.tensor([800])
        sigmas = model.model.model_sampling.sigma(timesteps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return (sigmas,)


NODE_CLASS_MAPPINGS = {
    "HyperSDXL1StepUnetScheduler": HyperSDXL1StepUnetScheduler,
}
