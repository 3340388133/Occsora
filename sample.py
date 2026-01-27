# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import numpy as np
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse


def load_condition_vec(cond_idx: int, gt_mode_dir: str) -> np.ndarray:
    p = os.path.join(gt_mode_dir, f"i_iter_{cond_idx}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Condition file not found: {p}")
    v = np.load(p).astype(np.float32)
    # Pad/truncate to 64
    if v.shape[0] < 64:
        rep = (64 // v.shape[0]) + 1
        v = np.tile(v, rep)[:64]
    else:
        v = v[:64]
    return v


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    # y = torch.cat([y, y_null], 0)
    
    

    v = load_condition_vec(args.cond_idx, args.gt_mode_dir)  # (64,)
    print(f"Loaded condition: idx={args.cond_idx} from {args.gt_mode_dir}, shape={v.shape}, min={v.min():.4f}, mean={v.mean():.4f}, max={v.max():.4f}")
    y = np.stack([v, v], axis=0).astype(np.float32)  # (2, 64)

    

    
    
    
   
    
    
    
    y= torch.tensor(y, device=device) 


    z= torch.randn((2,128, 4, 25, 25), device=device)

    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    

    
    #z = z[:1,]
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )

    print(samples.shape)
    
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples_array = samples.cpu().numpy()

    save_path = '/root/OccSora-main/out/samples_array.npy'

    np.save(save_path, samples_array)

    print("NumPy array saved successfully at:", save_path)
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")

    parser.add_argument("--gt-mode-dir", type=str, default="/root/OccSora-main/out/gt_mode_occstats")
    parser.add_argument("--cond-idx", type=int, default=0)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=561)#71 32
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
