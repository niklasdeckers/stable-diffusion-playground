import collections
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.models.diffusion.plms import PLMSSampler

from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class Model:
    def __init__(self, model_path):
        seed_everything(42)
        opt = {"H": 512,
               "W": 512,
               "C": 4,
               "f": 8,
               "ddim_steps": 50,
               "ddim_eta": 0.0,
               "scale": 7.5,
               "precision": "autocast",
               "fixed_code": True,
               "n_iter": 1
               }
        self.opt = collections.namedtuple("opt", opt.keys())(**opt)
        config = OmegaConf.load("v1-inference.yaml")
        model = load_model_from_config(config, model_path)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.sampler = PLMSSampler(model)
        self.queue_lock = threading.Lock()

    def generate_images(self, text_prompt, n_samples):
        # from https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/scripts/txt2img.py
        with self.queue_lock:
            start_code = None
            if self.opt.fixed_code:
                start_code = torch.randn(
                    [n_samples, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f],
                    device=self.device)

            precision_scope = autocast if self.opt.precision == "autocast" else nullcontext
            with torch.no_grad():
                with precision_scope("cuda"):
                    with self.model.ema_scope():
                        all_samples = list()
                        for n in range(self.opt.n_iter):  # sampling
                            uc = None
                            if self.opt.scale != 1.0:
                                uc = self.model.get_learned_conditioning(n_samples * [""])
                            prompts = [text_prompt] * n_samples
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f]
                            samples_ddim, _ = self.sampler.sample(S=self.opt.ddim_steps,
                                                                  conditioning=c,
                                                                  batch_size=n_samples,
                                                                  shape=shape,
                                                                  verbose=False,
                                                                  unconditional_guidance_scale=self.opt.scale,
                                                                  unconditional_conditioning=uc,
                                                                  eta=self.opt.ddim_eta,
                                                                  x_T=start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            for each in x_samples_ddim:
                                all_samples.append(Image.fromarray(np.asarray(each * 255, dtype=np.uint8)))

            return all_samples
