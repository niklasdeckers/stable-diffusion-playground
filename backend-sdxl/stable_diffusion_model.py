import threading

import torch
from diffusers import DiffusionPipeline


class Model:
    def __init__(self, model_path):
        # for more options, see https://github.com/huggingface/diffusers/tree/sd_xl/src/diffusers/pipelines/stable_diffusion_xl

        self.pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True,
                                                      variant="fp16")
        self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe.to("cuda")

        self.generator = torch.Generator(device="cuda").manual_seed(42)

        self.queue_lock = threading.Lock()

    def generate_images(self, text_prompt, n_samples, seed=-1):
        with self.queue_lock:
            if seed != -1:
                self.generator = self.generator.manual_seed(int(seed))

            # images=self.pipe(prompt=text_prompt, generator=self.generator, num_images_per_prompt=n_samples).images
            # it seems like using num_images_per_prompt does not really provide a runtime advantage
            images = [self.pipe(prompt=text_prompt, generator=self.generator).images[0] for i in range(n_samples)]
            return images
