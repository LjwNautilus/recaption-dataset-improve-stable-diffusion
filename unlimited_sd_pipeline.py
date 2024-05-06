from diffusers import StableDiffusionPipeline
import torch
from tqdm.autonotebook import tqdm
from rich.progress import track

from unlimited_clip import UnlimitedCLIPTextEmbedder

class UnlimitedSDPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         safety_checker, feature_extractor,
                         image_encoder, requires_safety_checker)
        self.unlimited_text_encoder = \
            UnlimitedCLIPTextEmbedder(tokenizer, text_encoder)
        self.disable_progress = False
        
    def progress_bar(self, iterable=None, total=None):
        return tqdm(iterable, total=total, leave=False,
                    disable=self.disable_progress)
        
    @torch.no_grad()
    def __call__(self, prompt="", negative_prompt="",
                 prompt_embeds=None, negative_prompt_embeds=None,
                 clip_mode=UnlimitedCLIPTextEmbedder.CONCAT,
                 *args, **kwargs):
        if (isinstance(prompt, (list, tuple))
            and isinstance(negative_prompt, str)):
            negative_prompt = (negative_prompt,) * len(prompt)
        if (isinstance(negative_prompt, (list, tuple))
            and isinstance(prompt, str)):
            prompt = (prompt,) * len(negative_prompt)

        if prompt_embeds is None:
            prompt_embeds = self.unlimited_text_encoder(prompt,
                                                        mode=clip_mode)
        if negative_prompt_embeds is None:
            negative_prompt_embeds = \
                self.unlimited_text_encoder(negative_prompt, mode=clip_mode)

        conditions = [prompt_embeds, negative_prompt_embeds]
        prompt_embeds, negative_prompt_embeds = \
            self.unlimited_text_encoder.pad_tensors_to_same_length(conditions)

        return super().__call__(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            *args, **kwargs
        )