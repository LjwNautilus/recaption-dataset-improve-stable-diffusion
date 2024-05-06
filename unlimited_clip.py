import torch
from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                          CLIPTextModelWithProjection, CLIPTokenizerFast)

class UnlimitedCLIPTextEmbedder:
    CONCAT = 0
    MEAN = 1
    TRUNCATE = 2

    def __init__(self, tokenizer, clip_text_model, chunk_length=75):
        super().__init__()
        self.chunk_length = chunk_length

        # For SDXL with 2 CLIPs.
        if not isinstance(tokenizer, (list, tuple)):
            tokenizer = (tokenizer,)
        if not isinstance(clip_text_model, (list, tuple)):
            clip_text_model = (clip_text_model,)
        assert len(tokenizer) == len(clip_text_model)
        self.tokenizer = tokenizer
        self.text_model = clip_text_model

    def process_prompts(self, prompts: 'str | list[str]', tokenizer):
        vocab = tokenizer.get_vocab()
        punctuation_token = {
            vocab.get(',</w>', None), vocab.get('.</w>', None),
            vocab.get('"</w>', None), vocab.get("'</w>", None),
            vocab.get(':</w>', None), vocab.get(';</w>', None),
            vocab.get('!</w>', None), vocab.get('?</w>', None),
        }

        input_ids = tokenizer(prompts, truncation=False, padding=False,
                              add_special_tokens=False)['input_ids']
        if len(input_ids) == 0 or isinstance(input_ids[0], int):
            input_ids = (input_ids,)

        chunk = []
        chunk_list: 'list[list]' = []
        batch_chunk_list: 'list[list[list]]' = []
        last_punctuation = -1

        def next_chunk():
            '''Pads the current chunk and adds special tokens,
            adds it to the chunk list, then prepare for the next new chunk.'''

            nonlocal last_punctuation
            nonlocal chunk

            # Pads the current chunk to 75 tokens.
            chunk += [tokenizer.pad_token_id] \
                     * max(0, self.chunk_length - len(chunk))
            # Adds <bos> & <eos> to the chunk.
            chunk = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]

            last_punctuation = -1
            chunk_list.append(chunk)
            chunk = []

        for tokens in input_ids:
            tokens_len = len(tokens)
            if tokens_len <= self.chunk_length:
                tokens += [tokenizer.pad_token_id] \
                          * max(0, self.chunk_length - tokens_len)
                tokens = [tokenizer.bos_token_id] + tokens \
                         + [tokenizer.eos_token_id]
                batch_chunk_list.append((tokens,))
                last_punctuation = -1
                continue
            for token in tokens:
                if token in punctuation_token:
                    last_punctuation = len(chunk)
                elif len(chunk) == self.chunk_length and last_punctuation != -1:
                    break_location = last_punctuation + 1
                    relocation_tokens = chunk[break_location:]
                    chunk = chunk[:break_location]
                    next_chunk()
                    chunk = relocation_tokens
                # Go to the next chunk when the chunk is full.
                if len(chunk) == self.chunk_length:
                    next_chunk()
                chunk.append(token)
            if chunk or not chunk_list:
                next_chunk()
            batch_chunk_list.append(chunk_list)
            chunk_list = []

        return batch_chunk_list
    
    # Modifies from compel.Compel.pad_conditioning_tensors_to_same_length
    def pad_tensors_to_same_length(self, conditionings):
        emptystring_conditioning = self("")
        c0_shape = conditionings[0].shape
        if not all([len(c.shape) == len(c0_shape) for c in conditionings]):
            raise ValueError("Conditioning tensors must all have either 2 dimensions (unbatched) or 3 dimensions (batched)")

        if len(c0_shape) == 2:
            # need to be unsqueezed
            conditionings = [c.unsqueeze(0) for c in conditionings]
            c0_shape = conditionings[0].shape
        if len(c0_shape) != 3:
            raise ValueError(f"All conditioning tensors must have the same number of dimensions (2 or 3)")

        if not all([c.shape[0] == c0_shape[0] and c.shape[2] == c0_shape[2]
                    for c in conditionings]):
            raise ValueError(f"All conditioning tensors must have the same batch size ({c0_shape[0]}) and number of embeddings per token ({c0_shape[1]}")
        
        if len(emptystring_conditioning.shape) == 2:
            emptystring_conditioning = emptystring_conditioning.unsqueeze(0)
        empty_z = torch.cat([emptystring_conditioning] * c0_shape[0])
        max_token_count = max([c.shape[1] for c in conditionings])
        # if necessary, pad shorter tensors out with an emptystring tensor
        for i, c in enumerate(conditionings):
            while c.shape[1] < max_token_count:
                c = torch.cat([c, empty_z], dim=1)
                conditionings[i] = c
        return conditionings
    
    def get_embedding(self, tokenizer, text_model,
                      prompts: 'str | list[str]', mode=CONCAT):
        batch_chunk_list = self.process_prompts(prompts, tokenizer)
        batch_embeddings = [None] * len(batch_chunk_list)
        for i, chunk_list in enumerate(batch_chunk_list):
            if mode == self.TRUNCATE:
                chunk_list = (chunk_list[0],)
            chunk_tensor_list = \
                tuple(torch.tensor(chunk).to(self.text_model[0].device)
                      for chunk in chunk_list)
            embeddings = text_model(torch.stack(chunk_tensor_list))[0]
            if mode == self.CONCAT:
                batch_embeddings[i] = embeddings.flatten(end_dim=1)
            else:
                batch_embeddings[i] = embeddings.mean(dim=0)
        return torch.stack(batch_embeddings)
    
    def __call__(self, prompts: 'str | list[str]', mode=CONCAT):
        assert mode >= self.CONCAT and mode <= self.TRUNCATE
        batch_embeddings = tuple(self.get_embedding(self.tokenizer[i],
                                                    self.text_model[i],
                                                    prompts, mode)
                                 for i in range(len(self.tokenizer)))
        return torch.cat(batch_embeddings, dim=-1)

class UnlimitedCLIPScoreMetric:
    def __init__(self, model_path, device='cuda'):
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_path)
        self.text_model = \
            CLIPTextModelWithProjection.from_pretrained(model_path).to(device)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)
        self.visual_model = \
            CLIPVisionModelWithProjection.from_pretrained(model_path).to(device)
        
        self.text_embedder = \
            UnlimitedCLIPTextEmbedder(self.tokenizer, self.text_model)
        self.device = device
    
    @torch.inference_mode()
    def get_image_embedding(self, image) -> torch.Tensor:
        processed_img = self.image_processor(image, return_tensors='pt')
        pixel_values = processed_img['pixel_values'].to(self.device)
        return self.visual_model(pixel_values)[0]
    
    @torch.inference_mode()
    def get_text_embedding(self, text, mode) -> torch.Tensor:
        return self.text_embedder(text, mode)
    
    @torch.inference_mode()
    def __call__(self, image, text: 'str | list[str]',
                 mode=UnlimitedCLIPTextEmbedder.MEAN,
                 compute_sim_matrix=False) -> torch.Tensor:
        image_embedding = self.get_image_embedding(image).to(self.device)
        text_embedding = self.get_text_embedding(text, mode).to(self.device)
        # normalized features
        image_embedding = \
            image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
        text_embedding = \
            text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)
        if compute_sim_matrix:
            # Computes all-to-all similarity.
            # Returns (img_batch_size, text_batch_size).
            similarity = torch.matmul(image_embedding, text_embedding.t())
        else:
            # Computes one-to-one similarity.
            # Number of images and texts should be the same.
            # Returns (batch_size,)
            if image_embedding.shape[0] != text_embedding.shape[0]:
                raise ValueError('Expected the number of images '
                                 'and text examples to be the same')
            similarity = (image_embedding * text_embedding).sum(axis=-1)
        return torch.maximum(100 * similarity, torch.zeros_like(similarity[0]))