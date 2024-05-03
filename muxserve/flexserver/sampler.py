import numpy as np
import torch
import torch.nn as nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.sampler import _get_logits


class GreedySampler(nn.Module):

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, embedding: torch.Tensor, hidden_states: torch.Tensor,
                input_metadata: InputMetadata):

        if input_metadata.num_prompts > 0:
            assert input_metadata.num_generation_tokens == 0
            indices = np.cumsum(input_metadata.prompt_lens) - 1
            indices = torch.tensor(indices,
                                   dtype=torch.int,
                                   device=hidden_states.device)
            hidden_states = hidden_states.index_select(0, indices)

        logits = _get_logits(hidden_states, embedding, None, self.vocab_size)

        next_tokens = []
        if input_metadata.num_prompts > 0:
            num_tokens = input_metadata.num_prompts
        if input_metadata.num_generation_tokens > 0:
            num_tokens = input_metadata.num_generation_tokens
        next_tokens = torch.argmax(logits, dim=-1)
        next_tokens = next_tokens.cpu().numpy().tolist()
        next_tokens = next_tokens[:num_tokens]

        return next_tokens
